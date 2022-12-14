"""Implementation of OpenAI ES that can be used across various emitters.

See here for more info: https://arxiv.org/abs/1703.03864
"""
import logging

import gin
import numpy as np

logger = logging.getLogger(__name__)


class Adam:
    """Adam optimizer class.

    Adapted from
    https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/optimizers.py

    Refer also to https://arxiv.org/abs/1412.6980v5 -- note there are some
    slight differences between the implementation here and the implementation in
    the paper.
    """

    def __init__(self,
                 dim,
                 stepsize,
                 l2_coeff,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-08,
                 dtype=np.float32):
        self.dtype = dtype
        self.dim = dim
        self.stepsize = stepsize
        self.l2_coeff = l2_coeff
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Parameters defined in reset().
        self.t = None
        self.m = None
        self.v = None

        self.reset()

    def reset(self):
        """Resets Adam variables."""
        self.m = np.zeros(self.dim, dtype=self.dtype)
        self.v = np.zeros(self.dim, dtype=self.dtype)
        self.t = 0

    def update(self, theta, grad):
        """Updates theta based on the given gradient.

        Returns the ratio of |step|_2 / |theta|_2 (i.e. how large the step is
        compared to theta) as well as the new theta.
        """
        # L2 regularization: https://www.fast.ai/2018/07/02/adam-weight-decay/.
        # Apparently, weight decay is a bit better than L2 regularization, but
        # prior work (OpenAI ES and ME-ES) use L2 regularization and the
        # difference seems tiny anyway.
        grad += self.l2_coeff * theta

        self.t += 1
        a = self.stepsize * np.sqrt(1 - self.beta2**self.t) / (
            1 - self.beta1**self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad * grad)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        ratio = np.linalg.norm(step) / np.linalg.norm(theta)
        return ratio, theta + step


@gin.configurable
class OpenAIEvolutionStrategy:  # pylint: disable = too-many-instance-attributes
    """OpenAI-ES optimizer for use with emitters.

    The basic usage is:
    - Initialize the optimizer and reset it.
    - Repeatedly:
      - Request new solutions with ask()
      - Rank the solutions in the emitter (better solutions come first) and pass
        them back with tell().
      - Use check_stop() to see if the optimizer has reached a stopping
        condition, and if so, call reset().

    Args:
        sigma0 (float): Initial step size.
        batch_size (int): Number of solutions to evaluate at a time. If None, we
            calculate a default batch size based on solution_dim.
        solution_dim (int): Size of the solution space.
        weight_rule (str): Not used. Kept for consistency with other emitters.
        seed (int): Seed for the random number generator.
        dtype (str or data-type): Data type of solutions.
        mirror_sampling (bool): Whether to use mirror sampling when gathering
            solutions.
        adam_learning_rate (float): Known as alpha in the adam paper.
        adam_l2_coeff (float): Coefficient for L2 regularization (see
            Adam.update).
        max_gens (int): Maximum number of generations to run this optimizer.
            None indicates no maximum.
    """

    def __init__(  # pylint: disable = too-many-arguments
            self,
            sigma0,
            batch_size,
            solution_dim,
            weight_rule,  # pylint: disable = unused-argument
            seed,
            dtype,
            # Parameters after this can only be configured with gin. Above ones
            # are passed in from the emitter.
            mirror_sampling=gin.REQUIRED,
            adam_learning_rate=gin.REQUIRED,
            adam_l2_coeff=gin.REQUIRED,
            max_gens=None):
        # This default is from CMA-ES.
        default_batch_size = 4 + int(3 * np.log(solution_dim))
        if default_batch_size % 2 == 1:
            default_batch_size += 1
        self.batch_size = (default_batch_size
                           if batch_size is None else batch_size)

        self.sigma = sigma0  # Constant, unlike in CMA.
        self.solution_dim = solution_dim
        self.dtype = dtype
        self._rng = np.random.default_rng(seed)
        self.mirror_sampling = mirror_sampling
        self.max_gens = max_gens

        assert self.batch_size > 1, \
            ("Batch size of 1 currently not supported because rank"
             "normalization does not work with batch size of 1.")
        if mirror_sampling:
            assert self.batch_size % 2 == 0, \
                "If using mirror_sampling, batch_size must be an even number."

        # Strategy-specific params -> initialized in reset().
        self.adam = Adam(self.solution_dim,
                         adam_learning_rate,
                         adam_l2_coeff,
                         dtype=self.dtype)
        self.last_update_ratio = None
        self.current_gens = None
        self.mean = None
        self.noise = None

    def reset(self, x0):
        """Resets the optimizer to start at x0.

        Args:
            x0 (np.ndarray): Initial mean.
        """
        self.adam.reset()
        self.last_update_ratio = np.inf  # Updated at end of tell().
        self.current_gens = 0
        self.mean = np.array(x0, self.dtype)
        self.noise = None  # Becomes (batch_size, solution_dim) array in ask().

    def check_stop(self, ranking_values):  # pylint: disable = no-self-use
        """Checks if the optimization should stop and be reset.

        Args:
            ranking_values (np.ndarray): Array of objective values of the
                solutions, sorted in the same order that the solutions were
                sorted when passed to tell().
        Returns:
            True if any of the stopping conditions are satisfied.
        """
        if self.max_gens is not None and self.current_gens >= self.max_gens:
            logger.info("Stop detected due to reaching max number of gens")
            return True

        # Probably never going to be reached -- max gens and no improvement (in
        # the improvement emitter) are more likely stopping conditions.
        if self.last_update_ratio < 1e-9:
            logger.info("Stop detected due to small update.")
            return True

        # Fitness is too flat (only applies if there are at least 2
        # parents).
        if (len(ranking_values) >= 2 and
                np.abs(ranking_values[0] - ranking_values[-1]) < 1e-12):
            logger.info(
                "Stop detected due to flat fitness: first is %f and last is %f",
                ranking_values[0], ranking_values[-1])
            return True

        return False

    def ask(
            self,
            lower_bounds,  # pylint: disable = unused-argument
            upper_bounds,  # pylint: disable = unused-argument
    ):
        """Samples new solutions from the Gaussian distribution.

        Note: Bounds are currently not enforced.

        Args:
            lower_bounds (float or np.ndarray): scalar or (solution_dim,) array
                indicating lower bounds of the solution space. Scalars specify
                the same bound for the entire space, while arrays specify a
                bound for each dimension. Pass -np.inf in the array or scalar to
                indicated unbounded space.
            upper_bounds (float or np.ndarray): Same as above, but for upper
                bounds (and pass np.inf instead of -np.inf).
        """
        # TODO: Implement bounds handling and remove note above.

        if self.mirror_sampling:
            logger.info("Mirror sampling")
            noise_half = self._rng.standard_normal(
                (self.batch_size // 2, self.solution_dim), dtype=self.dtype)
            self.noise = np.concatenate((noise_half, -noise_half))
            solutions = self.mean[None] + self.sigma * self.noise
        else:
            logger.info("No mirror sampling")
            self.noise = self._rng.standard_normal(
                (self.batch_size, self.solution_dim), dtype=self.dtype)
            solutions = self.mean[None] + self.sigma * self.noise
        return solutions

    def tell(
            self,
            solutions,  # pylint: disable = unused-argument
            num_parents,  # pylint: disable = unused-argument
            ranking_indices):
        """Passes the solutions back to the optimizer.

        Args:
            solutions (np.ndarray): Array of ranked solutions. Not used.
            num_parents (int): Number of best solutions to select. Not used.
            ranking_indices (list of int): Indices that were used to order
                solutions from the original solutions returned in ask().
        """
        self.current_gens += 1

        # Indices come in decreasing order, so we reverse to get them to
        # increasing order.
        assert len(ranking_indices) == self.batch_size
        ranks = np.empty(self.batch_size, dtype=np.int32)

        # Assign ranks -- ranks[i] tells the rank of noise[i].
        ranks[ranking_indices[::-1]] = np.arange(self.batch_size)

        # Normalize ranks to [-0.5, 0.5].
        ranks = (ranks / (self.batch_size - 1)) - 0.5
        logger.info("Normalized min & max rank: %f %f", np.min(ranks),
                    np.max(ranks))

        # Compute the gradient.
        logger.info("Computing gradient")
        if self.mirror_sampling:
            half_batch = self.batch_size // 2
            gradient = np.sum(
                self.noise[:half_batch] *
                (ranks[:half_batch] - ranks[half_batch:])[:, None],
                axis=0)
            gradient /= half_batch * self.sigma
        else:
            gradient = np.sum(self.noise * ranks[:, None], axis=0)
            gradient /= self.batch_size * self.sigma
        logger.info("Updating mean")
        (self.last_update_ratio,
         self.mean) = self.adam.update(self.mean, -gradient)
        logger.info("New mean [:5]: %s", self.mean[:5])
        logger.info("Update ratio: %f", self.last_update_ratio)
