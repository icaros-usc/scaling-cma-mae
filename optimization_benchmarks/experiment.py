"""Runs CMA-MAE and its variants on sphere and arm repertoire."""
import pickle
import time

import fire
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from logdir import LogDir
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler
from ribs.visualize import grid_archive_heatmap

from utils.metric_logger import MetricLogger


def sphere(solution_batch):
    """Sphere function evaluation and measures for a batch of solutions.

    Args:
        solution_batch (np.ndarray): (batch_size, dim) batch of solutions.
    Returns:
        objective_batch (np.ndarray): (batch_size,) batch of objectives.
        measures_batch (np.ndarray): (batch_size, 2) batch of measures.
    """
    dim = solution_batch.shape[1]

    # Shift the Sphere function so that the optimal value is at x_i = 2.048.
    sphere_shift = 5.12 * 0.4

    # Normalize the objective to the range [0, 100] where 100 is optimal.
    best_obj = 0.0
    worst_obj = (-5.12 - sphere_shift)**2 * dim
    raw_obj = np.sum(np.square(solution_batch - sphere_shift), axis=1)
    objective_batch = (raw_obj - worst_obj) / (best_obj - worst_obj) * 100

    # Calculate measures.
    clipped = solution_batch.copy()
    clip_mask = (clipped < -5.12) | (clipped > 5.12)
    clipped[clip_mask] = 5.12 / clipped[clip_mask]
    measures_batch = np.concatenate(
        (
            np.sum(clipped[:, :dim // 2], axis=1, keepdims=True),
            np.sum(clipped[:, dim // 2:], axis=1, keepdims=True),
        ),
        axis=1,
    )

    return objective_batch, measures_batch


def arm(solution_batch, link_lengths):
    """Returns the objective values and measures for a batch of solutions.

    Args:
        solutions (np.ndarray): A (batch_size, dim) array where each row
            contains the joint angles for the arm. `dim` will always be 12
            in this tutorial.
        link_lengths (np.ndarray): A (dim,) array with the lengths of each
            arm link (this will always be an array of ones in the tutorial).
    Returns:
        objs (np.ndarray): (batch_size,) array of objectives.
        meas (np.ndarray): (batch_size, 2) array of measures.
    """
    objective_batch = -np.var(solution_batch, axis=1)

    # Remap the objective from [-1, 0] to [0, 100]
    objective_batch = (objective_batch + 1.0) * 100.0

    # theta_1, theta_1 + theta_2, ...
    cum_theta = np.cumsum(solution_batch, axis=1)
    # l_1 * cos(theta_1), l_2 * cos(theta_1 + theta_2), ...
    x_pos = link_lengths[None] * np.cos(cum_theta)
    # l_1 * sin(theta_1), l_2 * sin(theta_1 + theta_2), ...
    y_pos = link_lengths[None] * np.sin(cum_theta)

    measures_batch = np.concatenate(
        (
            np.sum(x_pos, axis=1, keepdims=True),
            np.sum(y_pos, axis=1, keepdims=True),
        ),
        axis=1,
    )

    return objective_batch, measures_batch


CONFIG = {
    "itrs": 10_000,
    "batch_size": 40,
    "cma_mae": {
        "archive": {
            "class": GridArchive,
            "dims": (100, 100),
            "kwargs": {
                "threshold_min": 0,
                "learning_rate": 0.001,
            }
        },
        "emitters": [{
            "class": EvolutionStrategyEmitter,
            "kwargs": {
                "sigma0": 0.02,
                "ranker": "imp",
                "selection_rule": "mu",
                "restart_rule": "basic",
            },
            "num_emitters": 5,
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    },
    "sep_cma_mae": {
        "archive": {
            "class": GridArchive,
            "dims": (100, 100),
            "kwargs": {
                "threshold_min": 0,
                "learning_rate": 0.001,
            }
        },
        "emitters": [{
            "class": EvolutionStrategyEmitter,
            "kwargs": {
                "sigma0": 0.02,
                "ranker": "imp",
                "selection_rule": "mu",
                "restart_rule": "basic",
                "es": "sep_cma_es",  # Using Sep-CMA-ES.
            },
            "num_emitters": 5
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    },
    "lm_ma_mae": {
        "archive": {
            "class": GridArchive,
            "dims": (100, 100),
            "kwargs": {
                "threshold_min": 0,
                "learning_rate": 0.001,
            }
        },
        "emitters": [{
            "class": EvolutionStrategyEmitter,
            "kwargs": {
                "sigma0": 0.02,
                "ranker": "imp",
                "selection_rule": "mu",
                "restart_rule": "basic",
                "es": "lm_ma_es",  # Using LM-MA-ES.
            },
            "num_emitters": 5,
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    },
    "openai_mae": {
        "archive": {
            "class": GridArchive,
            "dims": (100, 100),
            "kwargs": {
                "threshold_min": 0,
                "learning_rate": 0.001,
            }
        },
        "emitters": [{
            "class": EvolutionStrategyEmitter,
            "kwargs": {
                "sigma0": 0.02,
                "ranker": "imp",
                "selection_rule": "mu",
                "restart_rule": "basic",
                "es": "openai_es",  # Using OpenAI-ES.
                "es_kwargs": {
                    "lr": 0.01,
                    "l2_coeff": 0.005,
                }
            },
            "num_emitters": 5,
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    },
}


def create_scheduler(algorithm, solution_dim, max_bound, seed=None):
    """Creates a scheduler based on the algorithm name.

    Args:
        algorithm (str): Name of the algorithm passed into sphere_main.
        max_bound (float): Bound of the archive. Different for sphere and arms.
        solution_dim (int): Dimensionality of sphere function.
        seed (int): Main seed or the various components.
    Returns:
        ribs.schedulers.Scheduler: A ribs scheduler for running the algorithm.
    """
    algo_config = CONFIG[algorithm]

    bounds = [(-max_bound, max_bound), (-max_bound, max_bound)]
    initial_sol = np.zeros(solution_dim)

    # Create archive.
    archive = GridArchive(solution_dim=solution_dim,
                          ranges=bounds,
                          dims=algo_config["archive"]["dims"],
                          seed=seed,
                          **algo_config["archive"]["kwargs"])

    # Create result archive.
    result_archive = GridArchive(solution_dim=solution_dim,
                                 ranges=bounds,
                                 dims=algo_config["archive"]["dims"],
                                 seed=seed)

    batch_size = algo_config[
        "batch_size"] if "batch_size" in algo_config else CONFIG["batch_size"]

    # Create emitters. Each emitter needs a different seed so that they do not
    # all do the same thing, hence we create an rng here to generate seeds. The
    # rng may be seeded with None or with a user-provided seed.
    rng = np.random.default_rng(seed)
    emitters = []
    for e in algo_config["emitters"]:
        emitter_class = e["class"]
        emitters += [
            emitter_class(archive,
                          x0=initial_sol,
                          **e["kwargs"],
                          batch_size=batch_size,
                          seed=s)
            for s in rng.integers(0, 1_000_000, e["num_emitters"])
        ]

    # Create Scheduler
    scheduler = Scheduler(archive,
                          emitters,
                          result_archive=result_archive,
                          add_mode="batch",
                          **algo_config["scheduler"]["kwargs"])
    scheduler_name = scheduler.__class__.__name__

    print(f"Create {scheduler_name} for {algorithm} with learning rate "
          f"{archive.learning_rate} and add mode batch, using solution dim "
          f"{solution_dim} and archive dims {algo_config['archive']['dims']}.")
    return scheduler


def save_heatmap(archive, heatmap_path):
    """Saves a heatmap of the archive to the given path.

    Args:
        archive (GridArchive): The archive to save.
        heatmap_path: Image path for the heatmap.
    """
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(archive, vmin=0, vmax=100)
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close(plt.gcf())


def main(exp_name,
         algorithm,
         solution_dim,
         itrs: int = None,
         log_freq=500,
         seed=None):
    """Runs experiment.

    Args:
        exp_name (str): Name of experiment (either "sphere" or "arm").
        algorithm (str): Name of the algorithm (see CONFIG).
        solution_dim (int): Dimension of the function.
        itrs (int): Overrides the iterations in CONFIG.
        log_freq (int): Number of iterations to wait before recording metrics
            and saving heatmap.
        seed (int): Seed for the algorithm. By default, there is no seed.
    """
    logdir = LogDir(f"{exp_name}_{algorithm}_{solution_dim}", uuid=True)
    print(f"Logging directory: {logdir.logdir}")

    max_bound = (solution_dim / 2 *
                 5.12 if exp_name == "sphere" else solution_dim)
    scheduler = create_scheduler(algorithm, solution_dim, max_bound, seed=seed)
    result_archive = scheduler.result_archive

    metric_list = [
        # "Interal" should be "Internal" but we leave it as is for
        # backwards compatibility.
        ("QD Score", True),
        ("Archive Coverage", True),
        ("Objective Max", False),
        ("Objective Mean", False),
        ("Iteration Algo Time", True),
        ("Iteration Interal Time", True),
        ("Iteration Process Time", True),
        ("Total Algo Time", True),
        ("Total Interal Time", True),
        ("Total Process Time", True),
    ]
    metrics = MetricLogger(metric_list)

    algo_time = 0.0  # Algorithm wall-time
    internal_time = 0.0  # Algorithm wall-time excluding evaluation
    process_time = 0.0  # Algorithm processor-time

    itrs = CONFIG["itrs"] if itrs is None else itrs

    for itr in tqdm.trange(1, itrs + 1):
        itr_start_algo = time.time()
        itr_start_process = time.process_time()

        solution_batch = scheduler.ask()

        # Evaluate fitness based on experiment.
        itr_start_eval = time.time()
        objective_batch, measure_batch = sphere(
            solution_batch) if exp_name == "sphere" else arm(
                solution_batch, np.ones(solution_dim))
        eval_time = time.time() - itr_start_eval

        scheduler.tell(objective_batch, measure_batch)

        # Update time.
        itr_algo_time = time.time() - itr_start_algo
        itr_process_time = time.process_time() - itr_start_process
        algo_time += itr_algo_time
        internal_time += itr_algo_time - eval_time
        process_time += itr_process_time

        # Logging and output (every iteration).
        # Record and display metrics.
        metrics.start_itr()
        metrics.add("QD Score", result_archive.stats.qd_score)
        metrics.add("Archive Coverage", result_archive.stats.coverage)
        metrics.add("Objective Max", result_archive.stats.obj_max)
        metrics.add("Objective Mean", result_archive.stats.obj_mean)
        metrics.add("Iteration Algo Time", itr_algo_time)
        metrics.add("Iteration Interal Time", itr_algo_time - eval_time)
        metrics.add("Iteration Process Time", itr_process_time)
        metrics.add("Total Algo Time", algo_time)
        metrics.add("Total Interal Time", internal_time)
        metrics.add("Total Process Time", process_time)
        metrics.end_itr()

        # Logging and output (occasional).
        final_itr = itr == itrs
        if itr % log_freq == 0 or final_itr:
            tqdm.tqdm.write(f"Iteration {itr} | Archive Coverage: "
                            f"{result_archive.stats.coverage * 100:.3f}% "
                            f"QD Score: {result_archive.stats.qd_score:.3f}")

    save_heatmap(result_archive, logdir.pfile("final_heatmap.png"))

    # Dump metrics to json.
    metrics.to_json(logdir.file("metrics.json"))

    # Save the final archive as Dataframe.
    with logdir.pfile("archive.csv").open("w") as file:
        result_archive.as_pandas().to_csv(file)

    # Write process and non-process time.
    with logdir.pfile("time.txt").open("w") as file:
        file.write(f"Algorithm Time: {algo_time}s\n")
        file.write(f"Process Time: {process_time}s")

    # Save scheduler.
    with logdir.pfile("scheduler.pkl", touch=True).open("wb") as file:
        pickle.dump(scheduler, file)


if __name__ == '__main__':
    fire.Fire(main)
