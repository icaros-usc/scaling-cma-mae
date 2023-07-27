"""Runs CMA-MAE variants on the Kheperax maze task.

Adapted from:
https://github.com/adaptive-intelligent-robotics/QDax/blob/b44969f94aaa70dc6e53aaed95193f65f20400c2/examples/scripts/me_example.py
"""
import pickle
import time

import fire
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.flatten_util import ravel_pytree
from kheperax.task import KheperaxConfig, KheperaxTask
from logdir import LogDir
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler
from ribs.visualize import grid_archive_heatmap
from tqdm import tqdm, trange

from utils.metric_logger import MetricLogger

CONFIG = {
    "itrs": 10_000,
    "batch_size": 40,
    "cma_mae": {
        "archive": {
            "class": GridArchive,
            "dims": (100, 100),
            "kwargs": {
                "threshold_min": 0.0,
                "learning_rate": 0.01,
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
                "threshold_min": 0.0,
                "learning_rate": 0.01,
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
                "learning_rate": 0.01,
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
                "learning_rate": 0.01,
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


def create_scheduler(algorithm, solution_dim, max_bound, x0, seed=None):
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

    bounds = max_bound
    initial_sol = x0

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


def main(algorithm, seed):
    # Init a random key
    random_key = jax.random.PRNGKey(seed)
    random_key, subkey = jax.random.split(random_key)

    # Define Task configuration
    config_kheperax = KheperaxConfig.get_default()

    config_kheperax.mlp_policy_hidden_layer_sizes = (8, 8)

    # Create Kheperax Task.
    (
        env,
        policy_network,
        scoring_fn,
    ) = KheperaxTask.create_default_task(
        config_kheperax,
        random_key=subkey,
    )

    fake_batch = jnp.zeros(shape=(1, env.observation_size))

    random_key, subkey = jax.random.split(random_key)
    example_init_parameters = policy_network.init(subkey, fake_batch)
    flattened_parameters, _array_to_pytree_fn = ravel_pytree(
        example_init_parameters)
    solution_dim = len(flattened_parameters)
    print(solution_dim)

    # # Define a metrics function
    # metrics_fn = functools.partial(
    #     default_qd_metrics,
    #     qd_offset=0.5,
    # )

    # Compute the centroids
    min_bd, max_bd = env.behavior_descriptor_limits
    bounds = [(min_bd[0], max_bd[0]), (min_bd[1], max_bd[1])]

    random_key, subkey = jax.random.split(random_key)
    scheduler = create_scheduler(algorithm,
                                 solution_dim,
                                 bounds,
                                 flattened_parameters,
                                 seed=np.asarray(subkey))
    result_archive = scheduler.result_archive

    def evaluate(params, random_key):
        params = jnp.asarray(params)
        params_pytree = jax.vmap(_array_to_pytree_fn)(params)

        random_key, subkey = jax.random.split(random_key)
        fitness, descriptor, info, _ = scoring_fn(params_pytree, subkey)

        best_fit = 0.0
        worst_fit = -0.5
        fitness = (fitness - worst_fit) / (best_fit - worst_fit) * 100

        if np.any(~np.isfinite(fitness)):
            __import__('pdb').set_trace()

        return np.asarray(fitness), np.asarray(descriptor), info

    logdir = LogDir(f"maze_{algorithm}", uuid=True)
    print(f"Logging directory: {logdir.logdir}")

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

    itrs = CONFIG["itrs"]
    for itr in trange(1, itrs + 1):
        itr_start_algo = time.time()
        itr_start_process = time.process_time()

        random_key, subkey = jax.random.split(random_key)
        solution_batch = scheduler.ask()

        # Evaluate fitness based on experiment.
        itr_start_eval = time.time()
        objective_batch, measure_batch, _ = evaluate(solution_batch, subkey)
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
        if itr % 250 == 0 or final_itr:
            tqdm.write(f"Iteration {itr} | Archive Coverage: "
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


if __name__ == "__main__":
    fire.Fire(main)
