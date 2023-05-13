"""Runs multiple instances of experiment.py in parallel with Dask."""
import fire
from experiment import main
from dask.distributed import Client, LocalCluster


def parallel_main(exp_name, algorithm, solution_dim, trials=10):
    """Runs multiple experiments in parallel.

    See experiment.py for most of these args.

    Args:
        trials (int): Number of experimental trials to run.
    """

    def exp_func(i):
        return main(exp_name, algorithm, solution_dim)

    # Create Dask cluster for running experiments.
    cluster = LocalCluster(
        processes=True,  # Each worker is a process.
        # Create one worker per trial (assumes >=trials cores, but okay if this
        # is not the case).
        n_workers=trials,
        threads_per_worker=1,  # Each worker process is single-threaded.
    )
    client = Client(cluster)

    # Run the experiments.
    trial_ids = list(range(trials))
    futures = client.map(exp_func, trial_ids)
    results = client.gather(futures)


if __name__ == '__main__':
    fire.Fire(parallel_main)
