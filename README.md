# Training Diverse High-Dimensional Controllers by Scaling Covariance Matrix Adaptation MAP-Annealing

This is the official implementation of "Training Diverse High-Dimensional
Controllers by Scaling Covariance Matrix Adaptation MAP-Annealing" by
[Bryon Tjanaka](https://btjanaka.net),
[Matthew C. Fontaine](https://scholar.google.com/citations?user=RqSvzikAAAAJ),
[David H. Lee](https://www.linkedin.com/in/lee-david-haolong),
[Aniruddha Kalkar](https://aniruddhakalkar.github.io), and
[Stefanos Nikolaidis](https://stefanosnikolaidis.net).

For more info, visit the following links:

- [Full Paper (arXiv)](https://arxiv.org/abs/2210.02622)
- [Supplemental Website](https://scalingcmamae.github.io)

To cite this paper, please use the following bibtex:

```bibtex
@misc{tjanaka2022training,
      title={Training Diverse High-Dimensional Controllers by Scaling Covariance Matrix Adaptation MAP-Annealing},
      author={Bryon Tjanaka and Matthew C. Fontaine and Aniruddha Kalkar and Stefanos Nikolaidis},
      year={2022},
      eprint={2210.02622},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

We primarily use the [pyribs](https://pyribs.org/) library in this
implementation. If you use this code in your research, please also cite pyribs:

```bibtex
@misc{pyribs,
  title = {pyribs: A bare-bones Python library for quality diversity
           optimization},
  author = {Bryon Tjanaka and Matthew C. Fontaine and David H. Lee and
            Yulun Zhang and Trung Tran Minh Vu and Sam Sommerer and
            Nathan Dennler and Stefanos Nikolaidis},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/icaros-usc/pyribs}},
}
```

## Contents

<!-- vim-markdown-toc GFM -->

* [Manifest](#manifest)
* [Getting Started](#getting-started)
* [Running Experiments](#running-experiments)
  * [Logging Directory Manifest](#logging-directory-manifest)
  * [Running on Slurm](#running-on-slurm)
  * [Running Locally](#running-locally)
  * [Testing Experiments](#testing-experiments)
  * [Reloading](#reloading)
  * [Robustness (AKA Corrected Metrics)](#robustness-aka-corrected-metrics)
* [Ablation on Archive Learning Rate](#ablation-on-archive-learning-rate)
* [Optimization Benchmarks](#optimization-benchmarks)
  * [Experiments](#experiments)
  * [Analysis](#analysis)
* [Results](#results)
* [Implementation](#implementation)
  * [Algorithms](#algorithms)
  * [Other Notable Files](#other-notable-files)
  * [Makefile](#makefile)
  * [Testing](#testing)
  * [Code Reading Tips](#code-reading-tips)
* [Miscellaneous](#miscellaneous)
* [License](#license)

<!-- vim-markdown-toc -->

## Manifest

- `config/`: [gin](https://github.com/google/gin-config) configuration files.
- `docs/`: Additional documentation.
- `src/`: Python implementations and related tools.
- `scripts/`: Bash scripts.
- `optimization_benchmarks/`: Benchmarks on optimization problems.

## Getting Started

1. **Clone the repo:**
   ```bash
   git clone https://github.com/icaros-usc/scaling-cma-mae.git
   ```
1. **Install Singularity:** Our primary code runs in a
   [Singularity / Apptainer](https://apptainer.org) container. See
   [here](https://apptainer.org/user-docs/master/quick_start.html) to install
   Singularity.
1. **Build or download the container:** Build the Singularity container with
   ```bash
   sudo make container.sif
   ```
   Alternatively,
   [download the container here](https://www.dropbox.com/s/bhuyj6ld48bem33/container.sif?dl=0)
   and place it in the root directory of this repo.
1. **(Optional) Install NVIDIA drivers and CUDA:** The node where the main
   script runs should have a GPU with NVIDIA drivers and CUDA installed (we have
   not included CUDA in the container). This is only necessary if you are
   running algorithms which use TD3.

## Running Experiments

There are two commands for running experiments.

1. `scripts/run_local.sh` runs on a local machine:
   ```bash
   bash scripts/run_local.sh CONFIG SEED NUM_WORKERS
   ```
   Where `CONFIG` is a [gin](https://github.com/google/gin-config) file in
   `config/`, `SEED` is a random integer seed (we used values from 1-100), and
   `NUM_WORKERS` is the number of worker processes.
2. `scripts/run_slurm.sh` runs on a SLURM cluster:
   ```bash
   bash scripts/run_slurm.sh CONFIG SEED HPC_CONFIG
   ```
   Here, `HPC_CONFIG` is the path to a config in `config/hpc`. It specifies the
   number of nodes on the cluster and the number of workers per node.

In our paper, we evaluated three CMA-MAE variants (sep-CMA-MAE, LM-MA-MAE,
OpenAI-MAE) and five baselines (CMA-MEGA (ES), CMA-MEGA (TD3, ES),
PGA-MAP-Elites, ME-ES, MAP-Elites) in four environments (QD Ant, QD
Half-Cheetah, QD Hopper, QD Walker). We have included config files for all of
these experiments. To replicate results from the paper, you will need to run
each of the following commands with different random seeds.

```bash
# QD Ant
bash scripts/run_slurm.sh config/qd_ant/sep_cma_mae.gin SEED config/hpc/100.sh
bash scripts/run_slurm.sh config/qd_ant/lm_ma_mae.gin SEED config/hpc/100.sh
bash scripts/run_slurm.sh config/qd_ant/openai_mae.gin SEED config/hpc/100.sh
bash scripts/run_slurm.sh config/qd_ant/cma_mega_es.gin SEED config/hpc/100.sh
bash scripts/run_slurm.sh config/qd_ant/cma_mega_td3_es.gin SEED config/hpc/100_gpu.sh
bash scripts/run_slurm.sh config/qd_ant/pga_me.gin SEED config/hpc/100_gpu.sh
bash scripts/run_slurm.sh config/qd_ant/me_es.gin SEED config/hpc/100_high_mem.sh
bash scripts/run_slurm.sh config/qd_ant/map_elites.gin SEED config/hpc/100.sh

# QD Half-Cheetah
bash scripts/run_slurm.sh config/qd_half_cheetah/sep_cma_mae.gin SEED config/hpc/100.sh
bash scripts/run_slurm.sh config/qd_half_cheetah/lm_ma_mae.gin SEED config/hpc/100.sh
bash scripts/run_slurm.sh config/qd_half_cheetah/openai_mae.gin SEED config/hpc/100.sh
bash scripts/run_slurm.sh config/qd_half_cheetah/cma_mega_es.gin SEED config/hpc/100.sh
bash scripts/run_slurm.sh config/qd_half_cheetah/cma_mega_td3_es.gin SEED config/hpc/100_gpu.sh
bash scripts/run_slurm.sh config/qd_half_cheetah/pga_me.gin SEED config/hpc/100_gpu.sh
bash scripts/run_slurm.sh config/qd_half_cheetah/me_es.gin SEED config/hpc/100_high_mem.sh
bash scripts/run_slurm.sh config/qd_half_cheetah/map_elites.gin SEED config/hpc/100.sh

# QD Hopper
bash scripts/run_slurm.sh config/qd_hopper/sep_cma_mae.gin SEED config/hpc/100.sh
bash scripts/run_slurm.sh config/qd_hopper/lm_ma_mae.gin SEED config/hpc/100.sh
bash scripts/run_slurm.sh config/qd_hopper/openai_mae.gin SEED config/hpc/100.sh
bash scripts/run_slurm.sh config/qd_hopper/cma_mega_es.gin SEED config/hpc/100.sh
bash scripts/run_slurm.sh config/qd_hopper/cma_mega_td3_es.gin SEED config/hpc/100_gpu.sh
bash scripts/run_slurm.sh config/qd_hopper/pga_me.gin SEED config/hpc/100_gpu.sh
bash scripts/run_slurm.sh config/qd_hopper/me_es.gin SEED config/hpc/100_high_mem.sh
bash scripts/run_slurm.sh config/qd_hopper/map_elites.gin SEED config/hpc/100.sh

# QD Walker
bash scripts/run_slurm.sh config/qd_walker/sep_cma_mae.gin SEED config/hpc/100.sh
bash scripts/run_slurm.sh config/qd_walker/lm_ma_mae.gin SEED config/hpc/100.sh
bash scripts/run_slurm.sh config/qd_walker/openai_mae.gin SEED config/hpc/100.sh
bash scripts/run_slurm.sh config/qd_walker/cma_mega_es.gin SEED config/hpc/100.sh
bash scripts/run_slurm.sh config/qd_walker/cma_mega_td3_es.gin SEED config/hpc/100_gpu.sh
bash scripts/run_slurm.sh config/qd_walker/pga_me.gin SEED config/hpc/100_gpu.sh
bash scripts/run_slurm.sh config/qd_walker/me_es.gin SEED config/hpc/100_high_mem.sh
bash scripts/run_slurm.sh config/qd_walker/map_elites.gin SEED config/hpc/100.sh
```

To run locally, replace `run_slurm.sh` with `run_local.sh` and pass a number of
workers instead of an HPC config:

```bash
bash scripts/run_local.sh config/qd_ant/sep_cma_mae.gin SEED 100
```

Regardless of whether experiments are run locally or on a cluster, all results
are placed in a _logging directory_ under `logs/`. The directory's name is of
the form `logs/%Y-%m-%d_%H-%M-%S_dashed-name`, e.g.
`logs/2020-12-01_15-00-30_experiment-1`. Refer to the
[logging directory manifest](#logging-directory-manifest) for a list of files in
the directory. `run_local.sh` and `run_slurm.sh` additionally output a separate
directory which stores the stdout of the scheduler and workers; see
[Running Locally](#running-locally) and [Running on Slurm](#running-on-slurm)
for more info.

Refer to `src/analysis/figures.py` and `src/analysis/supplemental.py` for how to
analyze results and generate figures.

The remainder of this section provides useful info for running experiments.

### Logging Directory Manifest

Each logging directory contains the following files:

```text
- config.gin                  # All experiment config variables, lumped into one file.
- seed                        # Text file containing the seed for the experiment.
- reload.pkl                  # Data necessary to reload the experiment if it fails.
- reload_td3.pkl              # Pickle data for TD3 (only applicable in some experiments).
- reload_td3.pth              # PyTorch models for TD3 (only applicable in some experiments).
- metrics.json                # Metrics like QD score; intended for MetricLogger.
- all_results.pkl             # All returns and BCs from function evaluations during the run.
- hpc_config.sh               # Same as the config in the Slurm dir, if Slurm is used.
- archive/                    # Snapshots of the full archive, including solutions and
                              # metadata, in pickle format.
- archive_history.pkl         # Stores objective values and behavior values necessary
                              # to reconstruct the archive. Solutions and metadata are
                              # excluded to save memory.
- dashboard_status.txt        # Job status which can be picked up by dashboard scripts.
                              # Only used during execution.
- local_YYYY-MM-DD_HH-MM-SS/  # Local log dir (only exists if running locally).
  - experiment.out            # Output of the experiment.
  - scheduler.out             # Output of the scheduler.
  - workers.out               # Output of the workers.
  - logdir                    # File containing the name of the main logdir.
- slurm_YYYY-MM-DD_HH-MM-SS/  # Slurm log dir (only exists if using Slurm).
                              # There can be a few of these if there were reloads.
  - config/
    - [config].sh             # Copied from `config/hpc`
  - job_ids.txt               # Job IDs; can be used to cancel job (scripts/slurm_cancel.sh).
  - logdir                    # File containing the name of the main logdir.
  - scheduler.slurm           # Slurm script for scheduler and experiment invocation.
  - scheduler.out             # stdout and stderr from running scheduler.slurm.
  - worker-{i}.slurm          # Slurm script for worker i.
  - worker-{i}.out            # stdout and stderr for worker i.
```

### Running on Slurm

> In addition to a logging directory, `run_slurm.sh` outputs a Slurm directory
> with items like the content of stdout on scheduler and workers. To move these
> into the logging directory, run `slurm_postprocess.sh` (see below).

There are a number of helpful utilities associated with Slurm scripts. These
reminders are output on the command line by `run_slurm.sh` after it executes:

- `tail -f ...` - Use this to monitor stdout and stderr of the main experiment
  script.
- `bash scripts/slurm_cancel.sh ...` - This will cancel the job.
- `ssh -N ...` - This will set up a tunnel from the HPC to your laptop so you
  can monitor the Dask dashboard. Run this on your _local_ machine.
- `bash scripts/slurm_postprocess.sh ...` - This will move the slurm logs into
  the logging directory. Run it _after_ the experiment has finished.

You can monitor the status of your slurm experiments with:

```bash
watch scripts/slurm_dashboard.sh
```

Since the dashboard output can be quite long, it can be useful to scroll through
it. For this, consider an alternative to `watch`, such as
[viddy](https://github.com/sachaos/viddy).

### Running Locally

Similar to `run_slurm.sh`, `run_local.sh` outputs a local directory with items
like the content of stdout on scheduler and workers. To move these into the
logging directory, run `local_postprocess.sh`.

You can monitor the status of your slurm experiments with:

```bash
watch scripts/local_dashboard.sh
```

### Testing Experiments

To test an experiment configuration with smaller settings, add `_test` to the
end of a name, e.g. `config/qd_ant/cma_mega_es.gin_test`. Then, the original
config (`config/qd_ant/cma_mega_es.gin`) and `config/test.gin` will be included.

### Reloading

While the experiment is running, its state is saved to "reload files" (AKA
checkpoints) in the logging directory. If the experiment fails, e.g. due to
memory limits, time limits, or network connection issues, run this command with
the name of the existing logging directory:

```bash
bash scripts/slurm_reload.sh LOGDIR
```

This will continue the job with the exact same configurations as before. For
finer-grained control, refer to the `-r` flag in `run_slurm.sh`. `run_local.sh`
also provides an option for reloading:

```bash
bash scripts/run_local.sh CONFIG SEED NUM_WORKERS LOGDIR
```

### Robustness (AKA Corrected Metrics)

Corrected metrics (we call these Robustness in our code) are computed after
experiments are completed with `src/analysis/robustness.py` through the shell
script `scripts/run_robustness_local.sh`. Running the script outputs several
files with outputs from the robustness computations in each logging directory.
Analysis of these outputs may be done with `src/analysis/figures.py` as follows:

```bash
# Aggregates logging directory data into `figure_data_robust.json`
python -m src.analysis.figures collect [YOUR_MANIFEST_FILE] --robust --output figure_data_robust.json

# We don't run `comparison` as there is no intermediate data to plot.

# Statistical analysis; output in `stats_tests_robust`
python -m src.analysis.figures tests figure_data_robust.json --output stats_tests_robust

# Table; output in `results_single_table_robust.tex`
python -m src.analysis.figures single_table figure_data_robust.json --output results_single_table_robust.tex
```

## Ablation on Archive Learning Rate

We also run an ablation where we look at how the choice of alpha affects the
performance of sep-CMA-MAE. This ablation is run in the same manner as the
experiments above, but with the following commands. Note that the default config
for sep-CMA-MAE (`sep_cma_mae.gin`) uses alpha=0.001.

```bash
# QD Ant
bash scripts/run_slurm.sh config/qd_ant/sep_cma_mae_0-0.gin SEED config/hpc/100.sh
bash scripts/run_slurm.sh config/qd_ant/sep_cma_mae.gin SEED config/hpc/100.sh
bash scripts/run_slurm.sh config/qd_ant/sep_cma_mae_0-01.gin SEED config/hpc/100.sh
bash scripts/run_slurm.sh config/qd_ant/sep_cma_mae_0-1.gin SEED config/hpc/100.sh
bash scripts/run_slurm.sh config/qd_ant/sep_cma_mae_1-0.gin SEED config/hpc/100.sh

# QD Half-Cheetah
bash scripts/run_slurm.sh config/qd_half_cheetah/sep_cma_mae_0-0.gin SEED config/hpc/100.sh
bash scripts/run_slurm.sh config/qd_half_cheetah/sep_cma_mae.gin SEED config/hpc/100.sh
bash scripts/run_slurm.sh config/qd_half_cheetah/sep_cma_mae_0-01.gin SEED config/hpc/100.sh
bash scripts/run_slurm.sh config/qd_half_cheetah/sep_cma_mae_0-1.gin SEED config/hpc/100.sh
bash scripts/run_slurm.sh config/qd_half_cheetah/sep_cma_mae_1-0.gin SEED config/hpc/100.sh

# QD Hopper
bash scripts/run_slurm.sh config/qd_hopper/sep_cma_mae_0-0.gin SEED config/hpc/100.sh
bash scripts/run_slurm.sh config/qd_hopper/sep_cma_mae.gin SEED config/hpc/100.sh
bash scripts/run_slurm.sh config/qd_hopper/sep_cma_mae_0-01.gin SEED config/hpc/100.sh
bash scripts/run_slurm.sh config/qd_hopper/sep_cma_mae_0-1.gin SEED config/hpc/100.sh
bash scripts/run_slurm.sh config/qd_hopper/sep_cma_mae_1-0.gin SEED config/hpc/100.sh

# QD Walker
bash scripts/run_slurm.sh config/qd_walker/sep_cma_mae_0-0.gin SEED config/hpc/100.sh
bash scripts/run_slurm.sh config/qd_walker/sep_cma_mae.gin SEED config/hpc/100.sh
bash scripts/run_slurm.sh config/qd_walker/sep_cma_mae_0-01.gin SEED config/hpc/100.sh
bash scripts/run_slurm.sh config/qd_walker/sep_cma_mae_0-1.gin SEED config/hpc/100.sh
bash scripts/run_slurm.sh config/qd_walker/sep_cma_mae_1-0.gin SEED config/hpc/100.sh
```

Analysis should avoid using the default names to prevent name conflicts with the
main results:

```bash
# Aggregates logging directory data into `figure_data_alpha.json`
python -m src.analysis.figures collect [YOUR_MANIFEST_FILE] --key AlphaAblation --output figure_data_alpha.json

# Plots comparison figures in the `comparison_alpha/` directory.
python -m src.analysis.figures comparison figure_data_alpha.json --output comparison_alpha

# Table; output in `results_single_table_alpha.tex`
python -m src.analysis.figures single_table figure_data_alpha.json --output results_single_table_alpha.tex
```

Robustness / corrected metrics can also be run as described earlier, with
analysis performed as:

```bash
# Aggregates logging directory data into `figure_data_alpha_robust.json`
python -m src.analysis.figures collect [YOUR_MANIFEST_FILE] --key AlphaAblation --robust --output figure_data_alpha_robust.json

# Table; output in `results_single_table_alpha_robust.tex`
python -m src.analysis.figures single_table figure_data_alpha_robust.json --notime --output results_single_table_alpha_robust.tex
```

## Optimization Benchmarks

We also perform a study to understand the performance of CMA-MAE and its
variants on low-dimensional optimization benchmarks. The source code for this
study is located in `optimization_benchmarks/` and is run in a separate Conda
environment. However, the analysis code is shared with this main repo and is run
in the same container as the main experiments of this paper. This study may be
run as follows.

### Experiments

1. Change into `optimization_benchmarks/`:
   ```bash
   cd optimization_benchmarks
   ```
1. Set up a separate Conda environment:
   ```bash
   conda create --prefix ./env python=3.8
   ```
1. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
1. Run the experiments. To replicate our paper, you will need to run 10 trials
   of CMA-MAE and its variants in both the sphere and arm domains, with both 100
   and 1000 dimensions. The exact commands are as follows.
   `experiment_parallel.py` runs multiple experiments in parallel for
   convenience.
   ```bash
   python experiment_parallel.py sphere cma_mae 100 10
   python experiment_parallel.py sphere sep_cma_mae 100 10
   python experiment_parallel.py sphere lm_ma_mae 100 10
   python experiment_parallel.py sphere openai_mae 100 10
   python experiment_parallel.py sphere cma_mae 1000 10
   python experiment_parallel.py sphere sep_cma_mae 1000 10
   python experiment_parallel.py sphere lm_ma_mae 1000 10
   python experiment_parallel.py sphere openai_mae 1000 10
   python experiment_parallel.py arm cma_mae 100 10
   python experiment_parallel.py arm sep_cma_mae 100 10
   python experiment_parallel.py arm lm_ma_mae 100 10
   python experiment_parallel.py arm openai_mae 100 10
   python experiment_parallel.py arm cma_mae 1000 10
   python experiment_parallel.py arm sep_cma_mae 1000 10
   python experiment_parallel.py arm lm_ma_mae 1000 10
   python experiment_parallel.py arm openai_mae 1000 10
   ```
   If you wish to modify the configuration for the experiments, see the `CONFIG`
   variable in `experiment.py`.

### Analysis

The analysis for the low-dimensional study is done in the main directory of this
repo within the Singularity container used in the main experiments. To perform
this analysis, first create a manifest file containing the logging directories
as described in `src/analysis/figures.py`. Then, run the following commands
within the container. To start a shell in the container, you can use
`make shell` (see the `Makefile`) or run `singularity shell container.sif`.

```bash
# Aggregates logging directory data into `figure_data_optbench.json`
python -m src.analysis.figures_optbench collect [YOUR_MANIFEST_FILE]

# Plots comparison figures in the `comparison_optbench/` directory.
python -m src.analysis.figures_optbench comparison

# Statistical analysis; output in `stats_tests_optbench`
python -m src.analysis.figures_optbench tests

# Table; output in `results_table_optbench.tex`
python -m src.analysis.figures_optbench table
```

## Results

Refer to our paper for results in each study.

## Implementation

Each experiment is structured as shown in the following diagram.
[Dask](https://dask.org) is the distributed compute library we use. When we run
an experiment, we connect to a Dask scheduler, which is in turn connected to one
or more Dask workers. Each component runs in a [Singularity](https://sylabs.io)
container.

<img alt="Experiment Layout"
     style="width: 50%; margin-left: auto; margin-right: auto; display: block;"
     src="docs/experiment_layout.svg"/>

### Algorithms

The algorithm implementations are primarily located in the following files:

- CMA-MAE variants: `src/emitters/annealing_emitter.py`, `src/emitters/opt`
- `CMA-MEGA (ES)` and `CMA-MEGA (TD3, ES)`:
  `src/emitters/gradient_improvement_emitter.py`
- `PGA-ME`: `src/emitters/pga_emitter.py`, `src/emitters/gaussian_emitter.py`
- `ME-ES`: `src/me_es/` (adapted from
  [authors' implementation](https://github.com/uber-research/Map-Elites-Evolutionary))
- `MAP-Elites`: `src/emitters/gaussian_emitter.py`

### Other Notable Files

- `src/main.py`: Entry point for all experiments.
- `src/manager.py`: Handles all experiments that are implemented with pyribs.
- `src/objectives/gym_control/`: This is the code that evaluates all solutions
  in the QDGym environments.

### Makefile

The Makefile has several useful commands. Run `make` for a full command
reference.

### Testing

There are some tests alongside the code to ensure basic correctness. To run
these, start a Singularity container with:

```bash
make shell
```

Within that container, execute:

```bash
make test
```

### Code Reading Tips

To understand the code, it will be useful to be familiar with the following
libraries:

- [pyribs](https://pyribs.org)
- [dask](https://dask.org)
- [gin](https://github.com/google/gin-config)

## Miscellaneous

- In the codebase, we refer to `behavior_values` and `BCs` (behavior
  characteristics). These are synonymous with `measures` in the paper.
- We use `PGA-ME` and `PGA-MAP-Elites` interchangeably in the code.
- We also use `iterations` and `generations` interchangeably.
- In our code (specifically `src/manager.py`), we measure `Robustness` on every
  iteration. However, this metric is only the robustness of the best-performing
  solution. Instead, we compute the Corrected Metrics described in the paper in
  a separate script (`src/analysis/robustness.py`) after experiments are
  completed.

## License

This code is released under the [MIT License](LICENSE), with the following
exceptions:

- The `src/me_es/` directory is derived from
  [Colas 2020](https://arxiv.org/pdf/2003.01825.pdf)
  ([repository](https://github.com/uber-research/Map-Elites-Evolutionary)) and
  is released under the [Uber Non-Commercial License](src/me_es/LICENSE).
- The `src/qd_gym/` directory is adapted from Olle Nilsson's
  [QDgym](https://github.com/ollenilsson19/QDgym) and is released under an
  [MIT license](src/qd_gym/LICENSE).
