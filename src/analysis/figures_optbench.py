r"""Plots figures and tables for the optimization benchmarks. Instructions are
similar to figures.py but with different filenames.

Also see the README in the root of this repo for more instructions.
"""
import itertools
import json
import pickle as pkl
import shutil
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Iterable

import fire
import gin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin
import scipy.stats
import seaborn as sns
import slugify
from logdir import LogDir
from loguru import logger
from ruamel.yaml import YAML
from statsmodels.graphics.factorplots import interaction_plot

from src.analysis.utils import (is_me_es, load_experiment, load_me_es_objs,
                                load_metrics)
from src.mpl_styles import QUALITATIVE_COLORS
from src.mpl_styles.utils import mpl_style_file

# Metrics which we record in figure_data_optbench.json but do not plot.
METRIC_BLACKLIST = [
    "Normalized QD Score",
    "Normalized QD Score AUC",
    "Wallclock Time",
    "Process Time",
]

# Reordered version of Seaborn colorblind palette.
COLORBLIND_REORDERED = np.array(
    sns.color_palette("colorblind"))[[0, 1, 8, 2, 4, 3, 5, 6, 7, 9]]


def load_manifest(manifest: str, entry: str = "Paper"):
    """Loads the data and root directory of the manifest."""
    manifest = Path(manifest)
    data = YAML().load(manifest)[entry]
    root_dir = manifest.parent
    return data, root_dir


def exclude_from_manifest(paper_data, env, algo):
    """Whether to exclude the given algo."""
    return "exclude" in paper_data[env]["algorithms"][algo]


def verify_manifest(paper_data,
                    root_dir: Path,
                    reps: int,
                    key: str,
                    check_seed: bool = True):
    """Checks logging directories for correctness."""
    for env in paper_data:
        for algo in paper_data[env]["algorithms"]:
            if exclude_from_manifest(paper_data, env, algo):
                continue

            results = paper_data[env]["algorithms"][algo]
            if results[0] == "no_old_min_obj":
                results = results[1:]
            name = f"(Env: {env} Algo: {algo})"

            if reps is not None:
                # Check that the reps are correct.
                assert len(results) == reps, \
                    f"{name} {len(results)} dirs, needs {reps}"

            # Check logging dirs are unique.
            logdirs = [d["dir"] for d in results]
            assert len(logdirs) == len(set(logdirs)), \
                f"{name} {len(logdirs)} dirs listed, {len(set(logdirs))} unique"

            if check_seed:
                # Check seeds are unique.
                seeds = [d["seed"] for d in results]
                if len(seeds) != len(set(seeds)):
                    logger.warning(f"{name} {len(seeds)} seeds listed, "
                                   f"{len(set(seeds))} unique")

                # Check seeds match.
                for d in results:
                    logdir = LogDir("tmp", custom_dir=root_dir / d["dir"])
                    actual_seed = int(logdir.pfile("seed").open("r").read())
                    assert actual_seed == d["seed"], \
                        (f"{name} Seed for {logdir} ({d['seed']}) does not "
                         f"match actual seed ({actual_seed})")

            # Check that alphas match for the ablation study:
            if key == "AlphaAblation":
                for d in results:
                    with open(root_dir / d["dir"] / "config.gin") as file:
                        for line in file:
                            if line.startswith(
                                    "MAEGridArchive.archive_learning_rate"):
                                lr = float(line.split("=")[1].strip())
                                break
                        assert lr == float(algo)


def find_min(manifest: str):
    """Finds the min obj that was inserted into the archive in any experiment.

    This minimum objective is calculated for each of the environments. It is
    then used e.g. in computing the QD score.

    Args:
        manifest: Path to YAML file holding paths to logging directories.
    """
    paper_data, root_dir = load_manifest(manifest)
    data = defaultdict(list)
    for env in paper_data:
        for algo in paper_data[env]["algorithms"]:
            if exclude_from_manifest(paper_data, env, algo):
                continue
            data[env].extend(
                [d["dir"] for d in paper_data[env]["algorithms"][algo]])

    for env in data:
        logdirs = data[env]
        min_score = np.inf
        min_logdir = None

        for d in logdirs:
            logdir = load_experiment(root_dir / d)

            if is_me_es():
                archive_history = load_me_es_objs(
                    logdir, return_archive=True).history()
            else:
                archive_type = str(gin.query_parameter("Manager.archive_type"))
                try:
                    result_archive_type = str(
                        gin.query_parameter("Manager.result_archive_type"))
                except ValueError:  # result_archive_type was not provided.
                    result_archive_type = archive_type
                if result_archive_type != "@GridArchive":
                    print(logdir.logdir)
                    raise TypeError(
                        f"Unknown archive type {result_archive_type}")

                archive_history_path = logdir.pfile("archive_history.pkl")
                with archive_history_path.open("rb") as file:
                    archive_history = pkl.load(file)

            for gen_history in archive_history:
                for obj, _ in gen_history:
                    min_score = min(min_score, obj)
                    min_logdir = root_dir / d

        logger.info("{}: {} ({})", env, min_score, min_logdir)


def collect(manifest: str,
            key: str = "Paper",
            reps: int = 10,
            output: str = "figure_data_optbench.json",
            check_seed: bool = False):
    """Collects data from logging directories for the optbench study.

    Args:
        manifest: Path to YAML file holding paths to logging directories.
        key: The key in the manifest for loading the data.
        reps: Number of times each experiment should be repeated.
        output: Path to save results.
    """
    logger.info("Loading manifest")
    paper_data, root_dir = load_manifest(manifest, key)

    logger.info("Verifying logdirs")
    verify_manifest(paper_data, root_dir, reps, key, check_seed)

    # Mapping from the name in the raw metrics to the name in the output.
    metric_names = {
        "QD Score": "QD Score",
        # "Normalized QD Score" gets added when we calculate "QD Score".
        "Archive Coverage": "Archive Coverage",
        "Objective Max": "Best Performance",
        "Total Algo Time": "Wallclock Time",
        "Total Process Time": "Process Time",
    }

    figure_data = {}

    logger.info("Loading Plot Data")
    for env in paper_data:
        figure_data[env] = OrderedDict()
        env_data = paper_data[env]

        for algo in env_data["algorithms"]:
            if exclude_from_manifest(paper_data, env, algo):
                continue

            figure_data[env][algo] = []

            entries = env_data["algorithms"][algo]
            no_old_min_obj = entries[0] == "no_old_min_obj"
            if no_old_min_obj:
                entries = entries[1:]

            for entry in entries:
                figure_data[env][algo].append(cur_data := {})
                logdir = LogDir("Placeholder",
                                custom_dir=root_dir / entry["dir"])
                metrics = load_metrics(logdir)

                # The experiments use 200 evals per iteration for 10,000
                # iterations.
                total_evals = list(range(0, 2_000_000 + 1, 200))
                cur_data["Evaluations"] = total_evals

                cur_data["series_metrics"] = {}
                qd_score_auc, norm_qd_score_auc = None, None
                for actual_name, figure_name in metric_names.items():
                    data = metrics.get_single(actual_name)

                    if actual_name == "QD Score":
                        # No need to adjust QD score in this experiment.
                        cur_data["series_metrics"][figure_name] = data["y"]

                        # Also add in normalized QD score.
                        max_qd_score = env_data["archive_size"] * (
                            env_data["max_obj"] - env_data["min_obj"])
                        norm_qd_score = (np.array(data["y"]) /
                                         max_qd_score).tolist()
                        cur_data["series_metrics"][
                            "Normalized QD Score"] = norm_qd_score

                        # Finally, add in AUCs - for this paper, it's reasonable
                        # to assume that every generation has the same number of
                        # evals.
                        evals_per_gen = total_evals[-1] / (len(total_evals) - 1)
                        qd_score_auc = sum(data["y"]) * evals_per_gen
                        norm_qd_score_auc = sum(norm_qd_score) * evals_per_gen
                    else:
                        cur_data["series_metrics"][figure_name] = data["y"]

                cur_data["point_metrics"] = {
                    "Internal Runtime (Minutes)":
                        metrics.get_single("Total Interal Time")["y"][-1] / 60,
                    "QD Score AUC":
                        qd_score_auc,
                    "Normalized QD Score AUC":
                        norm_qd_score_auc,
                }

    logger.info("Saving to {}", output)
    with open(output, "w") as file:
        json.dump(figure_data, file)
    logger.info("Done")


def legend_info(names: Iterable, palette: dict, markers: dict):
    """Creates legend handles and labels for the given palette and markers.

    Yes, this is kind of a hack.
    """
    _, ax = plt.subplots(1, 1)
    for name in names:
        # We just need the legend handles, so the plot itself is bogus.
        ax.plot(
            [0],
            [0],
            label=name,
            color=palette[name],
            marker=markers[name],
            markeredgewidth="0.75",
            markeredgecolor="white",
        )
    return ax.get_legend_handles_labels()


def load_figure_data(figure_data: str):
    with open(figure_data, "r") as file:
        return json.load(file)


def metric_from_entry(entry, metric):
    """Retrieves the metric from either series_metrics or point_metrics.

    entry is a dict in the list associated with figure_data[env][algo]
    """
    return (entry["series_metrics"][metric][-1] if metric
            in entry["series_metrics"] else entry["point_metrics"][metric])


def comparison(
    figure_data: str = "./figure_data_optbench.json",
    output: str = "comparison_optbench",
    palette_name: str = "colorblind_reordered",
    height: float = 1.9,
    plot_every: int = 25,
    sans: bool = False,
):
    """Plots the figure comparing metrics of all algorithms.

    Args:
        figure_data: Path to JSON file with figure data.
        outputs: Output directory for saving the figures.
        palette: Either a Seaborn color palette, "qualitative_colors" for
            QUALITATIVE_COLORS, or "colorblind_reordered" for
            COLORBLIND_REORDERED.
        height: Height in inches of each plot.
        plot_every: How frequently to plot points, e.g. plot every 100th point.
        sans: Pass this in to use Sans Serif fonts.
    """
    logger.info("Creating output directory")
    output = Path(output)
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir()

    figure_data = load_figure_data(figure_data)

    for qd_score_only in [True, False]:
        plot_data = {
            "Environment": [],
            "Algorithm": [],
            "Metric": [],
            "Evaluations": [],
            "Score": [],
        }

        logger.info("Loading Plot Data")
        all_algos = OrderedDict()  # Set of all algorithms, ordered by listing.
        for env in figure_data:
            cur_data = {
                "Environment": [],
                "Algorithm": [],
                "Metric": [],
                "Evaluations": [],
                "Score": [],
            }
            for algo in figure_data[env]:
                all_algos[algo] = None
                for entry in figure_data[env][algo]:
                    # Has a length of generations + 1, since we add an extra 0
                    # at the start.
                    evals = np.asarray(entry["Evaluations"])

                    entry_metrics = entry["series_metrics"]

                    # Reverse so that algorithms are ordered properly in terms
                    # of layers -- we need to reverse here so that the reverse
                    # later on works.
                    for metric in reversed(entry_metrics):
                        if qd_score_only:
                            if metric != "QD Score":
                                continue
                        else:
                            if metric in METRIC_BLACKLIST:
                                continue

                        # Plot fewer data points to reduce file size.

                        # Metrics may have length of generations or generations
                        # + 1, as only some metrics (like archive size) add a 0
                        # at the start.
                        metric_data = entry_metrics[metric]
                        raw_len = len(metric_data)
                        not_use_zero = int(len(metric_data) != len(evals))
                        gens = len(evals) - 1
                        # Start at 0 or 1 and end at gens.
                        x_data = np.arange(not_use_zero, gens + 1)

                        idx = list(range(0, raw_len, plot_every))
                        if idx[-1] != raw_len - 1:
                            # Make sure last index is included.
                            idx.append(raw_len - 1)

                        indexed_x_data = x_data[idx]
                        indexed_evals = evals[indexed_x_data]
                        indexed_metric_data = np.asarray(
                            entry_metrics[metric])[idx]
                        data_len = len(indexed_evals)

                        cur_data["Environment"].append(np.full(data_len, env))
                        cur_data["Algorithm"].append(np.full(data_len, algo))
                        cur_data["Metric"].append(np.full(data_len, metric))
                        cur_data["Evaluations"].append(indexed_evals)
                        cur_data["Score"].append(indexed_metric_data)

            # Reverse so that algorithms are ordered properly in terms of
            # layers.
            for d in plot_data:
                plot_data[d].append(np.concatenate(cur_data[d])[::-1])

        # Flatten everything so that Seaborn understands it.
        for d in plot_data:
            plot_data[d] = np.concatenate(plot_data[d])

        logger.info("Generating Plot")
        with mpl_style_file(
                "simple_sans.mplstyle" if sans else "simple.mplstyle") as f:
            with plt.style.context(f):
                if palette_name == "qualitative_colors":
                    colors = QUALITATIVE_COLORS
                elif palette_name == "colorblind_reordered":
                    # Rearrange the color-blind template.
                    colors = COLORBLIND_REORDERED
                else:
                    colors = sns.color_palette(palette_name)

                palette = dict(zip(all_algos, colors))
                markers = dict(zip(all_algos, itertools.cycle("oD^vXPps")))

                # This takes a while since it has to generate the bootstrapped
                # confidence intervals.
                grid = sns.relplot(
                    data=plot_data,
                    x="Evaluations",
                    y="Score",
                    hue="Algorithm",
                    style="Algorithm",
                    row="Metric",
                    col="Environment",
                    kind="line",
                    errorbar="se",
                    markers=markers,
                    markevery=(0.5, 10.0),
                    dashes=False,
                    # Slightly taller when only plotting QD score.
                    height=1.2 * height if qd_score_only else height,
                    aspect=1.35 if qd_score_only else 1.61803,  # Golden ratio.
                    facet_kws={"sharey": False},
                    palette=palette,
                    legend=False,
                    linewidth=1.25,
                )

                # Set titles to be the env name.
                grid.set_titles("{col_name}")

                # Turn off titles below top row (no need to repeat).
                for ax in grid.axes[1:].ravel():
                    ax.set_title("")

                limits = {
                    ("QD Score", "Sphere 100"): [0, 600_000],
                    ("QD Score", "Sphere 1000"): [0, 30_000],
                    ("QD Score", "Arm 100"): [0, 800_000],
                    ("QD Score", "Arm 1000"): [0, 800_000],
                    ("QD Score", "Maze"): [0, 800_000],
                    ("Archive Coverage", "Sphere 100"): [0, 1.0],
                    ("Archive Coverage", "Sphere 1000"): [0, 0.04],
                    ("Archive Coverage", "Arm 100"): [0, 1.0],
                    ("Archive Coverage", "Arm 1000"): [0, 1.0],
                    ("Archive Coverage", "Maze"): [0, 1.0],
                    ("Best Performance", "Sphere 100"): [92, 101],
                    ("Best Performance", "Sphere 1000"): [92, 101],
                    ("Best Performance", "Arm 100"): [92, 101],
                    ("Best Performance", "Arm 1000"): [92, 101],
                    ("Best Performance", "Maze"): [92, 101],
                }

                # Set the labels along the left column to be the name of the
                # metric.
                left_col = list(figure_data)[0]
                for (row_val, col_val), ax in grid.axes_dict.items():
                    ax.set_axisbelow(True)
                    if col_val == left_col:
                        ax.set_ylabel(row_val, labelpad=10.0)
                    else:
                        ax.set_ylabel("")

                    if (row_val, col_val) in limits:
                        ax.set_ylim(limits[(row_val, col_val)])

                    if (row_val, col_val) == ("QD Score", "QD Hopper"):
                        ax.set_yticks([0, 0.5e6, 1.0e6, 1.5e6, 2.0e6, 2.5e6])

                # Add legend and resize figure to fit it.
                grid.fig.legend(
                    *legend_info(all_algos, palette, markers),
                    bbox_to_anchor=[0.5, 1.0],
                    loc="upper center",
                    # Change // 1 to // 2 etc. for more rows.
                    ncol=(len(palette) + 1) // 1,
                )
                fig_width, fig_height = grid.fig.get_size_inches()
                legend_height = 0.30
                grid.fig.set_size_inches(fig_width, fig_height + legend_height)

                # Save the figure.
                grid.fig.tight_layout(rect=(0, 0, 1, fig_height /
                                            (fig_height + legend_height)))
                name = "comparison-qd-score" if qd_score_only else "comparison"
                for extension in ["pdf", "png", "svg"]:
                    filename = (output /
                                f"{name}{'-sans' if sans else ''}.{extension}")
                    logger.info("Saving {}", filename)
                    grid.fig.savefig(filename, dpi=300)

    logger.info("Done")


def comparison_high_res(
    figure_data: str = "./figure_data_optbench.json",
    output: str = "comparison_optbench_high_res",
):
    """Generates the larger version of the figure for the supplemental material.

    Simply calls comparison with the appropriate args.
    """
    return comparison(figure_data, output, height=4, plot_every=5)


# Header lines for table files.
TABLE_HEADER = r"""
% THIS FILE IS AUTO-GENERATED. DO NOT MODIFY THIS FILE DIRECTLY.

"""


def table(figure_data: str = "figure_data_optbench.json",
          transpose: bool = True,
          output: str = "results_table_optbench.tex"):
    """Creates Latex tables showing final values of metrics.

    Make sure to include the "booktabs" and "array" package in your Latex
    document.

    With transpose=False, a table is generated for each environment. Each table
    has the algorithms as rows and the metrics as columns.

    With transpose=True, a table is generated for each metric. Each table has
    the algorithms as rows and the environments as columns.

    Args:
        figure_data: Path to JSON file with figure data.
        transpose: See above.
        output: Path to save Latex table.
    """
    figure_data = load_figure_data(figure_data)

    # Safe to assume all envs have same metrics.
    first_env = list(figure_data)[0]
    first_algo = list(figure_data[first_env])[0]
    first_entry = figure_data[first_env][first_algo][0]
    metric_names = (list(first_entry["series_metrics"]) +
                    list(first_entry["point_metrics"]))
    #  for name in METRIC_BLACKLIST:
    #      if name in metric_names:
    #          metric_names.remove(name)
    if "QD Score AUC" in metric_names:
        # Move QD Score AUC immediately after QD Score.
        metric_names.remove("QD Score AUC")
        metric_names.insert(metric_names.index("QD Score") + 1, "QD Score AUC")
    logger.info("Metric names: {}", metric_names)

    table_data = {}
    logger.info("Gathering table data")
    for env in figure_data:
        table_data[env] = pd.DataFrame(index=list(figure_data[env]),
                                       columns=metric_names,
                                       dtype=str)
        for algo in figure_data[env]:
            for metric in metric_names:
                if metric in first_entry["series_metrics"]:
                    final_metric_vals = np.array([
                        entry["series_metrics"][metric][-1]
                        for entry in figure_data[env][algo]
                    ])
                else:
                    final_metric_vals = np.array([
                        entry["point_metrics"][metric]
                        for entry in figure_data[env][algo]
                    ])

                if metric == "QD Score AUC":
                    # Special case because these values are so large.
                    table_data[env][metric][algo] = (
                        f"{final_metric_vals.mean() / 1e12:,.2f}")
                elif metric == "QD Score":
                    table_data[env][metric][algo] = (
                        f"{final_metric_vals.mean() / 1e6:,.3f}")
                else:
                    table_data[env][metric][algo] = (
                        f"{final_metric_vals.mean():,.2f}")

    if transpose:
        # "Invert" table_data.
        table_data = {
            metric:
                pd.DataFrame({
                    env: df[metric] for env, df in table_data.items()
                }) for metric in metric_names
        }

    logger.info("Writing to {}", output)
    with open(output, "w") as file:
        file.write(TABLE_HEADER)
        for name, df in table_data.items():
            if name == "QD Score AUC":
                caption = name + " (multiple of $10^{12}$)"
            elif name == "QD Score":
                caption = name + " (multiple of $10^{6}$)"
            else:
                caption = name

            file.write("\\begin{table*}[t]\n")
            file.write("\\caption{" + caption + "}\n")
            file.write("\\label{table:" + slugify.slugify(name) + "}\n")
            file.write("\\begin{center}\n")
            file.write(
                df.to_latex(
                    column_format="l" + " R{0.9in}" * len(df.columns),
                    escape=False,
                ))
            file.write("\\end{center}\n")
            file.write("\\end{table*}\n")
            file.write("\n")

    logger.info("Done")


def calc_simple_main_effects(figure_data, anova_res, metric):
    """Calculates simple main effects in each environment.

    Reference:
    http://www.cee.uma.pt/ron/Discovering%20Statistics%20Using%20SPSS,%20Second%20Edition%20CD-ROM/Calculating%20Simple%20Effects.pdf
    """
    df_residual = anova_res["DF"][3]
    ms_residual = anova_res["MS"][3]

    data = {
        "Environment": ["Residual"],
        "SS": [anova_res["SS"][3]],
        "DF": [df_residual],
        "MS": [ms_residual],
        "F": [np.nan],
        "p-unc": [np.nan],
        "significant": [False],
    }

    for env in figure_data:
        data["Environment"].append(env)

        algos, metric_vals = [], []
        for algo in figure_data[env]:
            entry_metrics = [
                metric_from_entry(entry, metric)
                for entry in figure_data[env][algo]
            ]

            algos.extend([algo] * len(entry_metrics))
            metric_vals.extend(entry_metrics)

        one_way = pingouin.anova(
            dv=metric,
            between=["Algorithm"],
            data=pd.DataFrame({
                "Algorithm": algos,
                metric: metric_vals,
            }),
            detailed=True,
        )

        f_val = one_way["MS"][0] / ms_residual
        p_unc = scipy.stats.f(one_way["DF"][0], df_residual).sf(f_val)
        sig = p_unc < 0.05

        data["SS"].append(one_way["SS"][0])
        data["DF"].append(one_way["DF"][0])
        data["MS"].append(one_way["MS"][0])
        data["F"].append(f_val)
        data["p-unc"].append(p_unc)
        data["significant"].append(sig)

    return pd.DataFrame(data)


def run_pairwise_ttests(figure_data, metric):
    """Runs pairwise t-tests for the hypotheses in the paper."""
    metric_vals = {
        env: {
            algo: [
                metric_from_entry(entry, metric)
                for entry in figure_data[env][algo]
            ]
            for algo in figure_data[env]
        } for env in figure_data
    }

    def compare_to(main_algo, other_algos, bonf_n=None):
        results = {}
        for env in figure_data:
            df = pd.concat(
                [
                    pingouin.ttest(
                        metric_vals[env][main_algo],
                        metric_vals[env][algo],
                        paired=False,
                        alternative="two-sided",
                    )[["T", "dof", "alternative", "p-val"]]
                    for algo in other_algos
                ],
                ignore_index=True,
            )

            # Some hypotheses require overriding bonf_n.
            bonf_n = len(df["p-val"]) if bonf_n is None else bonf_n
            # Adapted from pingouin multicomp implementation:
            # https://github.com/raphaelvallat/pingouin/blob/c66b6853cfcbe1d6d9702c87c09050594b4cacb4/pingouin/multicomp.py#L122
            df["p-val"] = np.clip(df["p-val"] * bonf_n, None, 1)
            df["significant"] = np.less(
                df["p-val"],
                0.05,  # alpha
            )
            df = pd.concat(
                [
                    pd.DataFrame({
                        "Algorithm 1": [main_algo] * len(other_algos),
                        "Algorithm 2": other_algos,
                    }),
                    df,
                ],
                axis=1,
            )
            results[env] = df
        return results

    qd_score_vals = {
        env: {
            algo: [
                metric_from_entry(entry, "QD Score")
                for entry in figure_data[env][algo]
            ]
            for algo in figure_data[env]
        } for env in figure_data
    }

    # Only works on QD score.
    def noninferiority(main_algo, ref_algo, margin, bonf_n=None):
        results = {}
        for env in figure_data:
            df = pd.concat(
                [
                    pingouin.ttest(
                        qd_score_vals[env][main_algo],
                        np.array(qd_score_vals[env][algo]) - margin,
                        paired=False,
                        alternative="greater",
                    )[["T", "dof", "alternative", "p-val"]]
                    for algo in [ref_algo]
                ],
                ignore_index=True,
            )

            # Some hypotheses require overriding bonf_n.
            bonf_n = len(df["p-val"]) if bonf_n is None else bonf_n
            # Adapted from pingouin multicomp implementation:
            # https://github.com/raphaelvallat/pingouin/blob/c66b6853cfcbe1d6d9702c87c09050594b4cacb4/pingouin/multicomp.py#L122
            df["p-val"] = np.clip(df["p-val"] * bonf_n, None, 1)
            df["significant"] = np.less(
                df["p-val"],
                0.05,  # alpha
            )
            df = pd.concat(
                [
                    pd.DataFrame({
                        "Algorithm 1": [main_algo],
                        "Algorithm 2": [ref_algo],
                    }),
                    df,
                ],
                axis=1,
            )
            results[env] = df
        return results

    if metric == "Normalized QD Score":
        return {}
    elif metric == "Internal Runtime (Minutes)":
        return {}
    else:
        raise NotImplementedError(f"No hypotheses for {metric}")


def run_pairwise(figure_data, metric, alpha=0.05, test="tukey"):
    metric_vals = {
        env: {
            algo: [
                metric_from_entry(entry, metric)
                for entry in figure_data[env][algo]
            ]
            for algo in figure_data[env]
        } for env in figure_data
    }

    # Raw results from running the pairwise comparisons. This DF looks something
    # like:
    #
    #      A           B      p_tukey/pval
    # sep-CMA-MAE   CMA-MAE     0.05
    raw_pairwise = {}
    for env in figure_data:
        df = pd.DataFrame({
            "Algorithm": [
                algo for algo in metric_vals[env]
                for metric in metric_vals[env][algo]
            ],
            "Metric": [
                metric for algo in metric_vals[env]
                for metric in metric_vals[env][algo]
            ],
        })
        if test == "tukey":
            raw_pairwise[env] = pingouin.pairwise_tukey(df,
                                                        dv="Metric",
                                                        between="Algorithm")
        elif test == "games-howell":
            raw_pairwise[env] = pingouin.pairwise_gameshowell(
                df, dv="Metric", between="Algorithm")

    # Processed to look like:
    #
    #               sep-CMA-MAE      CMA-MAE    ...
    # sep-CMA-MAE       N/A             >
    #   CMA-MAE          <
    envs = list(figure_data)
    algos = list(figure_data[next(iter(figure_data))])
    res = pd.DataFrame(
        index=algos,
        columns=pd.MultiIndex.from_product([envs, algos]),
        dtype=object,
    )

    for env in raw_pairwise:
        # Algorithms can't compare with themselves.
        for algo in algos:
            res[env, algo][algo] = "N/A"

        # Use symbols to mark other comparisons.
        # pylint: disable = unused-variable
        for index, row in raw_pairwise[env].iterrows():
            if test == "tukey":
                p = row.loc["p-tukey"]
            elif test == "games-howell":
                p = row.loc["pval"]
            a, b = row.loc["A"], row.loc["B"]
            if p > alpha:
                # No significant difference.
                res[env, b][a] = "~"
                res[env, a][b] = "~"
            else:
                # Significant result.
                if row.loc["T"] < 0:
                    res[env, b][a] = "<"
                    res[env, a][b] = ">"
                else:
                    res[env, b][a] = ">"
                    res[env, a][b] = "<"

    processed_pairwise = res
    return raw_pairwise, processed_pairwise


def tests_for_metric(figure_data, root_dir: Path, metric: str):
    """Saves tests for the metric into a subdirectory of root_dir.

    Returns results from various tests.
    """
    output = root_dir / slugify.slugify(metric)
    output.mkdir()

    data = {
        "Environment": [],
        "Algorithm": [],
        metric: [],
    }
    grouped_scores = []
    grouped_scores_by_env = {}

    logger.info("Loading {}", metric)
    n_envs = len(figure_data)
    for env in figure_data:
        grouped_scores_by_env[env] = []
        n_algos = len(figure_data[env])
        for algo in figure_data[env]:
            entry_scores = [
                metric_from_entry(entry, metric)
                for entry in figure_data[env][algo]
            ]

            data["Environment"].extend([env] * len(entry_scores))
            data["Algorithm"].extend([algo] * len(entry_scores))
            data[metric].extend(entry_scores)

            grouped_scores.append(entry_scores)
            grouped_scores_by_env[env].append(entry_scores)
    df = pd.DataFrame(data)
    df.to_csv(output / "data.csv", index=False)

    logger.info("Drawing displot for normality")
    sns.displot(
        data=df,
        x=metric,
        col="Algorithm",
        row="Environment",
        facet_kws=dict(
            sharex=False,
            sharey=False,
        ),
        kind="kde",
    )
    plt.savefig(output / "displot.pdf")

    logger.info("Drawing qqplot for normality")
    fig, axs = plt.subplots(
        nrows=n_envs,
        ncols=n_algos,
        figsize=(4 * n_algos, 4 * n_envs),
    )
    for i_env, env in enumerate(figure_data):
        for i_algo, algo in enumerate(figure_data[env]):
            ax = axs[i_env, i_algo]
            entry_scores = [
                metric_from_entry(entry, metric)
                for entry in figure_data[env][algo]
            ]
            pingouin.qqplot(entry_scores, dist="norm", ax=ax)
            normality = pingouin.normality(entry_scores, method="shapiro")
            ax.set_title(
                f"{env} | {algo}\n"
                f"Normal: {normality['normal'][0]}, p={normality['pval'][0]}")
    fig.tight_layout()
    fig.savefig(output / "qqplot.pdf")

    logger.info("Drawing interaction plots")
    interaction_plot(
        df["Environment"],
        df["Algorithm"],
        df[metric],
        colors=COLORBLIND_REORDERED[:len(set(df["Algorithm"]))],
    )
    plt.savefig(output / "interaction_plot_1.png")
    interaction_plot(
        df["Algorithm"],
        df["Environment"],
        df[metric],
        colors=COLORBLIND_REORDERED[:len(set(df["Environment"]))],
    )
    plt.savefig(output / "interaction_plot_2.png")

    logger.info("Running ANOVA")
    anova_res = pingouin.anova(
        dv=metric,
        between=["Environment", "Algorithm"],
        data=df,
    )
    anova_res["significant"] = anova_res["p-unc"] < 0.05

    logger.info("Running One-Way ANOVAs")
    one_anova_res = {
        env:
            pingouin.anova(data=df[df["Environment"] == env],
                           dv=metric,
                           between="Algorithm") for env in figure_data
    }
    one_anova_str = "\n\n".join(f"### {env}\n{one_anova_res[env].to_markdown()}"
                                for env in one_anova_res)

    logger.info("Running One-Way Welch ANOVAs")
    welch_anova_res = {
        env:
            pingouin.welch_anova(data=df[df["Environment"] == env],
                                 dv=metric,
                                 between="Algorithm") for env in figure_data
    }
    welch_anova_str = "\n\n".join(
        f"### {env}\n{welch_anova_res[env].to_markdown()}"
        for env in welch_anova_res)

    logger.info("Running simple main effects")
    simple_main = calc_simple_main_effects(figure_data, anova_res, metric)

    logger.info("Running pairwise t-tests")
    ttests = run_pairwise_ttests(figure_data, metric)
    ttest_str_parts = {
        hypothesis:
            '\n\n'.join(f"""\
#### {env}

{d.to_markdown()}
""" for env, d in results.items()) for hypothesis, results in ttests.items()
    }
    ttests_str = '\n\n'.join(f"""\
### {hypothesis}

{str_part}
""" for hypothesis, str_part in ttest_str_parts.items())

    logger.info("Running Tukey tests")
    raw_tukey, processed_tukey = run_pairwise(figure_data, metric, test="tukey")

    logger.info("Running Games-Howell tests")
    raw_gameshowell, processed_gameshowell = run_pairwise(figure_data,
                                                          metric,
                                                          test="games-howell")

    logger.info("Checking homoscedasticity (equal variances)")
    var_test = pingouin.homoscedasticity(
        grouped_scores,
        method="levene",
        alpha=0.05,
    )

    logger.info("Homoscedasticity in each environment")
    var_test_by_env = {
        env:
            pingouin.homoscedasticity(grouped_scores_by_env[env],
                                      method="levene",
                                      alpha=0.05)
        for env in grouped_scores_by_env
    }
    var_test_by_env_str = "\n\n".join(
        f"### {env}\n{var_test_by_env[env].to_markdown()}"
        for env in var_test_by_env)

    markdown_file = output / "README.md"
    logger.info("Writing results to {}", markdown_file)
    with open(markdown_file, "w") as file:
        file.write(f"""\
# Statistical Tests for {metric}

Significance tests done by checking if `p < 0.05`

## ANOVA

{anova_res.to_markdown()}

See [here](https://pingouin-stats.org/generated/pingouin.anova.html) for more
info on the pingouin ANOVA method.

**Table Columns:**

- `Source`: Factor names
- `SS`: Sums of squares
- `DF`: Degrees of freedom
- `MS`: Mean squares
- `F`: F-values
- `p-unc`: uncorrected p-values
- `np2`: Partial eta-square effect sizes

### Analyses

- Environment: $F({anova_res['DF'][0]}, {anova_res['DF'][3]}) = {anova_res['F'][0]:.2f}$
- Algorithm: $F({anova_res['DF'][1]}, {anova_res['DF'][3]}) = {anova_res['F'][1]:.2f}$
- Environment * Algorithm: $F({anova_res['DF'][2]}, {anova_res['DF'][3]}) = {anova_res['F'][2]:.2f}$

## One-Way ANOVA in each environment

{one_anova_str}

## One-Way Welch ANOVA in each environment

{welch_anova_str}

## Normality

See [displot.pdf](./displot.pdf) and [qqplot.pdf](./qqplot.pdf)

## Homoscedasticity (Equal Variances)

{var_test.to_markdown()}

{var_test_by_env_str}

## Interaction Plots

![Interaction Plot 1](./interaction_plot_1.png)

![Interaction Plot 2](./interaction_plot_2.png)

## Simple Main Effects

{simple_main.to_markdown()}

## Pairwise t-tests

(p-values Bonferroni corrected within each environment / simple main effect;
alpha is still 0.05)

{ttests_str}

## Tukey

{processed_tukey.to_markdown()}

## Games-Howell

{processed_gameshowell.to_markdown()}
""")

    write_tests_as_latex(metric, anova_res, simple_main, ttests,
                         processed_tukey, processed_gameshowell, output)

    logger.info("Done")


def format_p_val(p, include_p=False, math=False):
    """Formats p-values for Latex."""
    if p < 0.001:
        p_str = f"{'p 'if include_p else '' }< 0.001"
    elif p >= 1.0:
        p_str = f"{'p = ' if include_p else ''}1"
    else:
        p_str = f"{'p = ' if include_p else ''}{p:.3f}"

    # Put significant p-values in bold.
    if p < 0.05:
        p_str = ("\\mathbf{" if math else "\\textbf{") + p_str + "}"

    return p_str


def write_tests_as_latex(metric, anova_res, simple_main, ttests,
                         processed_tukey, processed_gameshowell, output: Path):
    """Write the tests to latex files in the output directory."""
    logger.info("Writing tests as latex")

    # Write ANOVA results as list.
    with open(output / "anova.tex", "w") as file:
        caption = f"Simple main effects for {CAPTION_NAME[metric]}"
        label = f"{slugify.slugify(metric)}-main"
        file.write(TABLE_HEADER)
        file.write("\\begin{itemize}\n")
        file.write("\\item Interaction effect: "
                   f"$F({anova_res['DF'][2]}, {anova_res['DF'][3]}) = "
                   f"{anova_res['F'][2]:.2f}, "
                   f"{format_p_val(anova_res['p-unc'][2], True, True)}$\n")
        file.write("\\item Simple main effects:\n")
        file.write("  \\begin{itemize}\n")

        err_deg = simple_main["DF"][0]
        only_main = simple_main.loc[1:]
        for env, df, f, p in zip(only_main["Environment"], only_main["DF"],
                                 only_main["F"], only_main["p-unc"]):
            file.write(
                f"  \\item {env}: "
                f"$F({df}, {err_deg}) = {f:.2f}, {format_p_val(p, True, True)}$"
                "\n")

        file.write("  \\end{itemize}\n")
        file.write("\\end{itemize}\n")
        file.write("\n")

    # Write ttests as tables.
    table_i = 0
    for hypothesis, hyp_results in ttests.items():
        with open(output / f"ttests-{table_i}.tex", "w") as file:
            file.write(TABLE_HEADER)

            first_env_df = next(iter(hyp_results.values()))
            pval_df = pd.DataFrame(
                # Index is the algorithms.
                index=pd.MultiIndex.from_frame(
                    first_env_df[["Algorithm 1", "Algorithm 2"]]),
                # Columns are environments.
                columns=list(hyp_results),
                dtype=str,
            )

            for env, env_df in hyp_results.items():
                for algo1, algo2, p in zip(env_df["Algorithm 1"],
                                           env_df["Algorithm 2"],
                                           env_df["p-val"]):
                    pval_df[env][algo1, algo2] = format_p_val(p)

            caption = f"{hypothesis}"
            label = f"{slugify.slugify(metric)}-ttests-{table_i}"
            table_i += 1

            file.write("\\begin{table*}[t]\n")
            file.write("\\caption{" + caption + "}\n")
            file.write("\\label{table:" + slugify.slugify(label) + "}\n")
            file.write("\\begin{center}\n")
            file.write(
                pval_df.to_latex(
                    column_format=" L{1.2 in}" * 2 +
                    " R{0.9in}" * len(pval_df.columns),
                    escape=False,
                    multirow=True,
                ))
            file.write("\\end{center}\n")
            file.write("\\end{table*}\n")
            file.write("\n")

    # Write tukey as a Latex table.
    with open(output / "tukey.tex", "w") as file:
        file.write(TABLE_HEADER)
        latex_str = processed_tukey.to_latex(
            column_format="l" + "r" * len(processed_tukey.columns),
            escape=False,  # Escape all the special < characters.
            multirow=True,
        )
        latex_str = (latex_str.replace("<", "$<$").replace(">", "$>$").replace(
            "~", "$-$"))
        file.write(latex_str)

    # Write Games-Howell as a Latex table.
    with open(output / "games-howell.tex", "w") as file:
        file.write(TABLE_HEADER)
        latex_str = processed_gameshowell.to_latex(
            column_format="l" + "r" * len(processed_gameshowell.columns),
            escape=False,  # Escape all the special < characters.
            multirow=True,
        )
        latex_str = (latex_str.replace("<", "$<$").replace(">", "$>$").replace(
            "~", "$-$"))
        file.write(latex_str)


# Name in table captions.
CAPTION_NAME = {
    "Normalized QD Score": "QD score",
    "Normalized QD Score AUC": "QD score AUC",
    "Internal Runtime (Minutes)": "Internal Runtime (Minutes)",
}


def tests(figure_data: str = "figure_data_optbench.json",
          output: str = "stats_tests_optbench"):
    """Outputs information about statistical tests.

    We use Normalized scores since the ANOVA needs values to have the same
    scale.

    Args:
        figure_data: Path to JSON file with figure data.
        output: Directory to save (extended) outputs.
    """
    logger.info("Creating logging directory")
    output = Path(output)
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir()

    logger.info("Loading figure data")
    figure_data = load_figure_data(figure_data)

    with mpl_style_file("simple.mplstyle") as f:
        with plt.style.context(f):
            for metric in [
                    "Normalized QD Score",
                    #  "Normalized QD Score AUC",
                    "Internal Runtime (Minutes)",
            ]:
                logger.info("===== {} =====", metric)
                tests_for_metric(figure_data, output, metric)


if __name__ == "__main__":
    fire.Fire()
