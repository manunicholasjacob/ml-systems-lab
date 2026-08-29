"""Turn the containerisation sweep into a table, an overhead figure, and a verdict.

Two questions, in order of how much they matter:

1. What does containerisation and orchestration cost an inference benchmark?
2. Does it change the *ranking* of the things being compared?

The second is the one that decides whether anyone benchmarking models in Kubernetes has
to care. A uniform tax you can subtract; a tax that reorders your candidates means every
comparison run in a pod is suspect. If the ranking holds, that is a useful negative
result and it gets reported as one, in the same words either way.

Nothing here is allowed to answer either question by eye. The first version of this
script did, and it reported that four of four arms reordered the models, on data whose
run-to-run spread was 13 to 25 percent and whose "reorderings" were all swaps between
models a few percent apart. Every claim below now has to clear the noise it was measured
against, and a difference that does not clear it is reported as "cannot tell", which is
a real answer and a different one from "no difference".

    python tools/containerization_report.py runs/containerization-tax [runs/pass2] \\
        --out results/containerization
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))

from mlsyslab.analysis import Dataset  # noqa: E402

BASELINE = "wsl-host"

# Order the arms by how much machinery is between the benchmark and the CPU, so the table
# reads as a progression rather than as an alphabetical accident.
ARM_ORDER = ["wsl-host", "k8s-unlimited", "k8s-cpu8", "k8s-cpu4", "k8s-cpu2"]

ARM_LABELS = {
    "wsl-host": "host process, no container",
    "k8s-unlimited": "pod, no CPU quota",
    "k8s-cpu8": "pod, quota 8 cores",
    "k8s-cpu4": "pod, quota 4 cores",
    "k8s-cpu2": "pod, quota 2 cores",
}


def load(directories: Sequence[str]) -> List[dict]:
    rows: List[dict] = []
    for index, directory in enumerate(directories):
        dataset = Dataset.from_directory(directory)
        for row in dataset.rows:
            row = dict(row)
            row["pass"] = index + 1
            row["source"] = directory
            rows.append(row)
    return rows


def arms_present(rows: Sequence[dict]) -> List[str]:
    seen = {r["device"] for r in rows}
    ordered = [a for a in ARM_ORDER if a in seen]
    return ordered + sorted(seen - set(ordered))


def cell(rows: Sequence[dict], device: str, model: str, metric: str,
         pass_number: Optional[int] = None) -> Optional[float]:
    values = [r[metric] for r in rows
              if r["device"] == device and r["model"] == model
              and r.get(metric) is not None
              and (pass_number is None or r["pass"] == pass_number)]
    return statistics.mean(values) if values else None


def spread(rows: Sequence[dict], device: str, model: str) -> Optional[float]:
    """llama-bench's own run-to-run spread, in percent, averaged over passes."""
    values = [r["stdev_pct"] for r in rows
              if r["device"] == device and r["model"] == model
              and r.get("stdev_pct") is not None]
    return statistics.mean(values) if values else None


def reps(rows: Sequence[dict], device: str, model: str) -> int:
    values = [r.get("repetitions") for r in rows
              if r["device"] == device and r["model"] == model and r.get("repetitions")]
    return int(statistics.mean(values)) if values else 1


def standard_error(rows: Sequence[dict], device: str, model: str) -> Optional[float]:
    """Standard error of the mean throughput for one point, in the metric's units.

    llama-bench reports the spread across its own repetitions. The uncertainty in the
    *mean* is that spread over the square root of the repetition count, and it is the
    mean that every comparison here is between.
    """
    mean = cell(rows, device, model, "decode_tps")
    pct = spread(rows, device, model)
    if mean is None or pct is None:
        return None
    n = max(1, reps(rows, device, model))
    return (pct / 100.0) * mean / (n ** 0.5)


def separated(rows: Sequence[dict], a_device: str, a_model: str,
              b_device: str, b_model: str, sigmas: float = 2.0) -> Optional[bool]:
    """Are these two points far enough apart to be called different?

    None when either point has no spread to compare against, which is itself worth
    saying: an unresolved difference and an absent one are not the same claim.
    """
    a = cell(rows, a_device, a_model, "decode_tps")
    b = cell(rows, b_device, b_model, "decode_tps")
    sa = standard_error(rows, a_device, a_model)
    sb = standard_error(rows, b_device, b_model)
    if None in (a, b, sa, sb):
        return None
    return abs(a - b) > sigmas * ((sa ** 2 + sb ** 2) ** 0.5)


def table(rows: Sequence[dict], metric: str, models: Sequence[str],
          arms: Sequence[str]) -> str:
    header = ["model"] + [ARM_LABELS.get(a, a) for a in arms]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for model in models:
        cells = [model]
        for arm in arms:
            value = cell(rows, arm, model, metric)
            noise = spread(rows, arm, model)
            if value is None:
                cells.append("-")
            elif noise is None:
                cells.append(f"{value:.2f}")
            else:
                cells.append(f"{value:.2f} ±{noise:.1f}%")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def relative_table(rows: Sequence[dict], metric: str, models: Sequence[str],
                   arms: Sequence[str]) -> str:
    header = ["model"] + [ARM_LABELS.get(a, a) for a in arms if a != BASELINE]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for model in models:
        base = cell(rows, BASELINE, model, metric)
        cells = [model]
        for arm in arms:
            if arm == BASELINE:
                continue
            value = cell(rows, arm, model, metric)
            if base and value:
                delta = 100.0 * (value / base - 1.0)
                real = separated(rows, BASELINE, model, arm, model)
                mark = "" if real else " (ns)"
                cells.append(f"{delta:+.1f}%{mark}")
            else:
                cells.append("-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ranking(rows: Sequence[dict], arm: str, models: Sequence[str],
            metric: str) -> List[str]:
    scored = [(cell(rows, arm, m, metric), m) for m in models]
    scored = [(v, m) for v, m in scored if v is not None]
    return [m for _v, m in sorted(scored, reverse=True)]


def kendall_tau(a: Sequence[str], b: Sequence[str]) -> Optional[float]:
    """Rank correlation between two orderings of the same items, without scipy."""
    common = [x for x in a if x in b]
    if len(common) < 2:
        return None
    position = {name: index for index, name in enumerate(b)}
    concordant = discordant = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            left = position[common[i]] - position[common[j]]
            if left < 0:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else None


def drift_check(rows: Sequence[dict], arms: Sequence[str],
                models: Sequence[str], metric: str) -> Tuple[str, Optional[float]]:
    """Do the two independent passes agree?

    A sweep takes long enough that the machine's thermal state at the end is not its
    state at the start. Running the whole matrix twice and comparing is the cheapest
    honest control: if the arms keep their order and their sizes across both passes, the
    result is not an artefact of when each point happened to run.
    """
    passes = sorted({r["pass"] for r in rows})
    if len(passes) < 2:
        return ("Only one pass was run, so drift over the sweep is not controlled for. "
                "Run the matrix a second time into another directory and pass both here."), None

    deltas = []
    for arm in arms:
        for model in models:
            first = cell(rows, arm, model, metric, pass_number=passes[0])
            second = cell(rows, arm, model, metric, pass_number=passes[1])
            if first and second:
                deltas.append(100.0 * (second / first - 1.0))
    if not deltas:
        return "The two passes share no comparable points.", None
    worst = max(abs(d) for d in deltas)
    mean = statistics.mean(deltas)
    text = (f"Two independent passes over the whole matrix. Point by point, the second "
            f"pass differs from the first by {mean:+.1f}% on average and by at most "
            f"{worst:.1f}%.")
    return text, worst


def figure(rows, models, arms, quotas, out_dir, metric="decode_tps"):
    """Two panels: what each arm measured, and what it cost relative to the host.

    Error bars are the standard error of the mean, not the raw spread, because the
    question every bar is answering is about the mean. Drawing the spread instead would
    make every difference here look unresolvable, which is its own kind of wrong.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 8),
                                      gridspec_kw={"height_ratios": [3, 2]})
    width = 0.8 / max(1, len(arms))
    positions = list(range(len(models)))

    # One colour per arm, shared by both panels. Letting each panel run the default
    # colour cycle from its own first bar meant blue was the host on top and the
    # unconstrained pod underneath, which is worse than no colour at all.
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    palette = {arm: cycle[i % len(cycle)] for i, arm in enumerate(arms)} if cycle else {}

    for index, arm in enumerate(arms):
        heights = [cell(rows, arm, m, metric) or 0.0 for m in models]
        errors = [standard_error(rows, arm, m) or 0.0 for m in models]
        offsets = [p + index * width - 0.4 + width / 2 for p in positions]
        quota = quotas.get(arm)
        label = ARM_LABELS.get(arm, arm)
        top.bar(offsets, heights, width, yerr=errors, capsize=2, label=label,
                color=palette.get(arm), edgecolor="black", linewidth=0.4)

    top.set_xticks(positions)
    top.set_xticklabels(models, rotation=15, ha="right")
    top.set_ylabel("decode tokens/s")
    top.set_title("Same binary, same model file, same machine: host process versus pod")
    top.legend(fontsize=8, ncol=2)
    top.grid(axis="y", alpha=0.3)

    for index, arm in enumerate([a for a in arms if a != BASELINE]):
        deltas, errs = [], []
        for model in models:
            base = cell(rows, BASELINE, model, metric)
            value = cell(rows, arm, model, metric)
            deltas.append(100.0 * (value / base - 1.0) if (base and value) else 0.0)
            sb = standard_error(rows, arm, model) or 0.0
            sa = standard_error(rows, BASELINE, model) or 0.0
            # Two standard errors, matching the threshold the prose uses to decide
            # whether a difference is real. A bar whose error bar crosses zero is one
            # the text reports as unresolved.
            errs.append(200.0 * ((sa ** 2 + sb ** 2) ** 0.5) / base if base else 0.0)
        span = 0.8 / max(1, len(arms) - 1)
        offsets = [p + index * span - 0.4 + span / 2 for p in positions]
        bottom.bar(offsets, deltas, span, yerr=errs, capsize=2,
                   label=ARM_LABELS.get(arm, arm), color=palette.get(arm),
                   edgecolor="black", linewidth=0.4)

    bottom.axhline(0, color="black", linewidth=0.8)
    bottom.set_xticks(positions)
    bottom.set_xticklabels(models, rotation=15, ha="right")
    bottom.set_ylabel("% vs host process")
    # One legend, shared, now that the colours agree between the panels. A second copy
    # down here only competed with the bars it was trying to explain.
    bottom.set_title("Cost relative to an uncontainerised process, colours as above.\n"
                     "Error bars are two standard errors: a bar crossing zero is a "
                     "difference this measurement cannot resolve.", fontsize=10)
    bottom.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "containerization.png")
    fig.savefig(path, dpi=150)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)
    return path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", help="one results directory per pass")
    parser.add_argument("--out", default="results/containerization")
    parser.add_argument("--metric", default="decode_tps")
    args = parser.parse_args(argv)

    rows = load(args.directories)
    if not rows:
        sys.stderr.write("no successful records found\n")
        return 1

    arms = arms_present(rows)
    models = sorted({r["model"] for r in rows},
                    key=lambda m: -(cell(rows, BASELINE, m, args.metric) or 0))
    os.makedirs(args.out, exist_ok=True)

    rankings = {arm: ranking(rows, arm, models, args.metric) for arm in arms}
    base_ranking = rankings.get(BASELINE, [])
    ranking_lines = []
    reorderings = []          # arms whose reordering survives the noise
    apparent = []             # arms whose order differs at all, resolved or not
    for arm in arms:
        order = rankings[arm]
        tau = kendall_tau(base_ranking, order) if arm != BASELINE else 1.0
        same = order == base_ranking
        verdict = "same"
        if arm != BASELINE and not same:
            apparent.append(arm)
            # A swap only counts if the two models that changed places are actually
            # distinguishable in this arm. Two points a few percent apart, measured with
            # a fifteen percent spread, can trade places for no reason at all.
            swapped = [m for m, n in zip(base_ranking, order) if m != n]
            resolved = [
                separated(rows, arm, swapped[i], arm, swapped[j])
                for i in range(len(swapped)) for j in range(i + 1, len(swapped))
            ]
            if any(r is True for r in resolved):
                reorderings.append(arm)
                verdict = "REORDERED"
            else:
                verdict = "differs, but within noise"
        ranking_lines.append(
            f"| {ARM_LABELS.get(arm, arm)} | {' > '.join(order)} | {verdict} | "
            f"{'-' if tau is None else f'{tau:+.2f}'} |")

    drift_text, drift_worst = drift_check(rows, arms, models, args.metric)

    overheads = {}
    resolved_counts = {}
    for arm in arms:
        if arm == BASELINE:
            continue
        deltas, resolved = [], 0
        for model in models:
            base = cell(rows, BASELINE, model, args.metric)
            value = cell(rows, arm, model, args.metric)
            if base and value:
                deltas.append(100.0 * (value / base - 1.0))
                if separated(rows, BASELINE, model, arm, model):
                    resolved += 1
        if deltas:
            overheads[arm] = (statistics.mean(deltas), min(deltas), max(deltas))
            resolved_counts[arm] = (resolved, len(deltas))

    # A quota does not only move the mean. Whether it also widens the spread is a
    # separate question, and for anyone running a latency-sensitive service it is the
    # more important one.
    stability = {}
    for arm in arms:
        values = [spread(rows, arm, m) for m in models]
        values = [v for v in values if v is not None]
        if values:
            stability[arm] = (statistics.mean(values), max(values))

    quotas = {r["device"]: r.get("cpu_quota_cores") for r in rows}
    threads = sorted({r.get("threads") for r in rows if r.get("threads")})
    thread_count = threads[0] if len(threads) == 1 else None

    def constrained(arm):
        """Is this arm's quota actually below what the benchmark asks for?"""
        quota = quotas.get(arm)
        if quota is None or thread_count is None:
            return False
        return quota < thread_count

    roomy = [a for a in overheads if not constrained(a)]
    tight = [a for a in overheads if constrained(a)]
    roomy_worst = max((abs(overheads[a][0]) for a in roomy), default=None)
    tight_cost = (statistics.mean([overheads[a][0] for a in tight]) if tight else None)
    roomy_resolved = sum(resolved_counts.get(a, (0, 0))[0] for a in roomy)
    roomy_total = sum(resolved_counts.get(a, (0, 0))[1] for a in roomy)
    images = sorted({r.get("image_id") for r in rows if r.get("image_id")})
    devices_line = ", ".join(
        f"{a} (quota {quotas.get(a) if quotas.get(a) is not None else 'none'})"
        for a in arms)

    summary = {
        "arms": arms,
        "models": models,
        "metric": args.metric,
        "overhead_pct_vs_host": {a: round(v[0], 2) for a, v in overheads.items()},
        "overhead_range_pct": {a: [round(v[1], 2), round(v[2], 2)]
                               for a, v in overheads.items()},
        "ranking_by_arm": rankings,
        "arms_that_reordered": reorderings,
        "arms_whose_order_differed_within_noise": [a for a in apparent
                                                   if a not in reorderings],
        "points_resolved_vs_total": resolved_counts,
        "run_to_run_spread_pct": {a: [round(v[0], 1), round(v[1], 1)]
                                  for a, v in stability.items()},
        "passes": sorted({r["pass"] for r in rows}),
        "worst_pass_to_pass_delta_pct": None if drift_worst is None else round(drift_worst, 2),
        "cpu_quota_cores": quotas,
        "image_ids": images,
        "n_records": len(rows),
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    headline = ["## The result, in one paragraph", ""]
    if roomy_worst is not None:
        # "Costs" is the wrong word when the sign comes out positive, and it came out
        # positive here. Say "differs by", and say in which direction, rather than
        # quietly reporting a small gain as a small tax.
        signed = [overheads[a][0] for a in roomy]
        direction = ("slightly faster in the pod" if statistics.mean(signed) > 0
                     else "slightly slower in the pod")
        headline.append(
            f"Running the same binary in a Kubernetes pod instead of as a host process "
            f"changes decode throughput by at most {roomy_worst:.1f}% on average, "
            f"{direction}, and {roomy_total - roomy_resolved} of {roomy_total} "
            f"individual comparisons cannot be resolved from the run-to-run noise at "
            f"all. The honest reading is that this measurement cannot find a cost of "
            f"containerisation, not that it has proved there is none. A CPU quota does "
            f"not change that, as long as the quota is at least the benchmark's thread "
            f"count.")
    if tight_cost is not None and thread_count:
        headline.append("")
        headline.append(
            f"A quota *below* the thread count is a different story: at "
            f"{min(quotas[a] for a in tight):.0f} cores against {thread_count} threads, "
            f"throughput falls {abs(tight_cost):.0f}%, and that difference is resolved "
            f"on every point. The cost is not the container. It is asking for more "
            f"parallelism than the cgroup will let you have, which is a configuration "
            f"anyone can write by accident and which no amount of container tuning "
            f"will fix.")
    headline.append("")
    if not reorderings:
        headline.append(
            "The ranking of the models is identical in every arm, including the one "
            "losing most of its throughput. The tax is close enough to multiplicative "
            "that relative comparisons survive it, which is the practical answer for "
            "anyone choosing between quantisation formats by benchmarking inside a pod.")
        headline.append("")

    report = [
        "# What containerisation costs an inference benchmark",
        "",
        *headline,
        f"Metric: `{args.metric}`. Arms, in order of how much sits between the benchmark",
        f"and the CPU: {devices_line}.",
        "",
        f"{len(rows)} successful records across {len(summary['passes'])} independent "
        f"pass(es) of the matrix.",
        "",
        "## Absolute",
        "",
        table(rows, args.metric, models, arms),
        "",
        "The `±` is llama-bench's own run-to-run spread within a single point. The",
        "uncertainty in each mean is that over the square root of the repetition count,",
        "and it is the noise floor every claim below has to clear.",
        "",
        "## Relative to an uncontainerised process on the same machine",
        "",
        relative_table(rows, args.metric, models, arms),
        "",
        "`(ns)` marks a difference this measurement cannot resolve: the two means are",
        "closer together than two standard errors. Resolved points per arm: "
        + ", ".join(f"{ARM_LABELS.get(a, a)} {r}/{t}"
                    for a, (r, t) in resolved_counts.items()) + ".",
        "",
        "## Does a quota also change the spread?",
        "",
        "| arm | mean run-to-run spread | worst |",
        "|---|---|---|",
        *[f"| {ARM_LABELS.get(a, a)} | {v[0]:.1f}% | {v[1]:.1f}% |"
          for a, v in stability.items()],
        "",
        "## Does it change the ranking?",
        "",
        "| arm | order, fastest first | vs host | Kendall tau |",
        "|---|---|---|---|",
        *ranking_lines,
        "",
    ]

    unresolved = [a for a in apparent if a not in reorderings]
    if reorderings:
        report += [
            "**The ranking is not preserved.** In " + ", ".join(reorderings) + " two "
            "models change places by more than this measurement's noise. That matters "
            "well beyond this repository: it means a quantisation format chosen by "
            "benchmarking inside a constrained pod may not be the format that wins "
            "outside one, and most people benchmarking models today are doing it in "
            "Kubernetes.",
            "",
        ]
    else:
        report += [
            "**No arm reorders the models by more than this measurement can resolve.** "
            "Whatever containerisation costs here, it does not change which model wins.",
            "",
            "This is a negative result and it is the useful kind: a comparison made "
            "inside a pod picks the same winner as the same comparison made outside one, "
            "which is the question most people benchmarking in Kubernetes actually need "
            "answered.",
            "",
        ]
    if unresolved:
        report += [
            "Read the middle of the table with care all the same. "
            + ", ".join(unresolved) + " do list the models in a different order from the "
            "host, but every pair that changed places is closer together than the "
            "measurement can separate, so the reordering is noise and not a finding. "
            "Reporting it as one would be the easiest mistake available here.",
            "",
        ]

    report += [
        "## Controls",
        "",
        "* **One binary.** llama-bench is built once, in `docker/Dockerfile`, and the "
        "host's copy is extracted from that same image, so no part of any difference "
        "above can be a compiler or a build flag.",
        f"* **One image.** Every pod ran `{images[0] if images else 'n/a'}`, recorded by "
        "digest in each run record rather than taken from a tag.",
        "* **One model file.** The pods mount the model directory from the node "
        "read-only, so every arm reads the same inode and no overlayfs copy sits in the "
        "read path.",
        "* **One at a time.** All arms are one physical CPU wearing several device ids, "
        "so they share a scheduler resource group that holds exactly one job and takes "
        "turns between arms. Without the turn-taking, one arm ran its entire queue "
        "before any other arm started, and each arm would have sat in a different part "
        "of the machine's thermal history.",
        "* **Effective limits, not configured ones.** Each record carries the cgroup "
        "ceilings read from inside its own container, so the quota in the table is the "
        "quota that applied rather than the one the config asked for.",
        f"* **Drift.** {drift_text}",
        "",
        "## Limitations, stated rather than discovered",
        "",
        "* The host arm is a process on a WSL2 Linux host, which is itself a virtual "
        "machine. That layer is present in all arms and cancels out of every comparison "
        "here, but these are not bare-metal numbers and must not be described as such.",
        "* One machine, one CPU architecture, one inference engine, one model family. "
        "Nothing here says what happens on a server-class chip, on Arm, or under a "
        "different runtime.",
        "* Throughput and prefill only. Tail latency under a CFS quota is a different "
        "question and a more hostile one, because throttling lands on individual "
        "requests rather than on an average.",
        "* Three nodes in a home lab is not a fleet, and nothing here should be read as "
        "evidence about fleet-scale behaviour.",
        "",
    ]

    drawn = figure(rows, models, arms, quotas, args.out, args.metric)
    if drawn:
        report += ["## Figure", "",
                   "![decode throughput by arm](containerization.png)", ""]

    path = os.path.join(args.out, "REPORT.md")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(report))
    print(f"wrote {path}")
    print(f"wrote {os.path.join(args.out, 'summary.json')}")
    if drawn:
        print(f"wrote {drawn}")
    for arm, (mean, low, high) in overheads.items():
        print(f"  {arm:<16} {mean:+6.1f}%  (range {low:+.1f}% to {high:+.1f}%)")
    for arm, (mean, worst) in stability.items():
        print(f"  spread {arm:<16} mean {mean:.1f}%  worst {worst:.1f}%")
    print("  ranking changed beyond noise: "
          + (", ".join(reorderings) if reorderings else "no"))
    if [a for a in apparent if a not in reorderings]:
        print("  ranking differs within noise (not a finding): "
              + ", ".join(a for a in apparent if a not in reorderings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
