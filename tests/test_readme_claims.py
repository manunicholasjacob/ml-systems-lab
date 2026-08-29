"""The numbers in the README have to still be true.

A framework whose selling point is reproducibility cannot have a README that quietly
drifts away from the records it ships. Every figure checked here is recomputed from
`results/` and compared against what the prose says, so a dataset change or an analysis
change that moves a headline number fails here instead of in a reviewer's hands.

This exists because "the two campaigns agree within 1.6%" was in the README while the
committed records said 1.64% of the larger figure, 1.67% of the smaller and 1.66% of the
mean. Rounding to 1.6 meant picking the most flattering denominator.
"""

import os
import re

import pytest

from mlsyslab.analysis.dataset import Dataset, decode_roofline

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(name):
    with open(os.path.join(REPO, name), encoding="utf-8") as f:
        return f.read()


def _fit(directory, device_prefix):
    ds = Dataset.from_directory(os.path.join(REPO, "results", directory))
    for (device,), subset in ds.filter(backend="llamacpp").group_by("device"):
        if device.startswith(device_prefix):
            return decode_roofline(subset)
    pytest.skip(f"no {device_prefix} records in results/{directory}")


def test_the_published_pi_fit_is_what_the_readme_says():
    fit = _fit("paper12", "pi5")
    assert round(fit["bw_eff_GBs"], 1) == 10.7
    assert round(fit["r2"], 3) == 0.980
    assert fit["n"] == 7


def test_the_published_x86_fit_is_what_the_readme_says():
    fit = _fit("paper12", "laptop")
    assert round(fit["bw_eff_GBs"], 1) == 35.7
    assert round(fit["r2"], 3) == 0.980
    assert fit["n"] == 7


def test_the_two_campaigns_agree_to_the_percentage_the_readme_claims():
    a = _fit("paper12", "pi5")["bw_eff_GBs"]
    b = _fit("pi5-campaign", "pi5")["bw_eff_GBs"]
    gap = abs(a - b)
    # Quote the least flattering of the three denominators, rounded up, so the claim is
    # true however a reader chooses to compute it.
    worst = max(gap / min(a, b), gap / max(a, b), gap / ((a + b) / 2))
    claimed = 100 * worst
    assert 1.6 < claimed <= 1.7, claimed

    for name in ("README.md", "results/README.md"):
        text = _read(name)
        stated = re.search(r"agree\w*\s+(?:on the effective bandwidth\s+)?within ([\d.]+)%",
                           text)
        assert stated, f"{name} no longer states an agreement figure"
        assert float(stated.group(1)) >= claimed, (
            f"{name} claims {stated.group(1)}% but the records say {claimed:.2f}%")


def test_the_record_count_the_readme_quotes_is_right():
    text = _read("results/README.md")
    for directory, phrase in [("paper12", r"(\d+) records"),
                              ("pi5-campaign", r"(\d+) points"),
                              ("laptop-campaign", r"\((\d+) points")]:
        block = text.split(f"## {directory}/", 1)
        if len(block) < 2:
            continue
        stated = re.search(phrase, block[1][:600])
        if not stated:
            continue
        on_disk = len(Dataset.from_directory(
            os.path.join(REPO, "results", directory), include_failed=True))
        assert int(stated.group(1)) == on_disk, (directory, stated.group(1), on_disk)


# ------------------------------------------------- the containerisation study's numbers

def _containerization_summary():
    import json

    path = os.path.join(REPO, "results", "containerization", "summary.json")
    if not os.path.exists(path):
        pytest.skip("results/containerization has not been generated")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def test_the_readme_containerisation_table_matches_the_records():
    """Every percentage in that table is recomputed from the records that produced it."""
    summary = _containerization_summary()
    readme = _read("README.md")
    labels = {"pod, no CPU quota": "k8s-unlimited",
              "pod, quota 8 cores": "k8s-cpu8",
              "pod, quota 4 cores": "k8s-cpu4",
              "pod, quota 2 cores": "k8s-cpu2"}

    checked = 0
    for label, arm in labels.items():
        row = re.search(r"\|\s*(?:\*\*)?" + re.escape(label)
                        + r"(?:\*\*)?\s*\|\s*\*{0,2}([+-][\d.]+)% mean, (\d+) of (\d+)",
                        readme)
        assert row, f"README no longer states a figure for {label}"
        stated_mean = float(row.group(1))
        stated_resolved = int(row.group(2))
        stated_total = int(row.group(3))

        actual_mean = summary["overhead_pct_vs_host"][arm]
        actual_resolved, actual_total = summary["points_resolved_vs_total"][arm]
        assert abs(stated_mean - actual_mean) < 0.05, (label, stated_mean, actual_mean)
        assert (stated_resolved, stated_total) == (actual_resolved, actual_total), label
        checked += 1
    assert checked == 4


def test_the_readme_ranking_claim_is_what_the_records_say():
    summary = _containerization_summary()
    readme = _read("README.md")
    if summary["arms_that_reordered"]:
        assert "ranking of the five model and quantisation combinations is identical" \
            not in readme, "the README claims a preserved ranking the records contradict"
    else:
        assert "identical in all five" in readme
        # And no arm may even appear to reorder without the README saying so.
        assert not summary["arms_whose_order_differed_within_noise"] or \
            "within noise" in readme


def test_the_study_carries_both_passes_and_one_image_digest():
    summary = _containerization_summary()
    assert summary["passes"] == [1, 2], "the drift control needs two independent passes"
    assert summary["n_records"] == 50
    assert len(summary["image_ids"]) == 1, (
        "more than one image ran; the arms are not comparable")
