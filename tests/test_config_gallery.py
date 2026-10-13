"""Every shipped config loads, every template is usable as one, and the gallery page
in the docs is what tools/gen_config_gallery.py would write today.

A config that no longer parses is a broken example, and a gallery entry that no
longer matches its file is documentation lying about the code. Both fail here.
"""

import glob
import importlib.util
import os

import pytest

from mlsyslab.config import load

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_CONFIGS = sorted(
    glob.glob(os.path.join(REPO, "configs", "*.yaml"))
    + glob.glob(os.path.join(REPO, "configs", "examples", "*.yaml"))
)
TEMPLATES = sorted(glob.glob(os.path.join(REPO, "configs", "examples", "*.yaml")))


def _gallery_module():
    path = os.path.join(REPO, "tools", "gen_config_gallery.py")
    spec = importlib.util.spec_from_file_location("gen_config_gallery", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=os.path.basename)
def test_every_shipped_config_expands(path):
    config = load(path)
    assert config.specs, f"{path} expands to no runs"


def test_there_are_templates_to_copy():
    assert len(TEMPLATES) >= 5


@pytest.mark.parametrize("path", TEMPLATES, ids=os.path.basename)
def test_templates_mark_what_to_edit_and_say_how_to_run(path):
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    rel = os.path.relpath(path, REPO).replace(os.sep, "/")
    assert text.startswith("# "), "a template opens with the question it answers"
    assert "<-- EDIT" in text, "a template marks the machine-specific lines"
    assert f"mlsys run {rel}" in text, "the header shows the exact run command"
    assert f"mlsys doctor --config {rel}" in text, "and the check to run before it"


@pytest.mark.parametrize("path", TEMPLATES, ids=os.path.basename)
def test_templates_carry_nothing_from_the_authors_machines(path):
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    for private in ("C:/llmpc", "100.98.", "raspberry_pi_key", "/home/manu", "manu@"):
        assert private not in text, f"{private!r} belongs in a study, not a template"


def test_the_gallery_page_is_current():
    gallery = _gallery_module()
    with open(gallery.OUT, encoding="utf-8") as fh:
        committed = fh.read()
    assert committed == gallery.render(), (
        "docs/configs.md is stale: run python tools/gen_config_gallery.py")


def test_the_gallery_lists_every_config():
    with open(os.path.join(REPO, "docs", "configs.md"), encoding="utf-8") as fh:
        page = fh.read()
    for path in ALL_CONFIGS:
        rel = os.path.relpath(path, REPO).replace(os.sep, "/")
        assert f"`{rel}`" in page, f"{rel} is missing from the gallery"
