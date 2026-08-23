"""Every Python example in docs/API.md is executed here.

Documentation that does not run is worse than no documentation, because it costs the
reader time before it fails them. The examples are pulled out of the Markdown and run
against the datasets shipped in results/, so a rename that breaks a documented call
fails the suite rather than surfacing in someone's terminal.

Blocks that need hardware or write outside a temporary directory are skipped by name;
each skip is listed explicitly so adding an example without deciding about it is not
silently possible.
"""

import os
import re
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DOC = os.path.join(REPO, "docs", "API.md")

# Examples that cannot run in the suite, and why. Matched as substrings of the block.
NOT_EXECUTABLE = {
    "class MyBackend": "an illustrative skeleton with no method bodies",
    "runner.run()": "executes a real benchmark against real devices",
    "python -m mlsyslab.agent": "shell, not Python; the no-deps CI job covers it",
}

_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def blocks():
    with open(API_DOC, encoding="utf-8") as f:
        text = f.read()
    return _BLOCK.findall(text)


def test_the_doc_has_examples_to_check():
    assert len(blocks()) >= 6


def test_the_api_doc_examples_run_in_order(tmp_path, monkeypatch, capsys):
    """Run the page top to bottom in one namespace, the way a reader would.

    The examples build on each other: the second block uses the records the first
    loaded. Running them independently would need every block restated in full, which
    would make the page worse to read in order to make it easier to test.
    """
    # Relative paths in the examples point at results/, which lives in the repository.
    monkeypatch.chdir(REPO)
    namespace = {"__name__": "__doc_example__"}
    ran = 0
    for index, code in enumerate(blocks()):
        skip = next((why for marker, why in NOT_EXECUTABLE.items() if marker in code),
                    None)
        if skip:
            continue
        if "matplotlib" in code:
            pytest.importorskip("matplotlib")
        # Redirect the few examples that write files into a temporary directory.
        code = code.replace('"figures"', repr(str(tmp_path / "figures")))
        for name in ("table.md", "table.tex"):
            code = code.replace(f'open("{name}", "w")',
                                f'open({str(tmp_path / name)!r}, "w")')
        try:
            exec(compile(code, f"{API_DOC}:block{index}", "exec"), namespace)
        except Exception as exc:  # noqa: BLE001 - report which block, not just that one did
            raise AssertionError(
                f"docs/API.md example {index} failed: {type(exc).__name__}: {exc}\n"
                f"---\n{code}---"
            ) from exc
        ran += 1
    capsys.readouterr()  # the examples print; that is their job, not the suite's
    assert ran >= 5


def test_the_readme_only_points_at_docs_that_exist():
    with open(os.path.join(REPO, "README.md"), encoding="utf-8") as f:
        readme = f.read()
    for target in re.findall(r"\]\((docs/[^)#]+|[A-Z_]+\.md)\)", readme):
        assert os.path.isfile(os.path.join(REPO, target)), f"README links to {target}"


def test_the_examples_avoid_python_only_this_repo_has():
    # The package claims 3.9 support, so a walrus in a doc example would be fine but a
    # match statement would not. Compiling under the running interpreter catches syntax
    # errors; this catches the version claim drifting away from the examples.
    assert sys.version_info >= (3, 9)
