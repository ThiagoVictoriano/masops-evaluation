"""Tests for the synthetic mutation pipeline.

Three concerns covered per mutation:

* **Determinism.** Same ``(patch, seed)`` always returns the same diff.
* **Diff validity.** The output applies cleanly against the original
  fixture via ``git apply --check``.
* **Inapplicability.** Patches that lack the mutation's target raise
  :class:`ValueError`.

Diff validity tests need ``git`` available on ``PATH``; they are skipped
otherwise so the suite stays portable.
"""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from masops_evaluation.mutations import (
    MUTATION_TYPES,
    expand_scope,
    generate_synthetic_mutation,
    invert_conditional,
    remove_critical_line,
    remove_random_addition,
    source_label_for_mutation,
)


# --- Fixtures -------------------------------------------------------------

_TARGET_FILENAME = "target.py"

_TARGET_BEFORE = textwrap.dedent(
    """\
    def compute(value):
        if value > 0:
            return value
        return None
    """
)

# Diff that adds a new negative branch — gives us multiple '+' lines,
# a conditional operator (`<`), and a critical line (`return -value`).
_GOLD_PATCH = textwrap.dedent(
    """\
    diff --git a/target.py b/target.py
    --- a/target.py
    +++ b/target.py
    @@ -1,4 +1,6 @@
     def compute(value):
         if value > 0:
             return value
    +    if value < 0:
    +        return -value
         return None
    """
)

# Patch with no '+' lines (just context + a deletion).
_DELETION_ONLY_PATCH = textwrap.dedent(
    """\
    diff --git a/target.py b/target.py
    --- a/target.py
    +++ b/target.py
    @@ -1,4 +1,3 @@
     def compute(value):
         if value > 0:
    -        return value
         return None
    """
)

# Patch where all '+' lines are plain assignments — no invertible operator.
_NO_OPERATORS_PATCH = textwrap.dedent(
    """\
    diff --git a/target.py b/target.py
    --- a/target.py
    +++ b/target.py
    @@ -1,4 +1,5 @@
     def compute(value):
         if value > 0:
             return value
    +    counter = counter + 1
         return None
    """
)


_GIT_AVAILABLE = shutil.which("git") is not None


# --- Helpers --------------------------------------------------------------

def _make_repo(tmp_path: Path) -> Path:
    """Create a fresh git repo containing the fixture target file."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@masops.eval"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "masops-eval-test"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / _TARGET_FILENAME).write_text(_TARGET_BEFORE, encoding="utf-8")
    subprocess.run(["git", "add", _TARGET_FILENAME], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"],
        cwd=tmp_path,
        check=True,
    )
    return tmp_path


def _assert_patch_applies(repo: Path, patch_text: str) -> None:
    """Assert that ``patch_text`` passes ``git apply --check`` in ``repo``."""
    patch_file = repo / "_mutated.patch"
    patch_file.write_text(patch_text, encoding="utf-8")
    result = subprocess.run(
        ["git", "apply", "--check", str(patch_file)],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"git apply --check rejected the mutated patch.\n"
        f"stderr:\n{result.stderr}\n---patch---\n{patch_text}"
    )


# --- Determinism ----------------------------------------------------------

@pytest.mark.parametrize(
    "mutation_fn",
    [remove_random_addition, invert_conditional, expand_scope, remove_critical_line],
)
def test_mutation_is_deterministic(mutation_fn) -> None:  # type: ignore[no-untyped-def]
    first = mutation_fn(_GOLD_PATCH, seed=42)
    second = mutation_fn(_GOLD_PATCH, seed=42)
    assert first == second


def test_different_seeds_can_produce_different_outputs() -> None:
    # Across the four mutations, at least one should be seed-sensitive
    # (remove_random_addition has two candidate '+' lines, so seed matters).
    s1 = remove_random_addition(_GOLD_PATCH, seed=1)
    s2 = remove_random_addition(_GOLD_PATCH, seed=2)
    s3 = remove_random_addition(_GOLD_PATCH, seed=3)
    assert len({s1, s2, s3}) >= 2


# --- Diff validity via `git apply --check` --------------------------------

@pytest.mark.skipif(not _GIT_AVAILABLE, reason="git not on PATH")
def test_remove_random_addition_produces_valid_diff(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    mutated = remove_random_addition(_GOLD_PATCH, seed=42)
    _assert_patch_applies(repo, mutated)


@pytest.mark.skipif(not _GIT_AVAILABLE, reason="git not on PATH")
def test_invert_conditional_produces_valid_diff(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    mutated = invert_conditional(_GOLD_PATCH, seed=42)
    _assert_patch_applies(repo, mutated)


@pytest.mark.skipif(not _GIT_AVAILABLE, reason="git not on PATH")
def test_expand_scope_produces_valid_diff(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    mutated = expand_scope(_GOLD_PATCH, seed=42)
    _assert_patch_applies(repo, mutated)


@pytest.mark.skipif(not _GIT_AVAILABLE, reason="git not on PATH")
def test_remove_critical_line_produces_valid_diff(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    mutated = remove_critical_line(_GOLD_PATCH, seed=42)
    _assert_patch_applies(repo, mutated)


# --- Inapplicability raises ValueError ------------------------------------

def test_remove_random_addition_rejects_patch_without_additions() -> None:
    with pytest.raises(ValueError, match="no '\\+' addition"):
        remove_random_addition(_DELETION_ONLY_PATCH, seed=42)


def test_invert_conditional_rejects_patch_without_operators() -> None:
    with pytest.raises(ValueError, match="invertible operator"):
        invert_conditional(_NO_OPERATORS_PATCH, seed=42)


def test_expand_scope_rejects_empty_patch() -> None:
    with pytest.raises(ValueError, match="empty patch"):
        expand_scope("", seed=42)


def test_remove_critical_line_rejects_patch_without_additions() -> None:
    with pytest.raises(ValueError, match="no '\\+' addition"):
        remove_critical_line(_DELETION_ONLY_PATCH, seed=42)


# --- Behaviour spot-checks -----------------------------------------------

def test_remove_random_addition_removes_exactly_one_added_line() -> None:
    original_adds = [
        line for line in _GOLD_PATCH.splitlines() if line.startswith("+") and not line.startswith("+++")
    ]
    mutated = remove_random_addition(_GOLD_PATCH, seed=42)
    mutated_adds = [
        line for line in mutated.splitlines() if line.startswith("+") and not line.startswith("+++")
    ]
    assert len(mutated_adds) == len(original_adds) - 1


def test_invert_conditional_changes_an_added_line() -> None:
    mutated = invert_conditional(_GOLD_PATCH, seed=42)
    # The original "+    if value < 0:" should be flipped to ">=".
    added_lines = [
        line for line in mutated.splitlines() if line.startswith("+") and not line.startswith("+++")
    ]
    assert any(">=" in line or "<=" in line or "==" in line or "!=" in line for line in added_lines), (
        f"No invertible operator change observed in: {added_lines}"
    )


def test_expand_scope_appends_section_without_touching_original() -> None:
    mutated = expand_scope(_GOLD_PATCH, seed=42)
    assert _GOLD_PATCH.strip() in mutated, "Original sections should be preserved verbatim"
    assert "_masops_eval_scope_" in mutated
    assert "new file mode" in mutated


def test_remove_critical_line_prefers_return_keyword() -> None:
    """The patch contains ``return -value`` which should be the priority pick."""
    mutated = remove_critical_line(_GOLD_PATCH, seed=42)
    assert "return -value" not in mutated, (
        "Expected the critical 'return -value' addition to be removed"
    )


# --- Coordinator and rotation --------------------------------------------

def test_generate_synthetic_mutation_dispatches_by_type() -> None:
    for mtype in MUTATION_TYPES:
        # We use the gold patch which can satisfy all four mutations.
        mutated = generate_synthetic_mutation(_GOLD_PATCH, mtype, seed=7)
        assert mutated  # non-empty
        assert mutated != _GOLD_PATCH


def test_generate_synthetic_mutation_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unknown mutation_type"):
        generate_synthetic_mutation(_GOLD_PATCH, "nonsense", seed=1)  # type: ignore[arg-type]


def test_rotation_covers_all_four_types() -> None:
    """A naive rotation through MUTATION_TYPES touches every type exactly once per cycle."""
    seen = []
    for i in range(len(MUTATION_TYPES)):
        mtype = MUTATION_TYPES[i % len(MUTATION_TYPES)]
        seen.append(mtype)
    assert set(seen) == set(MUTATION_TYPES)


def test_source_label_for_mutation_round_trip() -> None:
    for mtype in MUTATION_TYPES:
        assert source_label_for_mutation(mtype) == f"synthetic_{mtype}"
