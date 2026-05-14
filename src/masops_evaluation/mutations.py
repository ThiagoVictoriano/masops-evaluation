"""Deterministic synthetic mutations of unified diffs.

Each public mutation function rewrites a gold patch into a defective variant
designed to exercise a specific aspect of the MAS-Ops Guardian's behaviour:

- :func:`remove_random_addition` removes one ``+`` line, simulating an
  incomplete fix.
- :func:`invert_conditional` flips a single boolean / comparison operator,
  simulating inverted logic.
- :func:`expand_scope` injects an unrelated new file modification on top of
  the gold patch, testing scope discipline.
- :func:`remove_critical_line` removes the ``+`` line that looks most
  load-bearing (``return`` / ``raise`` / ``assert`` keywords, otherwise the
  longest addition), simulating an effect-less fix.

Each mutation is **deterministic** given the same ``(patch, seed)`` pair and
produces a unified diff that should still pass ``git apply --check`` against
the same base commit (file-level structure preserved, hunk counts
recalculated after any line additions or removals).

Mutations that cannot be applied to a given input (e.g.
``invert_conditional`` on a patch with no invertible operators) raise
:class:`ValueError` with a clear message.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Literal


# --- Public type aliases ----------------------------------------------------

MutationType = Literal[
    "remove_addition",
    "invert_conditional",
    "expand_scope",
    "remove_critical_line",
]

MUTATION_TYPES: tuple[MutationType, ...] = (
    "remove_addition",
    "invert_conditional",
    "expand_scope",
    "remove_critical_line",
)


# --- Diff parser ----------------------------------------------------------

_HUNK_HEADER_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$"
)


@dataclass
class _Hunk:
    """A single hunk inside a unified diff section."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    context: str
    lines: list[str] = field(default_factory=list)

    def render_header(self) -> str:
        return (
            f"@@ -{self.old_start},{self.old_count} "
            f"+{self.new_start},{self.new_count} @@{self.context}"
        )

    def recalculate_counts(self) -> None:
        """Recompute ``old_count`` / ``new_count`` from the body lines."""
        old = 0
        new = 0
        for line in self.lines:
            if not line:
                # An empty body line is treated as a context line ("" == " ").
                old += 1
                new += 1
                continue
            head = line[0]
            if head == "+":
                new += 1
            elif head == "-":
                old += 1
            elif head == " ":
                old += 1
                new += 1
            # "\ No newline at end of file" markers start with "\" — skip.
        self.old_count = old
        self.new_count = new


@dataclass
class _FileSection:
    """One file modification within a patch (preamble + hunks)."""

    preamble: list[str] = field(default_factory=list)
    hunks: list[_Hunk] = field(default_factory=list)


@dataclass
class _Patch:
    """A parsed unified diff with one or more file sections."""

    sections: list[_FileSection] = field(default_factory=list)

    def serialize(self) -> str:
        out: list[str] = []
        for section in self.sections:
            out.extend(section.preamble)
            for hunk in section.hunks:
                out.append(hunk.render_header())
                out.extend(hunk.lines)
        return "\n".join(out) + ("\n" if out else "")


def _parse_patch(text: str) -> _Patch:
    """Parse a unified diff into the internal structured representation.

    The parser is intentionally permissive — it accepts patches without
    ``diff --git`` headers (lone ``--- / +++`` form) and patches with
    trailing context outside hunks (preamble-only sections).
    """
    patch = _Patch()
    current_section: _FileSection | None = None
    current_hunk: _Hunk | None = None

    for line in text.splitlines():
        if line.startswith("diff --git"):
            current_section = _FileSection(preamble=[line])
            current_hunk = None
            patch.sections.append(current_section)
            continue

        m = _HUNK_HEADER_RE.match(line)
        if m:
            if current_section is None:
                current_section = _FileSection()
                patch.sections.append(current_section)
            current_hunk = _Hunk(
                old_start=int(m.group(1)),
                old_count=int(m.group(2)) if m.group(2) else 1,
                new_start=int(m.group(3)),
                new_count=int(m.group(4)) if m.group(4) else 1,
                context=m.group(5),
            )
            current_section.hunks.append(current_hunk)
            continue

        if current_hunk is not None:
            current_hunk.lines.append(line)
        else:
            if current_section is None:
                current_section = _FileSection()
                patch.sections.append(current_section)
            current_section.preamble.append(line)

    return patch


# --- Iteration helpers ----------------------------------------------------

def _iter_added_lines(patch: _Patch) -> list[tuple[int, int, int, str]]:
    """Return ``(section_idx, hunk_idx, line_idx, content)`` for each ``+`` line.

    ``content`` is the line without the leading ``+``.
    """
    out: list[tuple[int, int, int, str]] = []
    for si, section in enumerate(patch.sections):
        for hi, hunk in enumerate(section.hunks):
            for li, line in enumerate(hunk.lines):
                if line.startswith("+") and not line.startswith("+++"):
                    out.append((si, hi, li, line[1:]))
    return out


# --- Mutation 1: remove_random_addition -----------------------------------

def remove_random_addition(patch: str, seed: int) -> str:
    """Remove one randomly chosen ``+`` line from the diff.

    Args:
        patch: A unified diff (the gold patch).
        seed: Random seed; the same value always yields the same result.

    Returns:
        A new unified diff with one fewer added line and recomputed hunk
        counts.

    Raises:
        ValueError: if the patch contains no removable ``+`` lines.
    """
    parsed = _parse_patch(patch)
    candidates = _iter_added_lines(parsed)
    if not candidates:
        raise ValueError(
            "remove_random_addition: patch has no '+' addition lines to remove."
        )
    rng = random.Random(seed)
    si, hi, li, _ = rng.choice(candidates)
    hunk = parsed.sections[si].hunks[hi]
    del hunk.lines[li]
    hunk.recalculate_counts()
    return parsed.serialize()


# --- Mutation 2: invert_conditional ---------------------------------------

# Order matters: longer / more specific patterns first so that re alternation
# returns the right match at each position.
_INVERT_MAP: dict[str, str] = {
    " is not ": " is ",
    " is ": " is not ",
    ">=": "<",
    "<=": ">",
    "==": "!=",
    "!=": "==",
    ">": "<=",
    "<": ">=",
    " and ": " or ",
    " or ": " and ",
}
_INVERT_PATTERN = re.compile(
    "|".join(re.escape(op) for op in _INVERT_MAP.keys())
)


def invert_conditional(patch: str, seed: int) -> str:
    """Flip a single conditional operator inside a ``+`` line.

    Targets the binary operators ``==``, ``!=``, ``<``, ``>``, ``<=``, ``>=``
    and the keyword operators ``and``, ``or``, ``is``, ``is not``. Only
    ``+`` lines are considered, so the mutation always changes the patch's
    behaviour rather than the surrounding context.

    Args:
        patch: A unified diff (the gold patch).
        seed: Random seed.

    Returns:
        A new unified diff with exactly one operator inverted.

    Raises:
        ValueError: if the patch has no addition line containing an
            invertible operator.
    """
    parsed = _parse_patch(patch)
    occurrences: list[tuple[int, int, int, int, str]] = []
    # (section_idx, hunk_idx, line_idx, match_start, op)
    for si, hi, li, content in _iter_added_lines(parsed):
        for match in _INVERT_PATTERN.finditer(content):
            occurrences.append((si, hi, li, match.start(), match.group(0)))

    if not occurrences:
        raise ValueError(
            "invert_conditional: no addition line contains an invertible operator."
        )

    rng = random.Random(seed)
    si, hi, li, start, op = rng.choice(occurrences)
    hunk = parsed.sections[si].hunks[hi]
    original = hunk.lines[li]
    # original = "+" + content; adjust start to the full-line offset (skip '+')
    new_line = (
        original[: 1 + start] + _INVERT_MAP[op] + original[1 + start + len(op) :]
    )
    hunk.lines[li] = new_line
    # Operator inversion is in-place; hunk counts unchanged but recompute for safety.
    hunk.recalculate_counts()
    return parsed.serialize()


# --- Mutation 3: expand_scope ---------------------------------------------

def expand_scope(patch: str, seed: int) -> str:
    """Append an unrelated new-file modification to test Guardian scope discipline.

    The injected section creates a brand-new debug file with a clearly
    labelled comment and a constant assignment. It is appended after the
    last existing file section so the original fix is preserved verbatim.

    The seed influences only the suffix of the debug file's name (so two
    different seeds produce different paths), keeping the result fully
    deterministic.

    Args:
        patch: A unified diff (the gold patch). Must be non-empty.
        seed: Random seed used to derive the debug file's stable suffix.

    Returns:
        A new unified diff that contains the original sections plus one
        extra new-file section.

    Raises:
        ValueError: if the original patch is empty (no sections to expand).
    """
    parsed = _parse_patch(patch)
    if not parsed.sections:
        raise ValueError(
            "expand_scope: cannot append a scope-expansion section to an empty patch."
        )

    # Stable, deterministic suffix derived from the seed.
    suffix = hashlib.sha256(str(seed).encode("utf-8")).hexdigest()[:8]
    debug_path = f"_masops_eval_scope_{suffix}.py"
    body_lines = [
        f"+# eval-mutation: scope expansion (seed={seed})",
        "+_MASOPS_EVAL_SCOPE_FLAG = True",
    ]

    new_section = _FileSection(
        preamble=[
            f"diff --git a/{debug_path} b/{debug_path}",
            "new file mode 100644",
            "--- /dev/null",
            f"+++ b/{debug_path}",
        ],
        hunks=[
            _Hunk(
                old_start=0,
                old_count=0,
                new_start=1,
                new_count=len(body_lines),
                context="",
                lines=body_lines,
            )
        ],
    )
    parsed.sections.append(new_section)
    return parsed.serialize()


# --- Mutation 4: remove_critical_line -------------------------------------

_CRITICAL_KEYWORD_RE = re.compile(r"\b(return|raise|assert)\b")


def remove_critical_line(patch: str, seed: int) -> str:
    """Remove the ``+`` line that appears most load-bearing.

    Heuristic priorities:

    1. ``+`` lines containing ``return``, ``raise``, or ``assert``
       (matched as whole words).
    2. Among those, the longest stripped content wins.
    3. If no critical keyword is found, fall back to the longest ``+`` line.
    4. Remaining ties are broken deterministically with ``random.Random(seed)``.

    Args:
        patch: A unified diff (the gold patch).
        seed: Random seed (used only as tie-breaker).

    Returns:
        A new unified diff with the selected ``+`` line removed and the
        affected hunk's counts recomputed.

    Raises:
        ValueError: if the patch contains no ``+`` lines.
    """
    parsed = _parse_patch(patch)
    candidates = _iter_added_lines(parsed)
    if not candidates:
        raise ValueError(
            "remove_critical_line: patch has no '+' addition lines to remove."
        )

    scored: list[tuple[int, int, int, int, int]] = []
    # (section_idx, hunk_idx, line_idx, priority, length)
    for si, hi, li, content in candidates:
        stripped = content.strip()
        priority = 2 if _CRITICAL_KEYWORD_RE.search(stripped) else 1
        scored.append((si, hi, li, priority, len(stripped)))

    max_priority = max(c[3] for c in scored)
    top = [c for c in scored if c[3] == max_priority]
    max_length = max(c[4] for c in top)
    top = [c for c in top if c[4] == max_length]

    rng = random.Random(seed)
    si, hi, li, _, _ = rng.choice(top)
    hunk = parsed.sections[si].hunks[hi]
    del hunk.lines[li]
    hunk.recalculate_counts()
    return parsed.serialize()


# --- Coordinator ---------------------------------------------------------

def generate_synthetic_mutation(
    gold_patch: str,
    mutation_type: MutationType,
    seed: int,
) -> str:
    """Dispatch to the requested mutation by name.

    Args:
        gold_patch: The pristine SWE-bench gold patch to mutate.
        mutation_type: One of :data:`MUTATION_TYPES`.
        seed: Deterministic seed forwarded to the mutation.

    Returns:
        The mutated unified diff.

    Raises:
        ValueError: if ``mutation_type`` is unknown, or if the chosen
            mutation cannot be applied to ``gold_patch``.
    """
    dispatch = {
        "remove_addition": remove_random_addition,
        "invert_conditional": invert_conditional,
        "expand_scope": expand_scope,
        "remove_critical_line": remove_critical_line,
    }
    if mutation_type not in dispatch:
        raise ValueError(
            f"Unknown mutation_type {mutation_type!r}; expected one of {MUTATION_TYPES}."
        )
    return dispatch[mutation_type](gold_patch, seed)


def source_label_for_mutation(mutation_type: MutationType) -> str:
    """Map a :data:`MutationType` to its ``candidate_patch_source`` label.

    Used by :mod:`masops_evaluation.run_evaluation` when persisting the
    :class:`~masops_evaluation.schemas.ExecutionRecord`.
    """
    return f"synthetic_{mutation_type}"
