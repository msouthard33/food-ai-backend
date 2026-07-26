"""Snapshot / format + contract-guard tests for the doctor/patient summary
report SIGNAL SECTION (Wave 2, Pillar 4).

Adds three things the shipped ``test_summary_signal.py`` did not cover:

  1. **Render snapshot for every ``direction`` x ``confidence`` combination** — the
     rendered signal section must show the right tier wording, must carry the
     exposed/control/episodes sample-size numbers on EVERY row, and must never leak a
     forbidden token (score / posterior / P(beta / credible / Bayes / lag / 24-72h).
  2. **Demotion honesty at the render layer** — every demoted row renders its caveat
     and is marked (``*``); only an undemoted ``trigger`` row is a confirmed headline.
  3. **A strengthened, AST-based contract guard** across the whole report/export layer
     (``summary_report_service`` + ``reports`` + ``summary_signal`` outside the
     ``assoc_guardrail`` seam call) — no current-engine OUTPUT-SHAPE identifier and no
     literal lag-window (24/36/48/72h) may appear in CODE. This is the mechanical
     enforcement that lets the rewired engine drop in with zero template changes.

These operate on the flowables ``_signal_section`` builds (extracting text from
``Paragraph.getPlainText()`` and ``Table._cellvalues``) so they assert on real rendered
output, not on the dataclass.
"""

import ast
import re
from itertools import product

import pytest

from app.services import summary_report_service as srs
from app.services import summary_signal as ss
from app.services.summary_report_service import (
    _CONFIDENCE_LABEL,
    _DIRECTION_LABEL,
    _signal_section,
)
from app.services.summary_signal import SummarySignalRow, _demotion


# ── helpers ─────────────────────────────────────────────────────────────────────

def _render_text(rows):
    """Render the signal section and return (full_text, table_cell_rows, paragraphs)."""
    from app.services.clinician_report_service import _styles

    story: list = []
    _signal_section(story, _styles(), rows)

    paragraphs: list[str] = []
    table_rows: list[list[str]] = []
    for flow in story:
        if hasattr(flow, "getPlainText"):
            paragraphs.append(flow.getPlainText())
        elif hasattr(flow, "_cellvalues"):
            for r in flow._cellvalues:
                table_rows.append([str(c) for c in r])
    full = "\n".join(paragraphs + ["\t".join(r) for r in table_rows])
    return full, table_rows, paragraphs


def _row(direction, confidence, *, demoted=None, reason=None,
         exposed=7, control=11, episodes=5, testable=True):
    if demoted is None:
        demoted, reason = _demotion(direction, confidence)
    if testable:
        or_, lo, hi, pv, test = 2.5, 1.4, 4.1, 0.01, "fisher"
    else:
        or_ = lo = hi = pv = None
        test = "skipped"
    return SummarySignalRow(
        food_name="Cheddar",
        direction=direction,
        confidence=confidence,
        demoted=demoted,
        demotion_reason=reason,
        exposed_count=exposed,
        control_count=control,
        symptom_episodes=episodes,
        odds_ratio=or_, ci_low=lo, ci_high=hi, p_value=pv, test=test,
    )


_DIRECTIONS = ("trigger", "protective", "inconclusive")
_CONFIDENCES = ("strong", "moderate", "preliminary", "insufficient")

#: Rendered-string bans (design spec §9 + signal contract §0). Case-insensitive.
_FORBIDDEN_RENDER = [
    "score", "posterior", "p(beta", "credible", "bayes", "combined_score",
    "hierarchical", "trigger_probability", "_driver_for_food",
]
_FORBIDDEN_RENDER_RE = re.compile("|".join(re.escape(t) for t in _FORBIDDEN_RENDER),
                                  re.IGNORECASE)
#: literal lag windows as an hour window, or the bare word "lag".
_LAG_RE = re.compile(r"\b(24|36|48|72)\s*h\b|\blag\b", re.IGNORECASE)


# ── 1. snapshot: every direction x confidence renders the right tier wording ─────

@pytest.mark.parametrize("direction,confidence", list(product(_DIRECTIONS, _CONFIDENCES)))
def test_every_direction_confidence_combo_renders_correct_tier_wording(direction, confidence):
    row = _row(direction, confidence)
    full, table_rows, _ = _render_text([row])

    # exactly one data row under the header
    data = [r for r in table_rows if r and r[0] == "Cheddar"]
    assert len(data) == 1, "one rendered row per signal"
    cells = data[0]

    # the direction label leads the "Reading" cell (may carry a demotion '*')
    assert _DIRECTION_LABEL[direction] in cells[1]
    # the ordinal confidence tier word is rendered verbatim
    assert _CONFIDENCE_LABEL[confidence] in cells[2]
    # no engine-shape / probability / lag vocabulary leaks into rendered text
    assert not _FORBIDDEN_RENDER_RE.search(full), f"forbidden token in: {full!r}"
    assert not _LAG_RE.search(full), f"lag vocabulary leaked: {full!r}"


def test_render_labels_cover_all_contract_enum_values():
    # every enum value the seam can emit has a plain-English render label
    assert set(_DIRECTION_LABEL) == set(_DIRECTIONS)
    assert set(_CONFIDENCE_LABEL) == set(_CONFIDENCES)


# ── 2. sample-size line present on EVERY row (Pillar 5, mandatory) ───────────────

@pytest.mark.parametrize("direction,confidence", list(product(_DIRECTIONS, _CONFIDENCES)))
def test_sample_size_numbers_present_on_every_row(direction, confidence):
    row = _row(direction, confidence, exposed=7, control=11, episodes=5)
    _, table_rows, _ = _render_text([row])
    cells = next(r for r in table_rows if r and r[0] == "Cheddar")
    # exposed / control / symptom-episode counts each render as their own cell
    assert "7" in cells and "11" in cells and "5" in cells, cells


def test_sample_size_shows_zero_control_days_not_testable_hidden():
    # the contrast-collapse case (0 control days) must still surface the honest denom
    row = _row("trigger", "moderate", exposed=18, control=0, episodes=18)
    _, table_rows, _ = _render_text([row])
    cells = next(r for r in table_rows if r and r[0] == "Cheddar")
    assert "18" in cells and "0" in cells


# ── 3. demotion honesty at the render layer ─────────────────────────────────────

def test_only_undemoted_trigger_is_a_confirmed_headline():
    for direction, confidence in product(_DIRECTIONS, _CONFIDENCES):
        row = _row(direction, confidence)
        _, table_rows, _ = _render_text([row])
        cells = next(r for r in table_rows if r and r[0] == "Cheddar")
        marked = cells[1].endswith("*")
        confirmed = (direction == "trigger" and confidence in ("strong", "moderate"))
        # demoted rows are always marked; confirmed headlines never are
        assert marked == (not confirmed), (direction, confidence, cells[1])


def test_every_demoted_row_renders_its_caveat():
    demoted_combos = [
        (d, c) for d, c in product(_DIRECTIONS, _CONFIDENCES)
        if _demotion(d, c)[0]
    ]
    for direction, confidence in demoted_combos:
        row = _row(direction, confidence)
        full, table_rows, paragraphs = _render_text([row])
        cells = next(r for r in table_rows if r and r[0] == "Cheddar")
        assert cells[1].endswith("*"), (direction, confidence)
        # the demotion reason surfaces as a plain-English caveat paragraph
        assert row.demotion_reason is not None
        assert any("shown but not confirmed" in p for p in paragraphs), full
        assert row.demotion_reason in full


def test_protective_strong_never_renders_as_confirmed_trigger():
    # the Garlic-mis-attribution guard: a protective food, even at 'strong', is demoted
    row = _row("protective", "strong")
    _, table_rows, _ = _render_text([row])
    cells = next(r for r in table_rows if r and r[0] == "Cheddar")
    assert cells[1].endswith("*")
    assert _DIRECTION_LABEL["trigger"] not in cells[1]


def test_empty_signal_list_renders_no_dead_end():
    full, _, paragraphs = _render_text([])
    assert paragraphs, "empty state must render explanatory copy, not a blank"
    assert "No food reached the reporting threshold" in full
    assert "No data" not in full  # patient-dignity: never a dead-end wall


# ── 4. strengthened contract guard (AST, whole report/export layer) ─────────────

#: engine OUTPUT-SHAPE identifiers that must never appear as code in the report layer.
_BANNED_ENGINE_IDENTIFIERS = {
    "combined_score", "_driver_for_food", "trigger_probability", "get_suspect_foods",
}
#: engine modules the report/export layer must NOT import (it goes through the seam).
_BANNED_ENGINE_MODULES = {
    "hierarchical_trigger", "bayesian_trigger", "trigger_service",
    "insights", "trigger_engine",
}
#: symbols the provider seam is allowed to pull from the interim engine modules.
_SEAM_ALLOWED_IMPORTS = {
    "GuardrailResult", "analyze_association_guardrail",
    "DEFAULT_LOOKBACK_DAYS", "food_components_by_name",
}


def _module_ast(module):
    import inspect
    return ast.parse(inspect.getsource(module))


def _code_identifiers(tree):
    """All Name ids + Attribute attrs actually used in CODE (docstrings excluded)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def _int_literals(tree):
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, int)
    }


def _imports(tree):
    """List of (module_tail, imported_name) for every import in the tree."""
    out: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            tail = node.module.split(".")[-1]
            for alias in node.names:
                out.append((tail, alias.name))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                out.append((alias.name.split(".")[-1], alias.name))
    return out


@pytest.mark.parametrize("module", [srs, ss])
def test_no_engine_output_shape_identifier_in_code(module):
    used = _code_identifiers(_module_ast(module))
    leaked = used & _BANNED_ENGINE_IDENTIFIERS
    assert not leaked, f"{module.__name__} references engine output shape: {leaked}"


@pytest.mark.parametrize("module", [srs, ss])
def test_no_literal_lag_window_in_code(module):
    lag_literals = _int_literals(_module_ast(module)) & {24, 36, 48, 72}
    assert not lag_literals, (
        f"{module.__name__} hard-codes a lag-window literal {lag_literals}; "
        "lag/window is an engine detail, never a report field"
    )


def test_report_export_layer_does_not_import_any_trigger_engine():
    # summary_report_service + reports.py must reach the signal only via the seam.
    from app.routers import reports as reports_router
    for module in (srs, reports_router):
        for tail, name in _imports(_module_ast(module)):
            assert tail not in _BANNED_ENGINE_MODULES, (
                f"{module.__name__} imports {name} from engine module {tail!r}; "
                "the report layer must consume ONLY build_summary_signal_rows"
            )


def test_provider_seam_only_pulls_allowlisted_symbols_from_engine_modules():
    # the seam MAY touch the interim engine modules, but only for the sanctioned
    # KB-lookup / lookback-constant / classical-guardrail symbols — never scoring.
    for tail, name in _imports(_module_ast(ss)):
        if tail in _BANNED_ENGINE_MODULES:
            assert name in _SEAM_ALLOWED_IMPORTS, (
                f"summary_signal imports {name!r} from {tail!r} — not on the seam "
                "allowlist; the seam must not import engine scoring symbols"
            )


def test_signal_row_is_the_only_shape_the_render_layer_reads():
    # every attribute the render layer reads off a signal row must be a real
    # SummarySignalRow field — proving the template speaks ONLY the stable contract.
    render_src = _module_ast(srs)
    row_fields = set(SummarySignalRow.__dataclass_fields__)
    read_attrs: set[str] = set()
    for node in ast.walk(render_src):
        # attributes read off the loop variable `r` in _signal_section
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "r":
                read_attrs.add(node.attr)
    assert read_attrs, "expected the render layer to read row fields"
    stray = read_attrs - row_fields
    assert not stray, f"render layer reads non-contract fields off the row: {stray}"
