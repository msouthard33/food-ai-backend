"""Snapshot / format + contract-guard tests for the doctor/patient summary
report SIGNAL SECTION (Wave 2, Pillar 4).

Reconciled 2026-07-26 (QA-SUMRPT-1) to the CANONICAL grouped **definition-list**
(``W2_summary_report_layout.md`` §5 / §8.3) — the deprecated 7-column table is gone.
These tests assert on the real rendered flowables (``Paragraph.getPlainText()``):

  1. **Every ``direction`` x ``confidence`` renders its FINAL §5.2 readout** — the bold
     tier word leads, the pinned verb "line up" is present, the food name is embedded,
     the mandatory §5.3 sample-size line rides on EVERY row, and no forbidden token
     (score / posterior / P(beta / credible / Bayes / lag / 24-72h) leaks.
  2. **Grouping + demotion honesty** — rows group under the exact §5.1 subheadings in
     the fixed trigger -> protective -> inconclusive order; demoted rows sink to the
     bottom of their group, render at reduced weight, and carry their §5.4 caveat; an
     undemoted trigger is the ONLY confirmed 'Worth discussing' headline (no caveat).
  3. **The AST-based contract guard** across the whole report/export layer — no
     current-engine OUTPUT-SHAPE identifier and no literal lag-window (24/36/48/72h)
     may appear in CODE, and the render layer reads ONLY ``SummarySignalRow`` fields.
"""

import ast
import re
from itertools import product

import pytest

from app.services import summary_report_service as srs
from app.services import summary_signal as ss
from app.services.summary_report_service import (
    _CAVEAT_BY_REASON,
    _CAVEAT_DISAGREE,
    _CAVEAT_GENERIC,
    _CAVEAT_MIXED,
    _CAVEAT_SMALL_SAMPLE,
    _GROUP_HEADING,
    _demoted_caveat,
    _signal_section,
)
from app.services.summary_signal import (
    DEMOTION_REASON_CODES,
    DemotionReason,
    SummarySignalRow,
    _demotion,
)


# ── helpers ─────────────────────────────────────────────────────────────────────

def _render_text(rows, *, clinician_detail=False):
    """Render the signal section and return (full_text, ordered_paragraph_texts)."""
    from app.services.clinician_report_service import _styles

    story: list = []
    _signal_section(story, _styles(), rows, clinician_detail=clinician_detail)

    paragraphs: list[str] = [
        flow.getPlainText() for flow in story if hasattr(flow, "getPlainText")
    ]
    return "\n".join(paragraphs), paragraphs


def _row(direction, confidence, *, demoted=None, reason=None,
         exposed=7, control=11, episodes=5, testable=True, food="Cheddar"):
    if demoted is None:
        demoted, reason = _demotion(direction, confidence)
    if testable:
        or_, lo, hi, pv, test = 2.5, 1.4, 4.1, 0.01, "fisher"
    else:
        or_ = lo = hi = pv = None
        test = "skipped"
    return SummarySignalRow(
        food_name=food,
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

#: FINAL §5.2 leading tier word for each (direction, confidence). Bold word always
#: leads the readout sentence (D9 doctrine: a number never leads).
_TIER_WORD = {
    ("trigger", "strong"): "Worth discussing",
    ("trigger", "moderate"): "Some evidence",
    ("trigger", "preliminary"): "Early signal",
    ("trigger", "insufficient"): "Not enough yet",
    ("protective", "strong"): "Sits well so far",
    ("protective", "moderate"): "Looks okay so far",
    ("protective", "preliminary"): "Leaning okay",
    ("protective", "insufficient"): "Not enough yet",
}
_INCONCLUSIVE_TIER = "No clear pattern"

_ALL_CAVEATS = (_CAVEAT_SMALL_SAMPLE, _CAVEAT_DISAGREE, _CAVEAT_MIXED, _CAVEAT_GENERIC)

#: Rendered-string bans (design spec §9 + signal contract §0). Case-insensitive.
_FORBIDDEN_RENDER = [
    "score", "posterior", "p(beta", "credible", "bayes", "combined_score",
    "hierarchical", "trigger_probability", "_driver_for_food",
]
_FORBIDDEN_RENDER_RE = re.compile("|".join(re.escape(t) for t in _FORBIDDEN_RENDER),
                                  re.IGNORECASE)
#: literal lag windows as an hour window, or the bare word "lag".
_LAG_RE = re.compile(r"\b(24|36|48|72)\s*h\b|\blag\b", re.IGNORECASE)


def _expected_tier(direction, confidence):
    return _INCONCLUSIVE_TIER if direction == "inconclusive" else _TIER_WORD[(direction, confidence)]


# ── 1. snapshot: every direction x confidence renders the FINAL §5.2 readout ─────

@pytest.mark.parametrize("direction,confidence", list(product(_DIRECTIONS, _CONFIDENCES)))
def test_every_direction_confidence_renders_final_readout(direction, confidence):
    row = _row(direction, confidence)
    full, paragraphs = _render_text([row])

    tier = _expected_tier(direction, confidence)
    headings = set(_GROUP_HEADING.values())
    # the bold tier word LEADS the readout sentence (a number never leads). The
    # readout carries the em-dash separator; exclude the same-prefixed group heading.
    readout = next(
        (p for p in paragraphs
         if p.lstrip().startswith(tier) and p not in headings and "—" in p),
        None,
    )
    assert readout is not None, f"no readout led by {tier!r} in {paragraphs!r}"
    # the food name is embedded in the readout sentence
    assert "Cheddar" in readout
    # never the banned causal verbs, anywhere in the section
    assert "caused" not in full.lower()
    assert "associated with" not in full.lower()
    # no engine-shape / probability / lag vocabulary leaks into rendered text
    assert not _FORBIDDEN_RENDER_RE.search(full), f"forbidden token in: {full!r}"
    assert not _LAG_RE.search(full), f"lag vocabulary leaked: {full!r}"


def test_pinned_association_verb_used_never_causal():
    # "line up with" is the pinned non-causal stand-in — present across the readout
    # matrix, and the banned causal verbs never appear.
    for direction, confidence in product(_DIRECTIONS, _CONFIDENCES):
        full, _ = _render_text([_row(direction, confidence)])
        assert "caused" not in full.lower()
        assert "associated with" not in full.lower()
    # the pinned verb surfaces on the readouts that assert a pattern
    strong_trig, _ = _render_text([_row("trigger", "strong")])
    assert re.search(r"\bline(d|s)?\b|\blining\b", strong_trig)


def test_final_approved_strings_render_verbatim():
    # spot-check the two PINNED verbatim strings survive rendering exactly (tone gate)
    strong_trig, _ = _render_text([_row("trigger", "strong", food="Cheddar")])
    assert "across enough logs, Cheddar consistently lined up with your higher-symptom days" in strong_trig

    prot_strong, _ = _render_text([_row("protective", "strong", food="Rice", demoted=False, reason=None)])
    assert (
        "across enough logs, Rice did not line up with your higher-symptom days. "
        "That's a reassuring sign, not a guarantee." in prot_strong
    )

    incon, _ = _render_text([_row("inconclusive", "moderate", food="Toast")])
    assert (
        "Toast showed up in your logs, but it didn't clearly line up with or against "
        "your symptoms in this window." in incon
    )


# ── 2. sample-size line present on EVERY row (Pillar 5, mandatory §5.3) ───────────

@pytest.mark.parametrize("direction,confidence", list(product(_DIRECTIONS, _CONFIDENCES)))
def test_sample_size_line_present_on_every_row(direction, confidence):
    row = _row(direction, confidence, exposed=7, control=11, episodes=5)
    full, paragraphs = _render_text([row])
    sample = next((p for p in paragraphs if p.startswith("Based on")), None)
    assert sample is not None, f"missing mandatory sample-size line: {paragraphs!r}"
    assert "7 days" in sample and "11 days" in sample and "5 symptom episodes" in sample


def test_sample_size_singular_plural_and_zero_control():
    # the contrast-collapse case (0 control days) still surfaces the honest denominator
    row = _row("trigger", "moderate", exposed=1, control=0, episodes=1)
    _, paragraphs = _render_text([row])
    sample = next(p for p in paragraphs if p.startswith("Based on"))
    assert "1 day" in sample and "0 days" in sample and "1 symptom episode" in sample
    assert "1 days" not in sample  # singular respected


# ── 3. grouping + demotion honesty (§5.1 / §5.4) ─────────────────────────────────

def test_groups_render_under_exact_subheadings_in_fixed_order():
    rows = [
        _row("inconclusive", "moderate", food="Toast"),
        _row("protective", "strong", food="Rice"),
        _row("trigger", "strong", food="Cheddar"),
    ]
    full, paragraphs = _render_text(rows)
    i_trig = paragraphs.index(_GROUP_HEADING["trigger"])
    i_prot = paragraphs.index(_GROUP_HEADING["protective"])
    i_incon = paragraphs.index(_GROUP_HEADING["inconclusive"])
    assert i_trig < i_prot < i_incon, paragraphs


def test_demoted_rows_sink_to_bottom_of_their_group():
    # an undemoted trigger and a demoted trigger share the trigger group; the
    # confirmed headline renders first, the demoted row (never 'Worth discussing') last
    rows = [
        _row("trigger", "insufficient", food="Kimchi"),  # demoted
        _row("trigger", "strong", food="Cheddar"),        # confirmed headline
    ]
    _, paragraphs = _render_text(rows)
    confirmed = next(i for i, p in enumerate(paragraphs) if p.lstrip().startswith("Worth discussing"))
    demoted = next(i for i, p in enumerate(paragraphs) if p.lstrip().startswith("Not enough yet"))
    assert confirmed < demoted, paragraphs


def test_every_demoted_row_renders_a_caveat():
    demoted_combos = [
        (d, c) for d, c in product(_DIRECTIONS, _CONFIDENCES) if _demotion(d, c)[0]
    ]
    for direction, confidence in demoted_combos:
        full, _ = _render_text([_row(direction, confidence)])
        assert any(cav in full for cav in _ALL_CAVEATS), (direction, confidence, full)


def test_confirmed_trigger_headline_has_no_caveat():
    for confidence in ("strong", "moderate"):
        full, _ = _render_text([_row("trigger", confidence)])
        assert not any(cav in full for cav in _ALL_CAVEATS), (confidence, full)


def test_protective_strong_never_a_confirmed_trigger():
    # the Garlic-mis-attribution guard: a protective food, even 'strong', is demoted
    row = _row("protective", "strong")
    full, paragraphs = _render_text([row])
    # renders under the protective subheading, carries a caveat, never 'Worth discussing'
    assert _GROUP_HEADING["protective"] in paragraphs
    assert any(cav in full for cav in _ALL_CAVEATS)
    assert "Worth discussing" not in full


# ── 3b. OQ-4: exact-code caveat mapping, every code reachable, no silent fallthrough ──

#: The intended §5.4 caveat for each canonical code (the contract the renderer honors).
_EXPECTED_CAVEAT = {
    DemotionReason.INSUFFICIENT_SAMPLE: _CAVEAT_SMALL_SAMPLE,
    DemotionReason.BELOW_THRESHOLD: _CAVEAT_SMALL_SAMPLE,
    DemotionReason.GUARDRAIL_DISAGREES: _CAVEAT_DISAGREE,
    DemotionReason.INCONCLUSIVE: _CAVEAT_MIXED,
    DemotionReason.PROTECTIVE: _CAVEAT_GENERIC,
}


def test_expected_caveat_table_covers_every_canonical_code():
    # the test's own expectation table must enumerate the whole enum — so adding a
    # DemotionReason without updating this file fails here, not silently downstream.
    assert {r.value for r in _EXPECTED_CAVEAT} == set(DEMOTION_REASON_CODES)


@pytest.mark.parametrize("reason", list(DemotionReason))
def test_every_canonical_code_maps_to_its_specific_caveat(reason):
    # exact-code mapping (OQ-4): each canonical code resolves to its approved §5.4
    # string. This is what makes the previously-unreachable DISAGREE caveat reachable.
    assert _demoted_caveat(reason.value) == _EXPECTED_CAVEAT[reason]


def test_disagree_caveat_is_now_reachable():
    # the exact bug OQ-4 names: no seam reason used to reach _CAVEAT_DISAGREE. The
    # canonical guardrail-disagreement code now resolves to it by exact match.
    assert _demoted_caveat(DemotionReason.GUARDRAIL_DISAGREES.value) == _CAVEAT_DISAGREE


def test_all_four_caveats_are_reachable_from_some_code():
    reached = {_demoted_caveat(code) for code in DEMOTION_REASON_CODES}
    for caveat in (_CAVEAT_SMALL_SAMPLE, _CAVEAT_DISAGREE, _CAVEAT_MIXED, _CAVEAT_GENERIC):
        assert caveat in reached, f"caveat unreachable from any canonical code: {caveat!r}"


def test_none_reason_takes_the_generic_hedge():
    assert _demoted_caveat(None) == _CAVEAT_GENERIC


def test_unmapped_code_raises_instead_of_silent_generic():
    # mutation guard: a code the seam can't emit (typo / stale rename) must NOT quietly
    # degrade to the generic caveat — it must blow up so the drift is caught.
    with pytest.raises(ValueError):
        _demoted_caveat("association test disagrees")  # the OLD prose, now a non-code
    with pytest.raises(ValueError):
        _demoted_caveat("not_a_real_demotion_code")


def test_renderer_caveat_map_matches_seam_code_set_exactly():
    # the renderer must cover EXACTLY the seam's canonical code set — no missing code
    # (would raise at runtime) and no extra key (a caveat for a code that can't occur).
    assert set(_CAVEAT_BY_REASON) == set(DEMOTION_REASON_CODES)


def test_seam_emits_only_canonical_codes_across_the_whole_matrix():
    # every (direction, confidence) the seam can produce yields either None or a code
    # the renderer knows — proving emit-side and map-side share one vocabulary.
    for direction, confidence in product(_DIRECTIONS, _CONFIDENCES):
        _, reason = _demotion(direction, confidence)
        assert reason is None or reason in DEMOTION_REASON_CODES, (direction, confidence, reason)


def test_demotion_reason_codes_are_bare_codes_not_prose():
    # the stored reason is a machine code, not patient-facing prose (that lives only in
    # the renderer's caveat strings). Guards against a regression to sentence reasons.
    for code in DEMOTION_REASON_CODES:
        assert " " not in code and code == code.lower()


def test_empty_signal_list_renders_warm_no_dead_end():
    full, paragraphs = _render_text([])
    assert paragraphs, "empty state must render explanatory copy, not a blank"
    assert "No food signals yet" in full
    assert "Keep logging what you can" in full
    assert "No data" not in full  # patient-dignity: never a dead-end wall


# ── 4. §5.5 clinician-only supporting detail (default OFF patient-side) ───────────

def test_supporting_detail_hidden_by_default():
    full, _ = _render_text([_row("trigger", "strong")])  # clinician_detail defaults OFF
    assert "Supporting detail" not in full
    assert "odds ratio" not in full


def test_supporting_detail_shown_only_when_toggle_on():
    full, _ = _render_text([_row("trigger", "strong")], clinician_detail=True)
    assert "Supporting detail (for clinicians)" in full
    assert "odds ratio 2.50" in full and "95% CI 1.40" in full and "fisher test" in full


def test_supporting_detail_omitted_for_skipped_test_even_with_toggle():
    full, _ = _render_text([_row("inconclusive", "insufficient", testable=False)],
                           clinician_detail=True)
    assert "Supporting detail" not in full


# ── 5. strengthened contract guard (AST, whole report/export layer) — UNCHANGED ──

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
        # attributes read off the loop variable `r` in _signal_section / helpers
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "r":
                read_attrs.add(node.attr)
    assert read_attrs, "expected the render layer to read row fields"
    stray = read_attrs - row_fields
    assert not stray, f"render layer reads non-contract fields off the row: {stray}"
