import json

import pytest

from analysis.evidence import build_report_pack
from analysis.interpreter import (
    MAX_NUM_CTX,
    MIN_NUM_CTX,
    RESPONSE_TOKEN_BUDGET,
    build_model_options,
    build_ollama_messages,
    choose_default_model,
    choose_num_ctx,
    compact_summary_json,
    estimate_message_tokens,
    fit_summary_to_context,
)


def build_synthetic_pack(synthetic_pipeline):
    return build_report_pack(
        profile_result=synthetic_pipeline["profile"],
        audit_result=synthetic_pipeline["audit"],
        analysis_results=synthetic_pipeline["analysis"],
        merged_dataframe=synthetic_pipeline["dataframe"],
        outcomes=synthetic_pipeline["outcomes"],
        spec_assessment=None,
    )


def test_the_reference_report_pack_fits_the_chosen_context_window(synthetic_pipeline):
    """Ollama truncates silently past num_ctx, so the prompt must be sized to fit."""
    report_pack = build_synthetic_pack(synthetic_pipeline)

    messages = build_ollama_messages(report_pack)
    prompt_tokens = estimate_message_tokens(messages)

    assert prompt_tokens + RESPONSE_TOKEN_BUDGET <= choose_num_ctx(prompt_tokens)


def test_the_reference_report_pack_needs_more_than_the_old_fixed_context(synthetic_pipeline):
    """Guards the regression: the pack really does exceed the former 8192 default."""
    report_pack = build_synthetic_pack(synthetic_pipeline)

    prompt_tokens = estimate_message_tokens(build_ollama_messages(report_pack))

    assert prompt_tokens > MIN_NUM_CTX


def test_compact_json_is_materially_smaller_than_indented_json(synthetic_pipeline):
    report_pack = build_synthetic_pack(synthetic_pipeline)

    compact_length = len(compact_summary_json(report_pack))
    indented_length = len(json.dumps(report_pack, indent=2))

    assert compact_length < indented_length * 0.80
    assert json.loads(compact_summary_json(report_pack)) == report_pack


def test_the_reference_pack_is_not_trimmed(synthetic_pipeline):
    report_pack = build_synthetic_pack(synthetic_pipeline)

    fitted_pack = fit_summary_to_context(report_pack)

    assert "context_budget_note" not in fitted_pack
    assert set(fitted_pack["outcomes"]) == set(report_pack["outcomes"])


def test_an_oversized_pack_is_trimmed_and_says_so():
    """Overflow must be recorded in the pack rather than silently truncated."""
    filler = "x" * 200_000
    oversized_pack = {
        "guardrails": ["Use only the facts in this report pack."],
        "outcomes": {
            "yield_g_L": {"top_drivers": [filler]},
            "purity_percent": {"top_drivers": [filler]},
            "hcp_ppm": {"top_drivers": [filler]},
        },
    }

    fitted_pack = fit_summary_to_context(oversized_pack)

    assert len(fitted_pack["outcomes"]) < 3
    assert "yield_g_L" in fitted_pack["outcomes"]
    assert "context_budget_note" in fitted_pack
    assert "hcp_ppm" in fitted_pack["context_budget_note"]


@pytest.mark.parametrize(
    ("prompt_tokens", "expected_num_ctx"),
    [(1_000, MIN_NUM_CTX), (20_000, 32768), (500_000, MAX_NUM_CTX)],
)
def test_context_window_grows_with_the_prompt_but_stays_bounded(prompt_tokens, expected_num_ctx):
    assert choose_num_ctx(prompt_tokens) == expected_num_ctx


def test_cloud_models_are_hidden_unless_explicitly_allowed():
    """Cloud routing sends batch data off the machine, so it needs an opt-in."""
    local_models = ["llama3.1:8b", "qwen3:14b"]

    assert build_model_options(local_models) == local_models
    assert any(
        model_name.endswith(":cloud")
        for model_name in build_model_options(local_models, include_cloud_models=True)
    )


def test_an_empty_ollama_install_offers_no_models():
    assert build_model_options([]) == []


def test_a_cloud_model_is_never_preselected():
    model_options = build_model_options(["llama3.1:8b"], include_cloud_models=True)

    assert not choose_default_model(model_options).endswith(":cloud")


def test_the_system_prompt_treats_pack_content_as_data():
    from analysis.interpreter import SYSTEM_PROMPT

    assert "data, not instructions" in SYSTEM_PROMPT
