"""Local Ollama interpretation for analysis results."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Iterable

import pandas as pd
import requests

from analysis.evidence import build_report_pack

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional until Phase 8 pins dependencies.
    load_dotenv = None


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1"
REQUEST_TIMEOUT_SECONDS = (5, 300)
KNOWN_OLLAMA_CLOUD_MODELS = [
    "nemotron-3-super:cloud",
]
OLLAMA_STOP_MARKERS = (
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|assistant|>",
    "<|user|>",
    "<think>",
    "</think>",
    "[THINK]",
    "[/THINK]",
    "<file",
    "</file>",
    "Thinking Process:",
    "Analyze the Request:",
    "Self-Correction",
    "Drafting the Response:",
    "Internal reasoning:",
    "Chain of thought:",
    "YouThinking Process:",
)
REPORT_TEXT_REPLACEMENTS = {
    "\u00a0": " ",
    "\u00b0": " deg ",
    "\u00b2": "2",
    "\u00b3": "3",
    "\u00b5": "u",
    "\u2013": "-",
    "\u2014": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2191": "up",
    "\u2193": "down",
    "\u2192": "->",
    "\u2212": "-",
    "\u2264": "<=",
    "\u2265": ">=",
    "\u0394": "Delta ",
    "\u03b4": "delta ",
}


class OllamaInterpreterError(RuntimeError):
    """Raised when local interpretation cannot be generated."""


SYSTEM_PROMPT = """You are a bioprocess data analysis expert.

Write for a process development scientist or manufacturing engineer who may not
have a statistician on staff. Be direct, practical, and careful.

Rules:
- Use only the supplied analysis summary.
- Do not claim causation; say "associated with" or "consistent with" unless the data clearly prove otherwise.
- Explain PCA, PLS, and Random Forest signals in plain language.
- Call out when a finding is mainly linear, non-linear, categorical, or exploratory.
- Include data quality warnings that affect confidence.
- Treat pre-flight audit warnings as important context, especially leakage, confounding, date/sequence drift, outliers, and small sample size.
- Treat OOB scores, cross-validated Q2, and time-ordered test scores as stronger evidence than training R2.
- Do not reveal chain-of-thought, hidden reasoning, scratch notes, or "Thinking Process" text.
- Do not output chat-template tokens such as <|endoftext|>, <|im_start|>, <|im_end|>, or <think>.
- Do not create or mention file paths unless the user explicitly asks for a file.
- Never assume typical industry specs, thresholds, or release limits. Use only supplied specs/windows from the analysis summary.
- If specs/windows are missing, say that no spec/window file was supplied.
- Phrase operational recommendations as investigations unless confirmatory evidence is supplied.
- Return only final markdown with concise sections. Start with "## Executive Summary".
"""


def get_ollama_base_url() -> str:
    """Return the configured Ollama base URL."""
    load_environment_if_available()
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/")


def get_configured_ollama_model() -> str:
    """Return the configured default Ollama model."""
    load_environment_if_available()
    return os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)


def load_environment_if_available() -> None:
    """Load .env when python-dotenv is installed."""
    if load_dotenv is not None:
        load_dotenv()


def get_available_ollama_models(base_url: str | None = None) -> list[str]:
    """List locally installed Ollama models."""
    base_url = (base_url or get_ollama_base_url()).rstrip("/")

    try:
        response = requests.get(f"{base_url}/api/tags", timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException as error:
        raise OllamaInterpreterError(
            f"Could not reach Ollama at {base_url}. Start Ollama, then try again."
        ) from error

    if response.status_code != 200:
        raise OllamaInterpreterError(
            f"Ollama returned HTTP {response.status_code} while listing models: {response.text}"
        )

    payload = response.json()
    return [model["name"] for model in payload.get("models", []) if "name" in model]


def build_model_options(available_models: list[str]) -> list[str]:
    """Merge locally listed models with known cloud model shortcuts."""
    model_options: list[str] = []
    for model_name in [*available_models, *KNOWN_OLLAMA_CLOUD_MODELS]:
        if model_name and model_name not in model_options:
            model_options.append(model_name)

    return model_options


def is_cloud_model(model_name: str | None) -> bool:
    """Return whether a model name points at Ollama Cloud."""
    if not model_name:
        return False

    normalized_model_name = model_name.strip().lower()
    return normalized_model_name.endswith(":cloud") or normalized_model_name.endswith("-cloud")


def choose_default_model(available_models: list[str]) -> str:
    """Choose a useful default from installed local models."""
    configured_model = get_configured_ollama_model()
    if configured_model in available_models:
        return configured_model

    preferred_patterns = [
        "qwen3.5:9b",
        "mistral-small",
        "ministral",
        "qwen3-35b",
        "llama3.1",
        "llama3",
        "gemma",
        "qwen",
    ]
    lowercase_models = {model_name.lower(): model_name for model_name in available_models}

    for preferred_pattern in preferred_patterns:
        for lowercase_model_name, model_name in lowercase_models.items():
            if preferred_pattern in lowercase_model_name:
                return model_name

    return available_models[0] if available_models else configured_model


def stream_interpretation(
    profile_result,
    audit_result,
    analysis_results: dict[str, Any],
    merged_dataframe: pd.DataFrame,
    outcomes: list[str],
    model: str,
    base_url: str | None = None,
    spec_assessment=None,
) -> Iterable[str]:
    """Stream markdown interpretation chunks from Ollama."""
    summary = build_interpretation_summary(
        profile_result=profile_result,
        audit_result=audit_result,
        analysis_results=analysis_results,
        merged_dataframe=merged_dataframe,
        outcomes=outcomes,
        spec_assessment=spec_assessment,
    )
    messages = build_ollama_messages(summary)
    base_url = (base_url or get_ollama_base_url()).rstrip("/")

    request_payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": 0.15,
            "num_ctx": 8192,
            "num_predict": 2200,
            "repeat_penalty": 1.15,
            "repeat_last_n": 256,
            "stop": list(OLLAMA_STOP_MARKERS),
        },
    }

    try:
        with requests.post(
            f"{base_url}/api/chat",
            json=request_payload,
            stream=True,
            timeout=REQUEST_TIMEOUT_SECONDS,
        ) as response:
            if response.status_code != 200:
                raise OllamaInterpreterError(
                    f"Ollama returned HTTP {response.status_code}: {response.text}"
                )

            pending_text = ""
            max_marker_length = max(len(marker) for marker in OLLAMA_STOP_MARKERS)

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "error" in chunk:
                    raise OllamaInterpreterError(f"Ollama error: {chunk['error']}")

                content = chunk.get("message", {}).get("content", "")
                if content:
                    pending_text += content
                    safe_text, pending_text, should_stop = split_stream_at_stop_marker(
                        pending_text=pending_text,
                        max_marker_length=max_marker_length,
                    )
                    if safe_text:
                        yield clean_visible_stream_text(safe_text)
                    if should_stop:
                        break

                if chunk.get("done"):
                    break

            if pending_text:
                final_text, _, _ = split_stream_at_stop_marker(
                    pending_text=pending_text,
                    max_marker_length=0,
                )
                if final_text:
                    yield clean_visible_stream_text(final_text)

    except requests.RequestException as error:
        raise OllamaInterpreterError(
            f"Could not reach Ollama at {base_url}. Start Ollama, then try again."
        ) from error


def generate_interpretation(
    profile_result,
    audit_result,
    analysis_results: dict[str, Any],
    merged_dataframe: pd.DataFrame,
    outcomes: list[str],
    model: str,
    base_url: str | None = None,
    spec_assessment=None,
) -> str:
    """Generate a complete non-streamed interpretation."""
    return sanitize_interpretation_text(
        "".join(
            stream_interpretation(
                profile_result=profile_result,
                audit_result=audit_result,
                analysis_results=analysis_results,
                merged_dataframe=merged_dataframe,
                outcomes=outcomes,
                model=model,
                base_url=base_url,
                spec_assessment=spec_assessment,
            )
        )
    )


def build_ollama_messages(summary: dict[str, Any]) -> list[dict[str, str]]:
    """Build chat messages for Ollama's /api/chat endpoint."""
    user_prompt = f"""/no_think
Interpret this Batch Insight Analyzer report pack.

Return markdown with these sections:
1. Executive summary
2. Top drivers by outcome
3. Specs and operating-window notes
4. Cross-cutting process patterns
5. Hypotheses to investigate next
6. Data quality and confidence notes

Analysis summary JSON:
{json.dumps(summary, indent=2)}

Important output rules:
- Return the final answer only.
- Use exactly these markdown section headings: ## Executive Summary, ## Top Drivers by Outcome, ## Specs and Operating-Window Notes, ## Cross-Cutting Process Patterns, ## Hypotheses to Investigate Next, ## Data Quality and Confidence Notes.
- Do not include hidden reasoning, scratchpad text, chat tokens, XML/file tags, or file paths.
- Label quartile thresholds and high-high interaction screens as exploratory, not optimized setpoints.
- When historical_response_bands are present, describe broad quartile bands and refined narrow bins separately. The refined bin is more exact but more noise-sensitive. Do not flatten middle-band patterns into "higher is better" or "lower is better".
- If specs/windows are supplied, call them a historical operating-window assessment, not a proven design space.
- Distinguish process windows from QC release specs.
- Never create an "assumed typical specs" section. Use only the supplied spec_challenge_table and out_of_spec_batches.
- Do not recommend widening or narrowing a spec unless the evidence is clearly sufficient; otherwise say what should be investigated.
- Do not invent numeric values. Quote exact means, limits, and counts only when they appear in the supplied JSON.
- When mentioning categorical levels, include the variable name and level together, such as reactor_id=RX-3 or raw_material_lot=RM-005.
- Prefer the deterministic top_drivers, model_quality, specs_and_windows, categorical_level_effects, numeric_driver_patterns, historical_response_bands, and exploratory_interactions fields over broad inference.
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def split_stream_at_stop_marker(
    pending_text: str,
    max_marker_length: int,
) -> tuple[str, str, bool]:
    """Return safe stream text while holding a small tail for split stop markers."""
    stop_index = find_first_stop_marker_index(pending_text)
    if stop_index is not None:
        return pending_text[:stop_index], "", True

    if max_marker_length <= 0 or len(pending_text) <= max_marker_length:
        return pending_text, "", False

    safe_length = len(pending_text) - max_marker_length
    return pending_text[:safe_length], pending_text[safe_length:], False


def find_first_stop_marker_index(text: str) -> int | None:
    """Find the earliest local-model artifact marker in text."""
    lower_text = text.lower()
    marker_indexes = [
        lower_text.find(marker.lower())
        for marker in OLLAMA_STOP_MARKERS
        if lower_text.find(marker.lower()) != -1
    ]
    return min(marker_indexes) if marker_indexes else None


def clean_visible_stream_text(text: str) -> str:
    """Remove small inline artifacts that can appear before a full stop marker."""
    cleaned_text = re.sub(r"<\|[^>]{1,80}\|>", "", text)
    cleaned_text = re.sub(r"</?think>", "", cleaned_text, flags=re.IGNORECASE)
    return normalize_report_text(cleaned_text)


def sanitize_interpretation_text(text: str) -> str:
    """Trim accidental local-model artifacts from a completed interpretation."""
    cleaned_text = text or ""

    stop_index = find_first_stop_marker_index(cleaned_text)
    if stop_index is not None:
        cleaned_text = cleaned_text[:stop_index]

    cleaned_text = clean_visible_stream_text(cleaned_text)
    cleaned_text = re.sub(
        r"</?file(?:\s+[^>]*)?>",
        "",
        cleaned_text,
        flags=re.IGNORECASE,
    )
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return normalize_report_text(cleaned_text).strip()


def normalize_report_text(text: str) -> str:
    """Use plain ASCII punctuation so downloaded Markdown opens cleanly on Windows."""
    normalized_text = text
    for original_text, replacement_text in REPORT_TEXT_REPLACEMENTS.items():
        normalized_text = normalized_text.replace(original_text, replacement_text)

    normalized_text = re.sub(r"\s+deg\s+C", " deg C", normalized_text)
    normalized_text = re.sub(r"[ \t]{2,}", " ", normalized_text)
    return normalized_text


def build_interpretation_summary(
    profile_result,
    audit_result,
    analysis_results: dict[str, Any],
    merged_dataframe: pd.DataFrame,
    outcomes: list[str],
    spec_assessment=None,
) -> dict[str, Any]:
    """Create a compact JSON-ready report pack for the local language model."""
    return build_report_pack(
        profile_result=profile_result,
        audit_result=audit_result,
        analysis_results=analysis_results,
        merged_dataframe=merged_dataframe,
        outcomes=outcomes,
        spec_assessment=spec_assessment,
    )


def build_audit_summary(audit_result) -> dict[str, Any]:
    """Summarize pre-flight audit results for the local language model."""
    if audit_result is None:
        return {}

    return {
        "summary": audit_result.summary,
        "warnings": audit_result.warnings,
        "leakage_name_warnings": dataframe_to_records(
            audit_result.leakage_name_warnings,
            max_rows=10,
        ),
        "feature_target_correlations": dataframe_to_records(
            audit_result.feature_target_correlations,
            max_rows=10,
        ),
        "date_or_drift_columns": dataframe_to_records(
            audit_result.date_or_drift_columns,
            max_rows=10,
        ),
        "confounding_categoricals": dataframe_to_records(
            audit_result.confounding_categoricals,
            max_rows=10,
        ),
        "high_cardinality_categoricals": dataframe_to_records(
            audit_result.high_cardinality_categoricals,
            max_rows=10,
        ),
        "outlier_flags": dataframe_to_records(audit_result.outlier_flags, max_rows=12),
        "multicollinearity_pairs": dataframe_to_records(
            audit_result.multicollinearity_pairs,
            max_rows=12,
        ),
    }


def build_spec_assessment_summary(spec_assessment) -> dict[str, Any]:
    """Summarize spec/window assessment results for the local language model."""
    if spec_assessment is None or not getattr(spec_assessment, "has_specs", False):
        return {"available": False}

    variable_summary = spec_assessment.variable_summary
    if variable_summary.empty:
        return {
            "available": True,
            "matched_spec_count": len(spec_assessment.matched_specs),
            "unmatched_spec_count": len(spec_assessment.unmatched_specs),
            "warnings": spec_assessment.warnings,
            "spec_challenge_table": [],
        }

    challenge_columns = [
        "variable",
        "role",
        "target",
        "lower_limit",
        "upper_limit",
        "unit",
        "criticality",
        "count",
        "percent_inside",
        "percent_outside",
        "below_limit_count",
        "above_limit_count",
        "percent_close_to_limit",
        "mean",
        "std",
        "min",
        "max",
        "used_range_ratio",
        "median_nearest_limit_margin",
        "worst_limit_margin",
        "max_driver_score",
        "confounding_flag",
        "classification",
        "reason",
    ]
    available_challenge_columns = [
        column_name for column_name in challenge_columns if column_name in variable_summary.columns
    ]

    return {
        "available": True,
        "important_instruction": (
            "Treat this as a historical operating-window assessment. Do not call it a design space. "
            "Do not recommend changing a spec without confirmatory evidence."
        ),
        "matched_spec_count": len(spec_assessment.matched_specs),
        "unmatched_spec_count": len(spec_assessment.unmatched_specs),
        "warnings": spec_assessment.warnings,
        "classification_counts": variable_summary["classification"].value_counts().to_dict()
        if "classification" in variable_summary.columns
        else {},
        "spec_challenge_table": dataframe_to_records(
            variable_summary[available_challenge_columns],
            max_rows=20,
        ),
        "out_of_spec_batches": dataframe_to_records(
            spec_assessment.out_of_spec_batches,
            max_rows=20,
        ),
        "outcome_means_by_spec_zone": dataframe_to_records(
            spec_assessment.outcome_zone_summary,
            max_rows=40,
        ),
    }


def build_model_quality_summary(
    analysis_results: dict[str, Any],
    outcomes: list[str],
) -> dict[str, Any]:
    """Summarize model fit signals that affect confidence."""
    summary = {}

    for outcome in outcomes:
        pls_result = analysis_results["pls"][outcome]
        random_forest_result = analysis_results["random_forest"][outcome]
        summary[outcome] = {
            "pls_cv_q2": round_or_none(pls_result["q2"]),
            "pls_training_r2": round_or_none(pls_result["training_r2"]),
            "pls_components": pls_result["n_components"],
            "pls_time_ordered_validation": sanitize_validation_summary(
                pls_result.get("time_ordered_validation", {})
            ),
            "random_forest_training_r2": round_or_none(random_forest_result["training_r2"]),
            "random_forest_oob_r2": round_or_none(random_forest_result.get("oob_r2")),
            "random_forest_time_ordered_validation": sanitize_validation_summary(
                random_forest_result.get("time_ordered_validation", {})
            ),
            "rows_used": random_forest_result["n_rows_used"],
        }

    return summary


def sanitize_validation_summary(validation_result: dict[str, Any]) -> dict[str, Any]:
    """Return JSON-safe validation metadata without internal indices."""
    return {
        key: round_or_none(value) if isinstance(value, float) else value
        for key, value in validation_result.items()
        if key not in {"train_index", "test_index"}
    }


def build_method_specific_summary(
    analysis_results: dict[str, Any],
    outcomes: list[str],
) -> dict[str, Any]:
    """Collect top method-specific variables and features."""
    summary = {}

    for outcome in outcomes:
        pls_result = analysis_results["pls"][outcome]
        random_forest_result = analysis_results["random_forest"][outcome]
        summary[outcome] = {
            "pls_top_variables": dataframe_to_records(
                pls_result["variable_importance"].head(8)
            ),
            "pls_top_features": dataframe_to_records(
                pls_result["feature_coefficients"].head(12)
            ),
            "random_forest_top_variables": dataframe_to_records(
                random_forest_result["variable_importance"].head(8)
            ),
            "random_forest_top_features": dataframe_to_records(
                random_forest_result["feature_importances"].head(12)
            ),
        }

    return summary


def summarize_categorical_level_effects(
    merged_dataframe: pd.DataFrame,
    profile_result,
    outcomes: list[str],
    ranked_drivers: pd.DataFrame,
    max_variables_per_outcome: int = 8,
) -> dict[str, Any]:
    """Summarize which categories sit above or below the outcome average."""
    variable_types = profile_result.variable_types
    categorical_variables = set(
        variable_types[
            variable_types["type"].isin(["categorical", "binary"])
        ]["column"].tolist()
    )
    summary: dict[str, Any] = {}

    for outcome in outcomes:
        top_variables = ranked_drivers[ranked_drivers["outcome"] == outcome][
            "process_variable"
        ].head(max_variables_per_outcome)
        candidate_variables = [
            variable
            for variable in top_variables
            if variable in categorical_variables and variable in merged_dataframe.columns
        ]
        outcome_summary = {}
        outcome_values = pd.to_numeric(merged_dataframe[outcome], errors="coerce")
        overall_mean = outcome_values.mean()

        for variable in candidate_variables:
            grouped = (
                merged_dataframe.assign(_outcome=outcome_values)
                .dropna(subset=[variable, "_outcome"])
                .groupby(variable)["_outcome"]
                .agg(["count", "mean"])
                .reset_index()
            )
            if grouped.empty:
                continue

            grouped["delta_from_overall_mean"] = grouped["mean"] - overall_mean
            grouped["mean"] = grouped["mean"].round(3)
            grouped["delta_from_overall_mean"] = grouped["delta_from_overall_mean"].round(3)
            grouped = grouped.sort_values(
                "delta_from_overall_mean",
                key=lambda values: values.abs(),
                ascending=False,
            )
            outcome_summary[variable] = dataframe_to_records(grouped, max_rows=8)

        summary[outcome] = outcome_summary

    return summary


def summarize_numeric_driver_patterns(
    merged_dataframe: pd.DataFrame,
    profile_result,
    outcomes: list[str],
    ranked_drivers: pd.DataFrame,
    max_variables_per_outcome: int = 5,
) -> dict[str, Any]:
    """Summarize simple numeric patterns for top drivers."""
    numeric_variables = set(
        profile_result.variable_types[
            profile_result.variable_types["type"] == "continuous"
        ]["column"].tolist()
    )
    summary: dict[str, Any] = {}

    for outcome in outcomes:
        outcome_values = pd.to_numeric(merged_dataframe[outcome], errors="coerce")
        top_numeric_variables = [
            variable
            for variable in ranked_drivers[ranked_drivers["outcome"] == outcome][
                "process_variable"
            ].head(max_variables_per_outcome)
            if variable in numeric_variables and variable in merged_dataframe.columns
        ]
        outcome_patterns = []

        for variable in top_numeric_variables:
            variable_values = pd.to_numeric(merged_dataframe[variable], errors="coerce")
            usable_data = pd.DataFrame(
                {"variable": variable_values, "outcome": outcome_values}
            ).dropna()
            if usable_data["variable"].nunique() < 4:
                continue

            usable_data["quartile"] = pd.qcut(
                usable_data["variable"],
                q=4,
                duplicates="drop",
            )
            quartile_summary = (
                usable_data.groupby("quartile", observed=True)["outcome"]
                .agg(["count", "mean"])
                .reset_index()
            )
            quartile_summary["quartile"] = quartile_summary["quartile"].astype(str)
            quartile_summary["mean"] = quartile_summary["mean"].round(3)

            outcome_patterns.append(
                {
                    "process_variable": variable,
                    "pearson_correlation": round_or_none(
                        usable_data["variable"].corr(usable_data["outcome"])
                    ),
                    "outcome_mean_by_variable_quartile": dataframe_to_records(
                        quartile_summary
                    ),
                }
            )

        summary[outcome] = outcome_patterns

    return summary


def summarize_numeric_interactions(
    merged_dataframe: pd.DataFrame,
    profile_result,
    outcomes: list[str],
    ranked_drivers: pd.DataFrame,
    max_variables_per_outcome: int = 6,
    max_pairs_per_outcome: int = 5,
) -> dict[str, Any]:
    """Screen top numeric-driver pairs for high-high outcome shifts."""
    numeric_variables = set(
        profile_result.variable_types[
            profile_result.variable_types["type"] == "continuous"
        ]["column"].tolist()
    )
    summary: dict[str, Any] = {}

    for outcome in outcomes:
        outcome_values = pd.to_numeric(merged_dataframe[outcome], errors="coerce")
        top_numeric_variables = [
            variable
            for variable in ranked_drivers[ranked_drivers["outcome"] == outcome][
                "process_variable"
            ].head(max_variables_per_outcome)
            if variable in numeric_variables and variable in merged_dataframe.columns
        ]
        pair_rows = []

        for first_index, first_variable in enumerate(top_numeric_variables):
            for second_variable in top_numeric_variables[first_index + 1 :]:
                first_values = pd.to_numeric(merged_dataframe[first_variable], errors="coerce")
                second_values = pd.to_numeric(merged_dataframe[second_variable], errors="coerce")
                usable_data = pd.DataFrame(
                    {
                        "first": first_values,
                        "second": second_values,
                        "outcome": outcome_values,
                    }
                ).dropna()
                if len(usable_data) < 10:
                    continue

                first_threshold = usable_data["first"].quantile(0.75)
                second_threshold = usable_data["second"].quantile(0.75)
                high_high_mask = (
                    (usable_data["first"] >= first_threshold)
                    & (usable_data["second"] >= second_threshold)
                )
                if int(high_high_mask.sum()) < 3:
                    continue

                high_high_mean = usable_data.loc[high_high_mask, "outcome"].mean()
                other_mean = usable_data.loc[~high_high_mask, "outcome"].mean()
                pair_rows.append(
                    {
                        "first_variable": first_variable,
                        "second_variable": second_variable,
                        "first_high_threshold": round_or_none(first_threshold),
                        "second_high_threshold": round_or_none(second_threshold),
                        "high_high_count": int(high_high_mask.sum()),
                        "high_high_outcome_mean": round_or_none(high_high_mean),
                        "other_outcome_mean": round_or_none(other_mean),
                        "delta_high_high_vs_other": round_or_none(high_high_mean - other_mean),
                    }
                )

        pair_rows = sorted(
            pair_rows,
            key=lambda row: abs(row["delta_high_high_vs_other"] or 0.0),
            reverse=True,
        )
        summary[outcome] = pair_rows[:max_pairs_per_outcome]

    return summary


def dataframe_to_records(dataframe: pd.DataFrame, max_rows: int | None = None) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe records."""
    if dataframe.empty:
        return []

    output_dataframe = dataframe.head(max_rows).copy() if max_rows else dataframe.copy()
    return json.loads(output_dataframe.to_json(orient="records"))


def round_or_none(value: Any, digits: int = 3) -> float | None:
    """Round values for JSON summaries while preserving missing values."""
    try:
        if pd.isna(value):
            return None
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None
