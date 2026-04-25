"""Local Ollama interpretation for analysis results."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Iterable

import pandas as pd
import requests

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
) -> Iterable[str]:
    """Stream markdown interpretation chunks from Ollama."""
    summary = build_interpretation_summary(
        profile_result=profile_result,
        audit_result=audit_result,
        analysis_results=analysis_results,
        merged_dataframe=merged_dataframe,
        outcomes=outcomes,
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
            "num_predict": 1400,
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
            )
        )
    )


def build_ollama_messages(summary: dict[str, Any]) -> list[dict[str, str]]:
    """Build chat messages for Ollama's /api/chat endpoint."""
    user_prompt = f"""/no_think
Interpret this Batch Insight Analyzer result.

Return markdown with these sections:
1. Executive summary
2. Top drivers by outcome
3. Cross-cutting process patterns
4. Hypotheses to investigate next
5. Data quality and confidence notes

Analysis summary JSON:
{json.dumps(summary, indent=2)}

Important output rules:
- Return the final answer only.
- Do not include hidden reasoning, scratchpad text, chat tokens, XML/file tags, or file paths.
- Label quartile thresholds and high-high interaction screens as exploratory, not optimized setpoints.
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
    return cleaned_text


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
    return cleaned_text.strip()


def build_interpretation_summary(
    profile_result,
    audit_result,
    analysis_results: dict[str, Any],
    merged_dataframe: pd.DataFrame,
    outcomes: list[str],
) -> dict[str, Any]:
    """Create a compact JSON-ready summary for the local language model."""
    ranked_drivers = analysis_results["ranked_drivers"]

    return {
        "data_profile": {
            "matched_batch_count": profile_result.matched_batch_count,
            "process_variable_count": len(profile_result.process_columns),
            "outcomes": outcomes,
            "variable_type_counts": profile_result.variable_type_counts,
            "warnings": profile_result.warnings,
            "high_missing_columns": dataframe_to_records(
                profile_result.missingness[profile_result.missingness["flag"]],
                max_rows=12,
            ),
            "near_constant_columns": dataframe_to_records(
                profile_result.near_constant_columns,
                max_rows=12,
            ),
            "outcome_statistics": dataframe_to_records(profile_result.outcome_statistics),
        },
        "preflight_audit": build_audit_summary(audit_result),
        "top_ranked_drivers_by_outcome": {
            outcome: dataframe_to_records(
                ranked_drivers[ranked_drivers["outcome"] == outcome].head(8)
            )
            for outcome in outcomes
        },
        "model_quality_by_outcome": build_model_quality_summary(analysis_results, outcomes),
        "pca_summary": {
            "explained_variance": dataframe_to_records(
                analysis_results["pca"]["explained_variance"],
                max_rows=8,
            ),
            "top_process_loadings": dataframe_to_records(
                analysis_results["pca"]["process_loadings"].head(10)
            ),
        },
        "method_specific_top_variables": build_method_specific_summary(
            analysis_results,
            outcomes,
        ),
        "categorical_level_effects": summarize_categorical_level_effects(
            merged_dataframe=merged_dataframe,
            profile_result=profile_result,
            outcomes=outcomes,
            ranked_drivers=ranked_drivers,
        ),
        "numeric_driver_patterns": summarize_numeric_driver_patterns(
            merged_dataframe=merged_dataframe,
            profile_result=profile_result,
            outcomes=outcomes,
            ranked_drivers=ranked_drivers,
        ),
        "numeric_interaction_screen": summarize_numeric_interactions(
            merged_dataframe=merged_dataframe,
            profile_result=profile_result,
            outcomes=outcomes,
            ranked_drivers=ranked_drivers,
        ),
    }


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
