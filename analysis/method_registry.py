"""Plain-language method explanations used by the UI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodExplainer:
    """Pedagogical description of one app method or result type."""

    name: str
    short_label: str
    what_it_is: str
    what_it_can_tell_you: str
    what_it_cannot_prove: str
    confidence_guidance: str
    next_step: str
    status: str = "available"


METHOD_EXPLAINERS = {
    "ranked_drivers": MethodExplainer(
        name="Ranked Drivers",
        short_label="Practical driver ranking",
        what_it_is=(
            "A combined ranking of process variables using PCA, PLS, and Random Forest signals."
        ),
        what_it_can_tell_you=(
            "Which variables are most associated with the selected QC outcome in the historical data."
        ),
        what_it_cannot_prove=(
            "It does not prove causation and it is not a formal statistical validation."
        ),
        confidence_guidance=(
            "Confidence is strongest when several methods agree and validation scores are acceptable."
        ),
        next_step="Review the driver breakdown, then check plots and domain plausibility.",
    ),
    "pca": MethodExplainer(
        name="PCA",
        short_label="Batch structure map",
        what_it_is=(
            "An unsupervised chemometrics method that summarizes major variation across process variables."
        ),
        what_it_can_tell_you=(
            "Which batches cluster, drift, or look unusual, and which variables drive overall process variation."
        ),
        what_it_cannot_prove=(
            "PCA does not know the selected QC outcome and does not prove a variable affects quality."
        ),
        confidence_guidance=(
            "Use PCA as a sanity check and exploration tool, especially before trusting supervised models."
        ),
        next_step="Color PCA plots by outcome, reactor, supplier, operator, or time/campaign.",
    ),
    "pls": MethodExplainer(
        name="PLS Regression",
        short_label="Linear QC prediction",
        what_it_is=(
            "A supervised latent-variable regression method commonly used in chemometrics."
        ),
        what_it_can_tell_you=(
            "Which process variables have a linear or near-linear relationship with a QC outcome."
        ),
        what_it_cannot_prove=(
            "PLS can miss non-linear sweet spots and cannot separate correlation from causation."
        ),
        confidence_guidance=(
            "Cross-validated Q2 and time-ordered test R2 matter more than training R2."
        ),
        next_step="Compare PLS drivers with Random Forest drivers; agreement is useful.",
    ),
    "random_forest": MethodExplainer(
        name="Random Forest",
        short_label="Non-linear baseline",
        what_it_is=(
            "A tree ensemble that can capture non-linear patterns and interactions in tabular data."
        ),
        what_it_can_tell_you=(
            "Which variables help a non-linear model predict the selected outcome."
        ),
        what_it_cannot_prove=(
            "It can overfit small datasets and its importances are model behavior, not causal effects."
        ),
        confidence_guidance=(
            "OOB R2, permutation importance, and time-ordered test R2 should guide trust."
        ),
        next_step="Use response plots or future SHAP views to understand direction and shape.",
    ),
    "interaction_screen": MethodExplainer(
        name="Interaction Screen",
        short_label="Exploratory high-high check",
        what_it_is=(
            "A simple comparison of outcome means when two numeric variables are both high versus the rest."
        ),
        what_it_can_tell_you=(
            "Whether a pair of variables may deserve a closer interaction review."
        ),
        what_it_cannot_prove=(
            "It is not a full interaction model and can be unstable when few batches sit in the high-high region."
        ),
        confidence_guidance="Treat it as a hypothesis generator only.",
        next_step="Confirm with targeted plots, model-based dependence checks, or designed experiments.",
    ),
    "response_bands": MethodExplainer(
        name="Suggested Response Bands",
        short_label="Historical sweet-spot hints",
        what_it_is=(
            "Quartile-based and narrower-bin summaries of where an outcome looked best historically."
        ),
        what_it_can_tell_you=(
            "Whether a variable appears to have a middle-band sweet spot rather than a simple higher/lower trend."
        ),
        what_it_cannot_prove=(
            "It is not an optimized setpoint, design space, or process recommendation by itself."
        ),
        confidence_guidance=(
            "The broad band is more stable; the refined bin is more exact but more noise-sensitive."
        ),
        next_step="Use as an investigation target or confirmation-run candidate.",
    ),
    "specs": MethodExplainer(
        name="Specs And Operating Windows",
        short_label="Historical challenge check",
        what_it_is=(
            "A comparison of supplied process windows/QC specs against historical values and model drivers."
        ),
        what_it_can_tell_you=(
            "Whether a window may be too wide, too narrow, off-center, or insufficiently explored."
        ),
        what_it_cannot_prove=(
            "It is not formal validation and does not justify spec changes by itself."
        ),
        confidence_guidance=(
            "Trust improves when the dataset covers the allowed range and the variable is a stable quality driver."
        ),
        next_step="Use the spec review checklist before changing any controlled range.",
    ),
    "llm_interpretation": MethodExplainer(
        name="LLM Interpretation",
        short_label="Plain-language narrative",
        what_it_is=(
            "A local or cloud Ollama model summarizing deterministic analysis facts."
        ),
        what_it_can_tell_you=(
            "A readable story for process scientists, engineers, or stakeholders."
        ),
        what_it_cannot_prove=(
            "It can still omit, distort, or overstate findings depending on the selected model."
        ),
        confidence_guidance=(
            "Use report validation warnings and the deterministic tables as the source of truth."
        ),
        next_step="Review warnings before sharing externally.",
    ),
    "catboost_shap": MethodExplainer(
        name="CatBoost + SHAP",
        short_label="Optional non-linear explainability",
        what_it_is=(
            "An optional CatBoost model that uses native SHAP values for global and per-batch feature attribution."
        ),
        what_it_can_tell_you=(
            "Which variables drive a tree model overall, and what contributed to an individual batch prediction."
        ),
        what_it_cannot_prove=(
            "SHAP explains model behavior; it is still not causation."
        ),
        confidence_guidance=(
            "Use CatBoost as a comparison layer; it does not currently change the main ranked-driver score."
        ),
        next_step="Install CatBoost and run it for selected outcomes when non-linear/categorical patterns matter.",
        status="optional",
    ),
    "opls": MethodExplainer(
        name="OPLS",
        short_label="Future chemometrics root-cause view",
        what_it_is=(
            "A planned PLS variant that separates outcome-related variation from structured unrelated variation."
        ),
        what_it_can_tell_you=(
            "Cleaner root-cause interpretation when background variation obscures a QC relationship."
        ),
        what_it_cannot_prove=(
            "It still needs validation and does not remove confounding automatically."
        ),
        confidence_guidance="Planned future method; not used in current rankings yet.",
        next_step="Implement OPLS1 with cross-validation and permutation testing later.",
        status="planned",
    ),
}


def get_method_explainer(method_key: str) -> MethodExplainer:
    """Return one method explanation."""
    return METHOD_EXPLAINERS[method_key]


def list_method_explainers(method_keys: list[str] | None = None) -> list[MethodExplainer]:
    """Return method explanations in a stable order."""
    if method_keys is None:
        method_keys = list(METHOD_EXPLAINERS.keys())
    return [METHOD_EXPLAINERS[key] for key in method_keys if key in METHOD_EXPLAINERS]
