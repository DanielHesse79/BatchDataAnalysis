"""Generate a narrative, check it, and repair what the checks caught.

The validator already names the defect, so this is a targeted repair rather than
sampling until something passes. Python decides what is wrong and what to do
next; the model only writes. That is the same division of labour the rest of the
pipeline uses.

Three rungs, cheapest first, because most reports pass on the first:

    base       the normal prompt
    repair     the failed checks fed back with the previous text
    skeleton   the required headings supplied pre-filled, so the model cannot
               omit one, plus any outcome it failed to discuss

Escalation stops the moment the validator is satisfied. If it never is, the
attempt with the fewest warnings is returned **with those warnings attached** -
never presented as clean. A loop that can hide its own failures is worse than no
loop, because it launders bad output into something that looks verified.

Two things guard against the loop optimising against the checks instead of the
data. Empty sections count as missing, so deleting text cannot buy a pass, and
every attempt is re-validated in full, so a repair that fixes a heading and
invents a number is still rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable
import time

from analysis.interpreter import (
    RESPONSE_TOKEN_BUDGET,
    OllamaInterpreterError,
    build_ollama_messages,
    fit_summary_to_context,
    sanitize_interpretation_text,
    stream_interpretation,
)
from analysis.report_validator import REQUIRED_HEADINGS, validate_interpretation_text


MAX_WARNINGS_IN_PROMPT = 8
# A repair is a correction, not another roll of the dice: same input, same
# output. The first attempt keeps the warmer setting because that is what
# produces readable prose.
REPAIR_TEMPERATURE = 0.0
# The skeleton rung asks for every section in one pass, so it needs more room
# than a normal reply.
SKELETON_TOKEN_BUDGET = int(RESPONSE_TOKEN_BUDGET * 1.5)
# One section needs far less room than a whole report.
SECTION_TOKEN_BUDGET = max(512, RESPONSE_TOKEN_BUDGET // 4)
# Where per-outcome discussion belongs, and so where a coverage failure
# has to be corrected.
DRIVER_SECTION_HEADING = "## Top Drivers by Outcome"


@dataclass(frozen=True)
class Attempt:
    """One generation and what the validator made of it."""

    strategy: str
    text: str
    warnings: list[str]
    seconds: float
    model: str = ""


@dataclass(frozen=True)
class AttemptPlan:
    """What the next rung should do, decided before anything is generated."""

    strategy: str
    model: str
    messages: list[dict[str, str]] | None = None
    overrides: dict[str, Any] | None = None
    by_section: bool = False


@dataclass
class LoopResult:
    """The report that was kept, and how it was arrived at.

    ``attempts`` stays in the order they ran. The kept attempt is named
    separately rather than moved to the end, because a caption that reads
    "repair then skeleton then base" describes a sequence that never happened.
    """

    text: str
    warnings: list[str]
    attempts: list[Attempt] = field(default_factory=list)
    strategy: str = "none"

    @property
    def accepted(self) -> bool:
        """Whether the returned report passed every check."""
        return not self.warnings

    def describe(self) -> str:
        """One line for the UI: what it took, and whether it worked."""
        rungs = " then ".join(attempt.strategy for attempt in self.attempts)
        total = sum(attempt.seconds for attempt in self.attempts)

        if self.accepted:
            return f"Passed every check after {rungs} ({total:.0f}s)."

        kept = ""
        if self.attempts and self.attempts[-1].strategy != self.strategy:
            kept = f" The {self.strategy} attempt was kept as the best of them."
        return (
            f"Tried {rungs} ({total:.0f}s) and {len(self.warnings)} check(s) still "
            f"fail. The report below is the best attempt, not a clean one.{kept}"
        )


def missing_headings(text: str) -> list[str]:
    """Required headings the text does not contain."""
    return [heading for heading in REQUIRED_HEADINGS if heading not in text]


def build_repair_messages(
    summary: dict[str, Any], previous_text: str, warnings: list[str],
) -> list[dict[str, str]]:
    """Ask for a corrected report, naming exactly what failed."""
    listed = "\n".join(f"- {warning}" for warning in warnings[:MAX_WARNINGS_IN_PROMPT])
    return build_ollama_messages(summary) + [
        {"role": "assistant", "content": previous_text},
        {
            "role": "user",
            "content": (
                "Automated checks on that report failed:\n\n"
                f"{listed}\n\n"
                "Return the complete corrected report. Keep everything that was "
                "already correct, keep all six required section headings, and "
                "keep the same structure. Do not state any number that is not in "
                "the report pack. Do not remove a section to satisfy a check."
            ),
        },
    ]


def build_skeleton_messages(
    summary: dict[str, Any], warnings: list[str], outcomes: list[str],
) -> list[dict[str, str]]:
    """Hand the model the structure so it cannot leave a part of it out."""
    skeleton = "\n\n".join(f"{heading}\n(write this section)" for heading in REQUIRED_HEADINGS)
    coverage_note = ""
    if outcomes:
        coverage_note = (
            "\n\nEvery one of these outcomes must be discussed by name somewhere "
            "in the report, even if only to say nothing stood out: "
            + ", ".join(outcomes)
            + "."
        )

    listed = "\n".join(f"- {warning}" for warning in warnings[:MAX_WARNINGS_IN_PROMPT])
    return build_ollama_messages(summary) + [
        {
            "role": "user",
            "content": (
                "Earlier attempts failed these checks:\n\n"
                f"{listed}\n\n"
                "Write the report again, filling in this exact skeleton. Keep "
                "every heading exactly as written and put real content under "
                "each one:\n\n"
                f"{skeleton}"
                f"{coverage_note}"
            ),
        },
    ]


def outcomes_from_pack(summary: dict[str, Any]) -> list[str]:
    """Outcome names the report is expected to discuss."""
    return [
        str(name)
        for name in summary.get("allowed_variables", {}).get("outcome_columns", [])
    ]


def generate_validated_report(
    report_pack: dict[str, Any],
    model: str,
    base_url: str | None = None,
    max_attempts: int = 5,
    generate: Callable[..., Iterable[str]] | None = None,
    fallback_models: list[str] | None = None,
) -> LoopResult:
    """Generate a report, repairing it until the validator is satisfied.

    ``generate`` is injectable so the loop can be tested without Ollama.
    """
    summary = fit_summary_to_context(report_pack)
    stream = generate or stream_interpretation
    outcomes = outcomes_from_pack(summary)

    attempts: list[Attempt] = []
    previous_text = ""

    for attempt_number in range(1, max(1, max_attempts) + 1):
        warnings_so_far = attempts[-1].warnings if attempts else []
        plan = plan_attempt(
            attempt_number, summary, previous_text, warnings_so_far, outcomes,
            model=model, fallback_models=fallback_models,
        )

        started = time.monotonic()
        try:
            if plan.by_section:
                text = generate_by_section(
                    summary, plan.model, base_url, stream, outcomes,
                )
            else:
                text = sanitize_interpretation_text(
                    "".join(
                        stream(
                            profile_result=None,
                            audit_result=None,
                            analysis_results={},
                            merged_dataframe=None,
                            outcomes=outcomes,
                            model=plan.model,
                            base_url=base_url,
                            report_pack=summary,
                            messages=plan.messages,
                            option_overrides=plan.overrides,
                        )
                    )
                )
        except OllamaInterpreterError:
            # The first rung failing means Ollama is unreachable or the model
            # wrote nothing; there is nothing to repair, so let the caller see it.
            if not attempts:
                raise
            break

        elapsed = time.monotonic() - started
        warnings = validate_interpretation_text(text, summary).warnings
        attempts.append(Attempt(plan.strategy, text, warnings, elapsed, plan.model))

        if not warnings:
            break
        previous_text = text

    if not attempts:
        return LoopResult(text="", warnings=["No report was generated."])

    # Fewest warnings wins; ties go to the earliest attempt, which is the one
    # written under the least constraint and so usually reads best.
    best = min(attempts, key=lambda attempt: (len(attempt.warnings), attempts.index(attempt)))
    return LoopResult(
        text=best.text,
        warnings=list(best.warnings),
        attempts=attempts,
        strategy=best.strategy,
    )


def replaying_generator(first_text: str, generate: Callable[..., Iterable[str]] | None = None):
    """Feed an already-generated report in as the loop's first attempt.

    The UI streams the first report so the user watches it appear. Handing that
    text back here lets the loop start from it instead of paying for it twice,
    and keeps every escalation decision in this module rather than the panel.
    """
    stream = generate or stream_interpretation
    state = {"replayed": False}

    def generate_or_replay(**kwargs):
        if not state["replayed"]:
            state["replayed"] = True
            return [first_text]
        return stream(**kwargs)

    return generate_or_replay


def build_section_messages(
    summary: dict[str, Any], heading: str, outcomes: list[str],
) -> list[dict[str, str]]:
    """Ask for the body of one section and nothing else."""
    requirement = ""
    if heading == DRIVER_SECTION_HEADING and outcomes:
        # The rung exists because asking for coverage in a whole-report prompt
        # did not work: ministral-3:14b left moisture_percent out of every
        # mock-spec report even when the skeleton named it. A section devoted to
        # the outcomes is a narrower instruction to disobey.
        requirement = (
            " Discuss each of these outcomes by name, even if only to say that "
            "nothing stood out for it: " + ", ".join(outcomes) + "."
        )

    return build_ollama_messages(summary) + [
        {
            "role": "user",
            "content": (
                f"Write only the content that belongs under the heading "
                f"'{heading}'. Do not repeat the heading itself and do not write "
                f"any other section.{requirement}"
            ),
        },
    ]


def generate_by_section(
    summary: dict[str, Any],
    model: str,
    base_url: str | None,
    stream: Callable[..., Iterable[str]],
    outcomes: list[str],
) -> str:
    """Build the report one section at a time and assemble it here.

    The most expensive rung, and the only one that cannot omit a section: the
    headings are written by this function, not by the model. Reached only when
    the skeleton has already failed twice.
    """
    parts = []
    for heading in REQUIRED_HEADINGS:
        body = sanitize_interpretation_text(
            "".join(
                stream(
                    profile_result=None,
                    audit_result=None,
                    analysis_results={},
                    merged_dataframe=None,
                    outcomes=outcomes,
                    model=model,
                    base_url=base_url,
                    report_pack=summary,
                    messages=build_section_messages(summary, heading, outcomes),
                    option_overrides={
                        "temperature": REPAIR_TEMPERATURE,
                        "num_predict": SECTION_TOKEN_BUDGET,
                    },
                )
            )
        )
        # A model asked for a section sometimes writes the heading anyway.
        if body.startswith(heading):
            body = body[len(heading):].lstrip()
        parts.append(f"{heading}\n{body.strip()}\n")

    return "\n".join(parts)


def plan_attempt(
    attempt_number: int,
    summary: dict[str, Any],
    previous_text: str,
    warnings: list[str],
    outcomes: list[str],
    model: str = "",
    fallback_models: list[str] | None = None,
) -> AttemptPlan:
    """Choose the rung for this attempt.

    ``messages`` of ``None`` means the standard prompt, which keeps the first
    attempt byte-for-byte identical to what the app sent before this loop
    existed.
    """
    if attempt_number == 1 or not previous_text:
        return AttemptPlan("base", model)

    # A missing heading is structural: feeding the text back rarely fixes it,
    # while handing over the skeleton does. Go there directly rather than
    # spending a rung on a repair that is unlikely to work.
    if attempt_number == 2 and not missing_headings(previous_text):
        return AttemptPlan(
            "repair", model,
            build_repair_messages(summary, previous_text, warnings),
            {"temperature": REPAIR_TEMPERATURE},
        )

    if attempt_number <= 3:
        return AttemptPlan(
            "skeleton", model,
            build_skeleton_messages(summary, warnings, outcomes),
            {"temperature": REPAIR_TEMPERATURE, "num_predict": SKELETON_TOKEN_BUDGET},
        )

    # Past this point the model has failed the same checks three times. Another
    # attempt with the same model and a stricter prompt is unlikely to differ,
    # so switch model if one was offered. Measured failures are complementary:
    # gpt-oss:20b never dropped an outcome but omitted headings, gemma4:e4b did
    # the reverse. A second model is a second set of blind spots, not a better
    # one.
    alternates = fallback_models or []
    if attempt_number >= 5 and alternates:
        chosen = alternates[(attempt_number - 5) % len(alternates)]
        return AttemptPlan("sections", chosen, by_section=True)

    return AttemptPlan("sections", model, by_section=True)
