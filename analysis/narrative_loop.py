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


@dataclass(frozen=True)
class Attempt:
    """One generation and what the validator made of it."""

    strategy: str
    text: str
    warnings: list[str]
    seconds: float


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
    max_attempts: int = 3,
    generate: Callable[..., Iterable[str]] | None = None,
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
        strategy, messages, overrides = plan_attempt(
            attempt_number, summary, previous_text, warnings_so_far, outcomes,
        )

        started = time.monotonic()
        try:
            text = sanitize_interpretation_text(
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
                        messages=messages,
                        option_overrides=overrides,
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
        attempts.append(Attempt(strategy, text, warnings, elapsed))

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


def plan_attempt(
    attempt_number: int,
    summary: dict[str, Any],
    previous_text: str,
    warnings: list[str],
    outcomes: list[str],
) -> tuple[str, list[dict[str, str]] | None, dict[str, Any] | None]:
    """Choose the rung for this attempt.

    Returning ``None`` messages means the standard prompt, which keeps the first
    attempt byte-for-byte identical to what the app sent before this loop
    existed.
    """
    if attempt_number == 1 or not previous_text:
        return "base", None, None

    # A missing heading is structural: feeding the text back rarely fixes it,
    # while handing over the skeleton does. Go there directly rather than
    # spending a rung on a repair that is unlikely to work.
    if attempt_number == 2 and not missing_headings(previous_text):
        return "repair", build_repair_messages(summary, previous_text, warnings), {
            "temperature": REPAIR_TEMPERATURE,
        }

    return "skeleton", build_skeleton_messages(summary, warnings, outcomes), {
        "temperature": REPAIR_TEMPERATURE,
        "num_predict": SKELETON_TOKEN_BUDGET,
    }
