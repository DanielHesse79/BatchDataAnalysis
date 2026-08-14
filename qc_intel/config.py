"""Method configuration.

TOML rather than YAML so the prototype needs no third-party parser, and because
TOML's types are unambiguous - a threshold cannot silently arrive as a string.

The central rule: `acceptance` comes from method validation and the analytical
plan and is never derived from data. `trending` is exploratory and never decides
whether a result passed. They are separate blocks so no UI can confuse them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = PACKAGE_ROOT / "config"


class ConfigError(ValueError):
    """Raised when a method configuration cannot be used."""


@dataclass(frozen=True)
class QcLevelConfig:
    """Fixed acceptance criteria for one QC level."""

    name: str
    nominal: float
    max_bias_percent: float
    evaluation_type: str = "quantitative"

    @property
    def acceptance_half_width(self) -> float:
        """Half-width of the acceptance window, in percent bias."""
        return float(self.max_bias_percent)


@dataclass(frozen=True)
class BaselineConfig:
    """A frozen period used for exploratory control limits.

    Never auto-extended. Recomputing the baseline over history that already
    contains the drift inflates the standard deviation until the chart hides
    the signal it exists to show.
    """

    start: str
    end: str
    approved_by: str = "pending"


@dataclass(frozen=True)
class TrendingConfig:
    """Exploratory detection thresholds. Not acceptance criteria."""

    baseline: BaselineConfig
    step_min_shift_fraction: float = 0.33
    step_window_runs: int = 8
    trend_min_runs: int = 8
    trend_min_total_shift_percent: float = 4.0
    trend_max_kendall_p: float = 0.05
    variance_cv_ratio: float = 1.5
    variance_min_runs: int = 12
    variance_max_p: float = 0.05
    event_lookback_days: tuple[int, ...] = (1, 7, 30)


@dataclass(frozen=True)
class MethodConfig:
    """Everything the engine needs to evaluate one method."""

    method_id: str
    units: str
    lloq: float
    uloq: float
    qc_levels: dict[str, QcLevelConfig]
    trending: TrendingConfig
    run_rule: str = "four_six_fifteen"
    notes: str = ""

    def level(self, qc_level: str) -> QcLevelConfig | None:
        """Return the configuration for a QC level, if it is configured."""
        return self.qc_levels.get(qc_level)


def load_method_config(path: Path | str) -> MethodConfig:
    """Load and validate one method configuration file."""
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path.name} is not valid TOML: {error}") from error

    for required_key in ("method_id", "acceptance", "trending"):
        if required_key not in raw:
            raise ConfigError(f"{path.name} is missing the '{required_key}' section.")

    acceptance = raw["acceptance"]
    level_block = acceptance.get("qc_levels")
    if not level_block:
        raise ConfigError(f"{path.name} defines no QC levels under [acceptance.qc_levels].")

    qc_levels: dict[str, QcLevelConfig] = {}
    for level_name, level_values in level_block.items():
        missing = [key for key in ("nominal", "max_bias_percent") if key not in level_values]
        if missing:
            raise ConfigError(
                f"{path.name}: QC level '{level_name}' is missing {', '.join(missing)}."
            )
        qc_levels[level_name] = QcLevelConfig(
            name=level_name,
            nominal=float(level_values["nominal"]),
            max_bias_percent=float(level_values["max_bias_percent"]),
            evaluation_type=str(level_values.get("evaluation_type", "quantitative")),
        )

    trending_block = raw["trending"]
    baseline_block = trending_block.get("baseline")
    if not baseline_block or "start" not in baseline_block or "end" not in baseline_block:
        raise ConfigError(
            f"{path.name}: [trending.baseline] needs an explicit start and end. "
            "A baseline derived from all history hides the drift it should reveal."
        )

    trending = TrendingConfig(
        baseline=BaselineConfig(
            start=str(baseline_block["start"]),
            end=str(baseline_block["end"]),
            approved_by=str(baseline_block.get("approved_by", "pending")),
        ),
        step_min_shift_fraction=float(trending_block.get("step_min_shift_fraction", 0.33)),
        step_window_runs=int(trending_block.get("step_window_runs", 8)),
        trend_min_runs=int(trending_block.get("trend_min_runs", 8)),
        trend_min_total_shift_percent=float(
            trending_block.get("trend_min_total_shift_percent", 4.0)
        ),
        trend_max_kendall_p=float(trending_block.get("trend_max_kendall_p", 0.05)),
        variance_cv_ratio=float(trending_block.get("variance_cv_ratio", 1.5)),
        variance_min_runs=int(trending_block.get("variance_min_runs", 12)),
        variance_max_p=float(trending_block.get("variance_max_p", 0.05)),
        event_lookback_days=tuple(trending_block.get("event_lookback_days", [1, 7, 30])),
    )

    return MethodConfig(
        method_id=str(raw["method_id"]),
        units=str(raw.get("units", "")),
        lloq=float(raw.get("lloq", 0.0)),
        uloq=float(raw.get("uloq", 0.0)),
        qc_levels=qc_levels,
        trending=trending,
        run_rule=str(acceptance.get("run_rule", "four_six_fifteen")),
        notes=str(raw.get("notes", "")),
    )


def load_all_method_configs(config_dir: Path | str = DEFAULT_CONFIG_DIR) -> dict[str, MethodConfig]:
    """Load every method configuration in a directory, keyed by method_id."""
    config_dir = Path(config_dir)
    configs: dict[str, MethodConfig] = {}
    for path in sorted(config_dir.glob("*.toml")):
        method_config = load_method_config(path)
        if method_config.method_id in configs:
            raise ConfigError(f"Duplicate method_id '{method_config.method_id}' in {path.name}.")
        configs[method_config.method_id] = method_config
    return configs
