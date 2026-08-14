"""PDF report generation for Batch Insight Analyzer."""

from __future__ import annotations

from datetime import datetime
from html import escape
from io import BytesIO
import math
import re
from typing import Any

import pandas as pd
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as ReportLabImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from analysis.evidence import build_operating_window_hints


MAX_INTERPRETATION_PARAGRAPHS = 45
MAX_APPENDIX_ROWS = 220
# A single cell taller than one page aborts the whole PDF build, so cell text is
# capped well below that.
MAX_TABLE_CELL_CHARACTERS = 400


class ReportGenerationError(RuntimeError):
    """Raised when a PDF report cannot be generated."""


def generate_analysis_report_pdf(
    profile_result,
    audit_result,
    analysis_results: dict[str, Any],
    merged_dataframe: pd.DataFrame,
    outcomes: list[str],
    interpretation_markdown: str = "",
    interpretation_validation_warnings: list[str] | None = None,
    spec_assessment=None,
    outcome_objectives: dict[str, str] | None = None,
) -> bytes:
    """Build a PDF report and return it as bytes.

    ``outcome_objectives`` must be the same overrides the UI and the LLM
    evidence pack used, or the printed response bands can point the opposite way
    from what the user saw on screen.
    """
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="Batch Insight Analyzer Report",
    )
    styles = build_report_styles()
    story: list[Any] = []
    ranked_drivers = analysis_results["ranked_drivers"]
    operating_window_hints = build_operating_window_hints(
        ranked_drivers=ranked_drivers,
        merged_dataframe=merged_dataframe,
        profile_result=profile_result,
        outcomes=outcomes,
        outcome_objectives=outcome_objectives,
    )

    add_title_section(story, styles)
    add_data_summary_section(story, styles, profile_result, audit_result, analysis_results)
    add_outcome_statistics_section(story, styles, profile_result)
    add_key_findings_section(
        story,
        styles,
        interpretation_markdown,
        interpretation_validation_warnings or [],
    )
    add_top_driver_sections(story, styles, ranked_drivers, outcomes, operating_window_hints)
    add_pca_section(story, styles, analysis_results, merged_dataframe, outcomes)
    add_spec_section(story, styles, spec_assessment)
    add_appendix_section(story, styles, ranked_drivers)

    try:
        document.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)
    except Exception as error:  # ReportLab raises layout errors that are not app errors.
        raise ReportGenerationError(
            f"The PDF layout engine could not place the report content: {error}"
        ) from error

    return buffer.getvalue()


def add_title_section(story: list[Any], styles: dict[str, ParagraphStyle]) -> None:
    """Add the report title and generation timestamp."""
    story.append(Paragraph("Batch Insight Analyzer Report", styles["title"]))
    story.append(
        Paragraph(
            f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            styles["small_center"],
        )
    )
    story.append(
        Paragraph(
            "Decision-support report for process understanding. Driver rankings show associations in historical data and are not proof of causation.",
            styles["body"],
        )
    )
    story.append(Spacer(1, 0.15 * inch))


def add_data_summary_section(
    story: list[Any],
    styles: dict[str, ParagraphStyle],
    profile_result,
    audit_result,
    analysis_results: dict[str, Any],
) -> None:
    """Add high-level data and model summary."""
    story.append(Paragraph("Data Summary", styles["heading"]))
    variable_counts = profile_result.variable_type_counts
    validation_column = analysis_results.get("validation_order_column") or "Not available"
    high_missing_count = int(profile_result.missingness["flag"].sum())
    near_constant_count = int(len(profile_result.near_constant_columns))

    summary_rows = [
        ["Matched batches", format_value(profile_result.matched_batch_count)],
        ["Process variables", format_value(len(profile_result.process_columns))],
        ["Continuous variables", format_value(variable_counts.get("continuous", 0))],
        ["Categorical variables", format_value(variable_counts.get("categorical", 0))],
        ["Binary variables", format_value(variable_counts.get("binary", 0))],
        ["Selected QC outcomes", ", ".join(profile_result.outcome_columns)],
        ["Columns above 20% missing", format_value(high_missing_count)],
        ["Constant / near-constant process columns", format_value(near_constant_count)],
        ["Time-ordered validation column", validation_column],
    ]
    story.append(build_key_value_table(summary_rows))

    warnings = list(profile_result.warnings)
    if audit_result is not None:
        warnings.extend(getattr(audit_result, "warnings", []))
    if warnings:
        story.append(Spacer(1, 0.08 * inch))
        story.append(Paragraph("Audit warnings", styles["subheading"]))
        for warning in warnings[:10]:
            story.append(Paragraph(f"- {escape_ascii(warning)}", styles["bullet"]))


def add_outcome_statistics_section(
    story: list[Any],
    styles: dict[str, ParagraphStyle],
    profile_result,
) -> None:
    """Add selected outcome summary statistics."""
    outcome_statistics = profile_result.outcome_statistics
    if outcome_statistics.empty:
        return

    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph("Outcome Statistics", styles["heading"]))
    display_columns = ["outcome", "count", "mean", "std", "min", "median", "max"]
    story.append(
        build_dataframe_table(
            outcome_statistics[display_columns],
            max_rows=30,
            col_widths=[1.45 * inch, 0.65 * inch, 0.75 * inch, 0.75 * inch, 0.75 * inch, 0.75 * inch, 0.75 * inch],
        )
    )


def add_key_findings_section(
    story: list[Any],
    styles: dict[str, ParagraphStyle],
    interpretation_markdown: str,
    interpretation_validation_warnings: list[str],
) -> None:
    """Add the local LLM interpretation, if available."""
    story.append(PageBreak())
    story.append(Paragraph("Key Findings", styles["heading"]))

    if not interpretation_markdown.strip():
        story.append(
            Paragraph(
                "No LLM interpretation had been generated when this PDF was created.",
                styles["body"],
            )
        )
        return

    if interpretation_validation_warnings:
        story.append(Paragraph("Report validation warnings", styles["subheading"]))
        for warning in interpretation_validation_warnings[:10]:
            story.append(Paragraph(f"- {escape_ascii(warning)}", styles["bullet"]))
        story.append(Spacer(1, 0.1 * inch))

    for paragraph_text, style_name in markdown_to_report_blocks(interpretation_markdown):
        story.append(Paragraph(paragraph_text, styles[style_name]))


def add_top_driver_sections(
    story: list[Any],
    styles: dict[str, ParagraphStyle],
    ranked_drivers: pd.DataFrame,
    outcomes: list[str],
    operating_window_hints: pd.DataFrame,
) -> None:
    """Add top driver chart and table for each selected outcome."""
    story.append(PageBreak())
    story.append(Paragraph("Top Drivers", styles["heading"]))

    for outcome in outcomes:
        outcome_drivers = ranked_drivers[ranked_drivers["outcome"] == outcome].head(15)
        if outcome_drivers.empty:
            continue

        story.append(Paragraph(escape_ascii(outcome), styles["subheading"]))
        chart_buffer = build_top_drivers_image(outcome_drivers, outcome)
        story.append(ReportLabImage(chart_buffer, width=7.1 * inch, height=3.8 * inch))

        table_columns = [
            "rank",
            "process_variable",
            "confidence",
            "evidence_methods",
            "combined_score",
        ]
        story.append(
            build_dataframe_table(
                outcome_drivers[table_columns],
                max_rows=10,
                col_widths=[0.45 * inch, 2.15 * inch, 0.75 * inch, 1.3 * inch, 0.85 * inch],
            )
        )

        outcome_window_hints = operating_window_hints[
            operating_window_hints["outcome"] == outcome
        ].head(6)
        if not outcome_window_hints.empty:
            story.append(Spacer(1, 0.08 * inch))
            story.append(Paragraph("Suggested Historical Response Bands", styles["subheading"]))
            story.append(
                Paragraph(
                    "Broad bands are quartile summaries. Refined bins are narrower best-observed regions, but they are more noise-sensitive. These are investigation targets, not validated setpoints.",
                    styles["body"],
                )
            )
            hint_columns = [
                "process_variable",
                "objective",
                "range_label",
                "refined_range_label",
                "mean_in_range",
                "refined_mean",
                "mean_outside_range",
                "pattern_type",
            ]
            available_hint_columns = [
                column for column in hint_columns if column in outcome_window_hints.columns
            ]
            story.append(
                build_dataframe_table(
                    outcome_window_hints[available_hint_columns],
                    max_rows=6,
                    col_widths=[
                        1.25 * inch,
                        0.75 * inch,
                        0.85 * inch,
                        0.85 * inch,
                        0.7 * inch,
                        0.7 * inch,
                        0.75 * inch,
                        0.8 * inch,
                    ],
                    font_size=6.6,
                )
            )
        story.append(Spacer(1, 0.16 * inch))


def add_pca_section(
    story: list[Any],
    styles: dict[str, ParagraphStyle],
    analysis_results: dict[str, Any],
    merged_dataframe: pd.DataFrame,
    outcomes: list[str],
) -> None:
    """Add PCA summary plot and explained variance."""
    if not outcomes:
        return

    story.append(PageBreak())
    story.append(Paragraph("PCA Overview", styles["heading"]))
    outcome = outcomes[0]
    pca_image = build_pca_scatter_image(
        analysis_results["pca"],
        merged_dataframe[outcome],
        outcome,
    )
    story.append(ReportLabImage(pca_image, width=7.1 * inch, height=4.1 * inch))

    explained_variance = analysis_results["pca"]["explained_variance"].copy()
    story.append(Paragraph("Explained variance", styles["subheading"]))
    story.append(
        build_dataframe_table(
            explained_variance.head(8),
            max_rows=8,
            col_widths=[1.1 * inch, 1.4 * inch, 1.6 * inch],
        )
    )


def add_spec_section(
    story: list[Any],
    styles: dict[str, ParagraphStyle],
    spec_assessment,
) -> None:
    """Add specs and operating-window assessment."""
    if spec_assessment is None or not getattr(spec_assessment, "has_specs", False):
        return

    variable_summary = spec_assessment.variable_summary
    if variable_summary.empty:
        return

    story.append(PageBreak())
    story.append(Paragraph("Specs and Operating Windows", styles["heading"]))
    story.append(
        Paragraph(
            "This is a historical operating-window assessment, not a proven design space. Confirm with designed or targeted experiments before changing limits.",
            styles["body"],
        )
    )
    display_columns = [
        "variable",
        "role",
        "percent_outside",
        "ppk",
        "max_driver_score",
        "classification",
        "reason",
    ]
    available_columns = [column for column in display_columns if column in variable_summary.columns]
    story.append(
        build_dataframe_table(
            variable_summary[available_columns],
            max_rows=30,
            col_widths=[
                1.4 * inch,
                0.5 * inch,
                0.7 * inch,
                0.5 * inch,
                0.7 * inch,
                1.0 * inch,
                1.95 * inch,
            ][: len(available_columns)],
        )
    )


def add_appendix_section(
    story: list[Any],
    styles: dict[str, ParagraphStyle],
    ranked_drivers: pd.DataFrame,
) -> None:
    """Add full ranked drivers appendix."""
    story.append(PageBreak())
    story.append(Paragraph("Appendix: Ranked Drivers", styles["heading"]))

    appendix_columns = [
        "outcome",
        "rank",
        "process_variable",
        "confidence",
        "evidence_methods",
        "combined_score",
        "pca_score",
        "pls_score",
        "rf_score",
    ]
    available_columns = [column for column in appendix_columns if column in ranked_drivers.columns]
    story.append(
        build_dataframe_table(
            ranked_drivers[available_columns].head(MAX_APPENDIX_ROWS),
            max_rows=MAX_APPENDIX_ROWS,
            col_widths=[
                1.0 * inch,
                0.35 * inch,
                1.55 * inch,
                0.65 * inch,
                1.0 * inch,
                0.65 * inch,
                0.55 * inch,
                0.55 * inch,
                0.55 * inch,
            ],
            font_size=6.3,
        )
    )


def build_top_drivers_image(outcome_drivers: pd.DataFrame, outcome: str) -> BytesIO:
    """Create a simple horizontal bar chart image for top drivers."""
    plot_data = outcome_drivers.sort_values("combined_score", ascending=False).head(15)
    width, height = 1300, 700
    margin_left, margin_right = 380, 100
    margin_top, margin_bottom = 80, 70
    image = PILImage.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(32, bold=True)
    label_font = load_font(22)
    small_font = load_font(18)

    draw.text((margin_left, 25), f"Top drivers: {outcome}", fill="#18201b", font=title_font)
    chart_width = width - margin_left - margin_right
    row_height = max(28, int((height - margin_top - margin_bottom) / max(len(plot_data), 1)))
    max_score = max(float(plot_data["combined_score"].max()), 1e-9)

    for row_index, (_, row) in enumerate(plot_data.iterrows()):
        y = margin_top + row_index * row_height
        score = float(row["combined_score"])
        bar_width = int(chart_width * score / max_score)
        variable_name = shorten_text(str(row["process_variable"]), 34)
        confidence = str(row.get("confidence", ""))
        color = get_confidence_color(confidence)

        draw.text((20, y + 4), variable_name, fill="#18201b", font=label_font)
        draw.rounded_rectangle(
            (margin_left, y, margin_left + bar_width, y + row_height - 8),
            radius=8,
            fill=color,
        )
        draw.text(
            (margin_left + bar_width + 10, y + 3),
            f"{score:.3f}",
            fill="#334155",
            font=small_font,
        )

    return image_to_png_buffer(image)


def build_pca_scatter_image(
    pca_result: dict[str, Any],
    outcome_values: pd.Series,
    outcome_name: str,
) -> BytesIO:
    """Create a simple PC1/PC2 scatter image colored by outcome."""
    transformed = pca_result["transformed_data"].copy()
    if "PC2" not in transformed.columns:
        transformed["PC2"] = 0.0

    aligned_outcomes = pd.to_numeric(
        outcome_values.reindex(transformed["source_index"]),
        errors="coerce",
    )
    plot_data = transformed.assign(outcome=aligned_outcomes.to_numpy()).dropna(
        subset=["PC1", "PC2", "outcome"]
    )

    width, height = 1300, 760
    margin_left, margin_right = 115, 95
    margin_top, margin_bottom = 90, 105
    image = PILImage.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(32, bold=True)
    label_font = load_font(20)
    small_font = load_font(17)

    draw.text((margin_left, 28), f"PCA score plot colored by {outcome_name}", fill="#18201b", font=title_font)
    plot_box = (
        margin_left,
        margin_top,
        width - margin_right,
        height - margin_bottom,
    )
    draw.rectangle(plot_box, outline="#94a3b8", width=2)

    if plot_data.empty:
        draw.text((margin_left + 30, margin_top + 30), "No PCA data available.", fill="#334155", font=label_font)
        return image_to_png_buffer(image)

    pc1_min, pc1_max = expand_range(plot_data["PC1"].min(), plot_data["PC1"].max())
    pc2_min, pc2_max = expand_range(plot_data["PC2"].min(), plot_data["PC2"].max())
    outcome_min, outcome_max = expand_range(plot_data["outcome"].min(), plot_data["outcome"].max())

    for _, row in plot_data.iterrows():
        x = scale_value(row["PC1"], pc1_min, pc1_max, plot_box[0], plot_box[2])
        y = scale_value(row["PC2"], pc2_min, pc2_max, plot_box[3], plot_box[1])
        color = interpolate_color(row["outcome"], outcome_min, outcome_max)
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline="#ffffff")

    draw.text((plot_box[0], height - 65), "PC1", fill="#334155", font=label_font)
    draw.text((22, plot_box[1] + 10), "PC2", fill="#334155", font=label_font)
    draw.text(
        (plot_box[2] - 310, height - 65),
        f"{outcome_name}: low blue, high orange",
        fill="#334155",
        font=small_font,
    )
    return image_to_png_buffer(image)


def markdown_to_report_blocks(markdown_text: str) -> list[tuple[str, str]]:
    """Convert a small markdown subset to report paragraphs."""
    blocks: list[tuple[str, str]] = []
    paragraph_count = 0

    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if paragraph_count >= MAX_INTERPRETATION_PARAGRAPHS:
            blocks.append(("Interpretation truncated for PDF length.", "body"))
            break

        style_name = "body"
        if line.startswith("##"):
            line = line.lstrip("#").strip()
            style_name = "subheading"
        elif line.startswith("- "):
            line = "- " + line[2:].strip()
            style_name = "bullet"
        elif re.match(r"^\d+\.\s+", line):
            style_name = "bullet"

        line = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", escape_ascii(line))
        blocks.append((line, style_name))
        paragraph_count += 1

    return blocks


def build_key_value_table(rows: list[list[Any]]) -> Table:
    """Build a two-column key/value table."""
    data = [[Paragraph(escape_ascii(key), table_cell_style()), Paragraph(escape_ascii(value), table_cell_style())] for key, value in rows]
    table = Table(data, colWidths=[2.35 * inch, 4.65 * inch], hAlign="LEFT")
    table.setStyle(base_table_style(header=False))
    return table


def build_dataframe_table(
    dataframe: pd.DataFrame,
    max_rows: int,
    col_widths: list[float] | None = None,
    font_size: float = 7.2,
) -> Table:
    """Build a ReportLab table from a dataframe."""
    display_dataframe = dataframe.head(max_rows).copy()
    rows = [[escape_ascii(shorten_text(str(column), MAX_TABLE_CELL_CHARACTERS)) for column in display_dataframe.columns]]
    for _, row in display_dataframe.iterrows():
        rows.append(
            [
                escape_ascii(shorten_text(format_value(row[column]), MAX_TABLE_CELL_CHARACTERS))
                for column in display_dataframe.columns
            ]
        )

    table_data = [
        [Paragraph(cell, table_cell_style(font_size=font_size)) for cell in row]
        for row in rows
    ]
    table = Table(table_data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(base_table_style(header=True, font_size=font_size))
    return table


def build_report_styles() -> dict[str, ParagraphStyle]:
    """Create PDF paragraph styles."""
    base_styles = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "BIA Title",
            parent=base_styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#18201b"),
            spaceAfter=8,
        ),
        "heading": ParagraphStyle(
            "BIA Heading",
            parent=base_styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=colors.HexColor("#0b5e58"),
            spaceBefore=8,
            spaceAfter=8,
        ),
        "subheading": ParagraphStyle(
            "BIA Subheading",
            parent=base_styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            textColor=colors.HexColor("#18201b"),
            spaceBefore=6,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "BIA Body",
            parent=base_styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12.2,
            textColor=colors.HexColor("#334155"),
            spaceAfter=5,
        ),
        "bullet": ParagraphStyle(
            "BIA Bullet",
            parent=base_styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.8,
            leading=11.4,
            textColor=colors.HexColor("#334155"),
            leftIndent=10,
            spaceAfter=3,
        ),
        "small_center": ParagraphStyle(
            "BIA Small Center",
            parent=base_styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#66736d"),
            spaceAfter=10,
        ),
    }
    return styles


def table_cell_style(font_size: float = 7.2) -> ParagraphStyle:
    """Return a compact table cell paragraph style."""
    return ParagraphStyle(
        "BIA Table Cell",
        fontName="Helvetica",
        fontSize=font_size,
        leading=font_size + 2.2,
        textColor=colors.HexColor("#18201b"),
    )


def base_table_style(header: bool = True, font_size: float = 7.2) -> TableStyle:
    """Return common table styling."""
    style_commands = [
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d9ded8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
    ]
    if header:
        style_commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eff4f1")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#18201b")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    return TableStyle(style_commands)


def draw_footer(canvas, document) -> None:
    """Draw footer on each PDF page."""
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#66736d"))
    canvas.drawString(0.55 * inch, 0.3 * inch, "Batch Insight Analyzer - local decision-support report")
    canvas.drawRightString(A4[0] - 0.55 * inch, 0.3 * inch, f"Page {document.page}")
    canvas.restoreState()


def format_value(value: Any) -> str:
    """Format table values compactly."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, (int,)):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def escape_ascii(value: Any) -> str:
    """Escape text for ReportLab paragraphs and keep output font-safe."""
    text = str(value)
    text = (
        text.replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2192", "->")
        .replace("\u00b2", "2")
        .replace("\u00b0", " deg ")
    )
    return escape(text, quote=False)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load a common Windows font with a safe fallback."""
    font_names = ["arialbd.ttf" if bold else "arial.ttf", "calibrib.ttf" if bold else "calibri.ttf"]
    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def image_to_png_buffer(image: PILImage.Image) -> BytesIO:
    """Return a PNG buffer ready for ReportLab."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def shorten_text(text: str, max_length: int) -> str:
    """Shorten chart labels without hiding the important prefix."""
    return text if len(text) <= max_length else text[: max_length - 3] + "..."


def get_confidence_color(confidence: str) -> str:
    """Map confidence labels to chart colors."""
    return {
        "high": "#0f766e",
        "medium": "#2563eb",
        "exploratory": "#64748b",
        "low": "#94a3b8",
    }.get(confidence.lower(), "#64748b")


def expand_range(min_value: float, max_value: float) -> tuple[float, float]:
    """Expand a numeric range slightly for plotting."""
    if pd.isna(min_value) or pd.isna(max_value):
        return 0.0, 1.0
    if min_value == max_value:
        return float(min_value) - 1.0, float(max_value) + 1.0
    padding = 0.07 * (float(max_value) - float(min_value))
    return float(min_value) - padding, float(max_value) + padding


def scale_value(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    """Scale one value between numeric ranges."""
    if src_max == src_min:
        return (dst_min + dst_max) / 2
    return dst_min + (float(value) - src_min) / (src_max - src_min) * (dst_max - dst_min)


def interpolate_color(value: float, min_value: float, max_value: float) -> str:
    """Interpolate from blue to orange for PCA outcome coloring."""
    if max_value == min_value:
        ratio = 0.5
    else:
        ratio = max(0.0, min(1.0, (float(value) - min_value) / (max_value - min_value)))
    blue = (37, 99, 235)
    orange = (234, 88, 12)
    color = tuple(int(blue[index] + ratio * (orange[index] - blue[index])) for index in range(3))
    return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
