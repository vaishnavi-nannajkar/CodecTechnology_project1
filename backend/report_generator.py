"""
PDF Report Generator
Generates a professional PDF report from sentiment analysis results.
Uses ReportLab or falls back to HTML-based generation.
"""

import io
import logging
import pandas as pd
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    _reportlab_available = True
except ImportError:
    _reportlab_available = False
    logger.warning("ReportLab not installed. Using plain-text PDF fallback.")


def _create_html_pdf_fallback(df: pd.DataFrame, keyword: str) -> bytes:
    """Simple HTML-based report as bytes when ReportLab is unavailable."""
    pos = (df["sentiment"] == "Positive").sum()
    neg = (df["sentiment"] == "Negative").sum()
    neu = (df["sentiment"] == "Neutral").sum()
    total = len(df)
    rep_score = (pos - neg) / total * 100 if total else 0

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Sentiment Report - {keyword}</title>
<style>
  body {{ font-family: Inter, Arial, sans-serif; margin: 40px; color: #1e293b; }}
  h1 {{ color: #6366f1; }}
  .kpi {{ display: inline-block; margin: 10px; padding: 20px; border-radius: 8px;
          background: #f8f9fb; border-left: 4px solid #6366f1; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
  th {{ background: #6366f1; color: white; padding: 10px; text-align: left; }}
  td {{ padding: 8px; border-bottom: 1px solid #e2e8f0; font-size: 0.85em; }}
  .pos {{ color: #10b981; font-weight: 600; }}
  .neg {{ color: #ef4444; font-weight: 600; }}
  .neu {{ color: #f59e0b; font-weight: 600; }}
</style>
</head>
<body>
<h1>🧠 AI Sentiment Intelligence Report</h1>
<p><strong>Keyword:</strong> {keyword}</p>
<p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
<hr>
<h2>Summary</h2>
<div>
  <div class="kpi"><strong>{total}</strong><br>Total Tweets</div>
  <div class="kpi" style="border-color:#10b981"><strong>{pos}</strong><br>Positive</div>
  <div class="kpi" style="border-color:#ef4444"><strong>{neg}</strong><br>Negative</div>
  <div class="kpi" style="border-color:#f59e0b"><strong>{neu}</strong><br>Neutral</div>
  <div class="kpi" style="border-color:#8b5cf6"><strong>{rep_score:+.1f}</strong><br>Reputation Score</div>
</div>
<h2>Tweet Results (Top 50)</h2>
<table>
  <tr><th>Tweet</th><th>Sentiment</th><th>Confidence</th></tr>
"""
    for _, row in df.head(50).iterrows():
        cls = row["sentiment"].lower()[:3]
        html += f"""
  <tr>
    <td>{row['tweet'][:150]}</td>
    <td class="{cls}">{row['sentiment']}</td>
    <td>{row['confidence']:.1%}</td>
  </tr>"""

    html += """
</table>
</body>
</html>"""
    return html.encode("utf-8")


def _create_reportlab_pdf(df: pd.DataFrame, keyword: str) -> bytes:
    """Generate a professional PDF using ReportLab."""
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "Title", parent=styles["Title"],
        fontSize=22, textColor=colors.HexColor("#6366f1"),
        spaceAfter=6,
    )
    heading_style = ParagraphStyle(
        "Heading", parent=styles["Heading2"],
        fontSize=14, textColor=colors.HexColor("#1e293b"),
        spaceBefore=16, spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#475569"),
        spaceAfter=4,
    )

    # Stats
    pos = (df["sentiment"] == "Positive").sum()
    neg = (df["sentiment"] == "Negative").sum()
    neu = (df["sentiment"] == "Neutral").sum()
    total = len(df)
    rep_score = (pos - neg) / total * 100 if total else 0

    story = []

    # Title
    story.append(Paragraph("🧠 AI Sentiment Intelligence Report", title_style))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#6366f1"), thickness=2))
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph(f"<b>Keyword / Brand:</b> {keyword}", body_style))
    story.append(Paragraph(
        f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style
    ))
    story.append(Spacer(1, 0.3 * inch))

    # Summary table
    story.append(Paragraph("Summary Statistics", heading_style))
    summary_data = [
        ["Metric", "Value"],
        ["Total Tweets", str(total)],
        ["Positive", f"{pos} ({pos/total*100:.1f}%)" if total else "0"],
        ["Negative", f"{neg} ({neg/total*100:.1f}%)" if total else "0"],
        ["Neutral",  f"{neu} ({neu/total*100:.1f}%)" if total else "0"],
        ["Reputation Score", f"{rep_score:+.1f}"],
        ["Avg Confidence", f"{df['confidence'].mean():.1%}" if total else "N/A"],
    ]

    table = Table(summary_data, colWidths=[3.5 * inch, 3.5 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6366f1")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fb")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.4 * inch))

    # Tweet results
    story.append(Paragraph("Analyzed Tweets (Top 50)", heading_style))
    tweet_data = [["#", "Tweet", "Sentiment", "Confidence"]]
    color_map = {
        "Positive": colors.HexColor("#10b981"),
        "Negative": colors.HexColor("#ef4444"),
        "Neutral": colors.HexColor("#f59e0b"),
    }

    sample = df.head(50)
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        tweet_data.append([
            str(i),
            row["tweet"][:100] + ("..." if len(row["tweet"]) > 100 else ""),
            row["sentiment"],
            f"{row['confidence']:.1%}",
        ])

    tweet_table = Table(
        tweet_data,
        colWidths=[0.4 * inch, 4.2 * inch, 1.1 * inch, 1.0 * inch],
    )
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fb")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#e2e8f0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("WORDWRAP", (1, 0), (1, -1), True),
    ]
    # Color sentiment column
    for row_idx, (_, row) in enumerate(sample.iterrows(), 1):
        sent_color = color_map.get(row["sentiment"], colors.black)
        style_cmds.append(("TEXTCOLOR", (2, row_idx), (2, row_idx), sent_color))
        style_cmds.append(("FONTNAME", (2, row_idx), (2, row_idx), "Helvetica-Bold"))

    tweet_table.setStyle(TableStyle(style_cmds))
    story.append(tweet_table)

    story.append(Spacer(1, 0.4 * inch))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "Generated by AI Sentiment Intelligence System | Powered by ML & BERT",
        ParagraphStyle("Footer", parent=styles["Normal"],
                       fontSize=8, textColor=colors.HexColor("#f1f2f3"),
                       alignment=TA_CENTER),
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


def generate_pdf_report(df: pd.DataFrame, keyword: str) -> bytes:
    """
    Generate a PDF report.
    Returns PDF bytes (or HTML bytes as fallback).
    """
    if _reportlab_available:
        try:
            return _create_reportlab_pdf(df, keyword)
        except Exception as e:
            logger.error(f"ReportLab PDF generation failed: {e}")

    return _create_html_pdf_fallback(df, keyword)