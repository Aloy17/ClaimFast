"""Render a one-page labelling guide PDF. Condensed from data/labelling_guide.md; rules are unchanged."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "submission" / "Labelling_Guide_OnePage.pdf"

base = getSampleStyleSheet()
body = ParagraphStyle("b", parent=base["BodyText"], fontName="Helvetica", fontSize=7.6, leading=9.4, spaceAfter=2)
h1 = ParagraphStyle("h1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=12.5, spaceAfter=2, textColor=colors.HexColor("#1F3864"))
h2 = ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=8.6, spaceBefore=4, spaceAfter=2, textColor=colors.HexColor("#1F3864"))
cell = ParagraphStyle("c", parent=body, fontSize=7.2, leading=8.8, spaceAfter=0)
hdr = ParagraphStyle("h", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)


def table(rows, widths):
    data = [[Paragraph(str(x), hdr if i == 0 else cell) for x in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3864")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BFBFBF")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
    ]))
    return t


P = lambda t: Paragraph(t, body)
story = [
    Paragraph("ClaimFast Labelling Guide (one page)", h1),
    P("Rules for labelling motor-insurance First Notice of Loss messages. Full version: <i>data/labelling_guide.md</i>. Dataset: 20 development + 60 test synthetic messages."),
    Paragraph("1. Fields", h2),
    table([
        ["Field", "Allowed values and format"],
        ["claim_type", "own_damage, third_party_liability, theft, total_loss, glass_only, not_a_claim"],
        ["urgency", "high, medium, low"],
        ["policy_number, location", "Text as written, or null"],
        ["incident_date", "YYYY-MM-DD only if the exact date is stated; relative dates (yesterday, today) = null"],
        ["vehicle_reg_no", "Uppercase, no spaces or hyphens, or null"],
        ["estimated_amount_inr", "Integer rupees, or null"],
        ["injury_reported, police_report_filed", "true = stated; false = clearly denied; null = not mentioned"],
    ], [4.3 * cm, 13.2 * cm]),
    Paragraph("2. Claim type", h2),
    table([
        ["Type", "Use when"],
        ["own_damage", "Only the policyholder's vehicle is damaged. Another vehicle or a hit-and-run driver does not by itself make it third-party liability."],
        ["third_party_liability", "Another person's vehicle or property is damaged, or someone other than the policyholder is injured (including a passenger in the insured vehicle). Proven fault is not required. Do not infer harm that is not reported."],
        ["theft", "Vehicle or a part is stolen. In an FNOL context a missing vehicle counts as suspected theft; do not infer that a police report was filed."],
        ["total_loss", "Destroyed or damaged beyond repair. Takes priority over other types; record any injury in injury_reported."],
        ["glass_only", "Damage limited to glass (windshield, windows, mirror glass)."],
        ["not_a_claim", "Question, policy request or other message that is not a claim."],
    ], [3.3 * cm, 14.2 * cm]),
    Paragraph("3. Urgency", h2),
    P("<b>high</b>: any injury is reported, or the vehicle is undrivable or unsafe. <b>medium</b>: a genuine loss with no injury and no unsafe or undrivable vehicle. "
      "<b>low</b>: minor or non-urgent matters, such as glass-only damage or a non-claim. Total loss alone is <b>not</b> high; do not infer that a vehicle is unsafe because it is beyond repair."),
    Paragraph("4. Missing and ambiguous information", h2),
    P("Use <b>null</b> when a value is not stated or cannot be determined confidently. Never invent policy numbers, dates, locations, registrations or amounts. "
      "For <b>not_a_claim</b>, still extract a policy number or registration that the message states; keep incident_date and estimated_amount_inr null when no incident or estimate is reported. "
      "Extract a stated location only if it is relevant to the event described."),
    Paragraph("5. Special cases", h2),
    P("<b>Multiple incidents</b>: label the incident tied to the claim; do not mix details across incidents. <b>Forwarded messages</b>: use the whole chain but do not treat quoted or unrelated mail as a new incident. "
      "<b>Hinglish and typos</b>: use the intended meaning when reasonably clear; do not guess when it is not. <b>Location convention</b>: gold labels may keep words such as 'near' or the venue type (for example 'near Wakad bridge')."),
    Paragraph("6. Reviewed ambiguous records", h2),
    table([
        ["Record", "Decision", "Reason"],
        ["DEV009, TEST029, TEST047", "theft, flagged ambiguous", "Vehicle reported missing in an FNOL context; theft suspected, not confirmed."],
        ["TEST009", "theft", "Customer explicitly suspects theft."],
        ["TEST027, TEST041", "own_damage (corrected)", "Only the insured vehicle's damage is reported."],
        ["TEST046", "third_party_liability", "Insured vehicle and another party's vehicle are both damaged."],
    ], [4.3 * cm, 4.0 * cm, 9.2 * cm]),
    Spacer(1, 2),
    P("<b>Independent check:</b> a second reviewer labels 20 sampled messages blind using <i>data/independent_review_template.csv</i>; "
      "<i>scripts/compare_independent_review.py</i> reports agreement and disagreements."),
]

OUT.parent.mkdir(parents=True, exist_ok=True)
doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm, topMargin=1.3 * cm, bottomMargin=1.2 * cm, title="ClaimFast Labelling Guide")
doc.build(story)
print("wrote", OUT)
