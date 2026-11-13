"""Build the ClaimFast final evaluation report PDF from saved analysis outputs.

Offline only. Reads outputs/final_benchmark/analysis/analysis.json and
raw_responses.jsonl; writes submission/ClaimFast_Evaluation_Report.pdf.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "outputs" / "final_benchmark" / "analysis" / "analysis.json"
LOG = ROOT / "outputs" / "final_benchmark" / "raw_responses.jsonl"
OUT_DIR = ROOT / "submission"
FIG_DIR = ROOT / "outputs" / "final_benchmark" / "analysis" / "figures"
PDF = OUT_DIR / "ClaimFast_Evaluation_Report.pdf"

STRATS = ["A", "B1", "B2", "B3", "C"]
NAMES = {"A": "A zero-shot", "B1": "B1 k=3", "B2": "B2 k=5", "B3": "B3 k=8", "C": "C reasoning"}
FIELDS = ["claim_type", "urgency", "policy_number", "incident_date", "location",
          "vehicle_reg_no", "estimated_amount_inr", "injury_reported", "police_report_filed"]
INPUT_PRICE, OUTPUT_PRICE = 0.075, 0.30  # USD per million tokens, Groq docs, 2026-11-13


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def location_equivalent(gold, pred):
    if gold == pred:
        return True
    return bool(gold and pred and pred.lower().strip() in gold.lower())


def load():
    report = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in LOG.open(encoding="utf-8") if line.strip()]
    return report, records


def normalised_exact(records):
    """Record exact match when location is compared leniently (label convention)."""
    out = {}
    for s in STRATS:
        rs = [r for r in records if r["strategy"] == s]
        ok = 0
        for r in rs:
            p = r.get("predicted") or {}
            good = all(
                (location_equivalent(r["expected"][f], p.get(f)) if f == "location"
                 else p.get(f) == r["expected"][f])
                for f in FIELDS
            )
            ok += good
        out[s] = ok / len(rs)
    return out


def heatmap(matrix, labels, title, path):
    short = {"third_party_liability": "third_party", "not_a_claim": "not_claim",
             "own_damage": "own_dmg", "total_loss": "total_loss", "glass_only": "glass",
             "theft": "theft", "high": "high", "medium": "medium", "low": "low"}
    rows = [l for l in labels if l in matrix]
    cols = [l for l in labels if l != "__invalid_prediction__"]
    data = [[matrix[a][p] for p in cols] for a in rows]
    fig, ax = plt.subplots(figsize=(3.6, 3.1), dpi=200)
    ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([short.get(c, c) for c in cols], rotation=40, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([short.get(c, c) for c in rows], fontsize=6.5)
    peak = max(max(r) for r in data) or 1
    for i, row in enumerate(data):
        for j, v in enumerate(row):
            ax.text(j, i, v, ha="center", va="center", fontsize=7,
                    color="white" if v > peak * 0.55 else "black")
    ax.set_xlabel("Predicted", fontsize=7)
    ax.set_ylabel("Gold", fontsize=7)
    ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    report, records = load()
    ps = report["per_strategy"]
    rs = report["run_statistics"]
    norm = normalised_exact(records)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for s in ("A", "B3", "C"):
        c = ps[s]["classification"]
        heatmap(c["claim_type"]["confusion_matrix"], c["claim_type"]["labels"],
                f"claim_type, {NAMES[s]} (3 runs pooled)", FIG_DIR / f"cm_claim_{s}.png")
        heatmap(c["urgency"]["confusion_matrix"], c["urgency"]["labels"],
                f"urgency, {NAMES[s]} (3 runs pooled)", FIG_DIR / f"cm_urgency_{s}.png")

    # trade-off chart
    fig, ax = plt.subplots(figsize=(5.2, 2.6), dpi=200)
    xs = [ps[s]["performance"]["cost_estimate_usd"]["observed_cost_usd"] / 180 * 1000 for s in STRATS]
    ys = [ps[s]["classification"]["claim_type"]["accuracy"] * 100 for s in STRATS]
    ax.scatter(xs, ys, s=40, color="#1F3864")
    for s, x, y in zip(STRATS, xs, ys):
        ax.annotate(s, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xlabel("Cost per 1,000 claims (USD)", fontsize=8)
    ax.set_ylabel("claim_type accuracy (%)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "tradeoff.png")
    plt.close(fig)

    base = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=base["BodyText"], fontName="Helvetica", fontSize=9, leading=12, alignment=TA_LEFT, spaceAfter=4)
    h1 = ParagraphStyle("h1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=15, spaceAfter=6, textColor=colors.HexColor("#1F3864"))
    h2 = ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#1F3864"))
    small = ParagraphStyle("small", parent=body, fontSize=7.5, leading=9.5)
    cell = ParagraphStyle("cell", parent=body, fontSize=7.5, leading=9, spaceAfter=0)

    hstyle = ParagraphStyle("hdr", parent=body, fontName="Helvetica-Bold", fontSize=6.8, leading=8, textColor=colors.white, spaceAfter=0)

    def table(data, widths, header=True, fs=7.5):
        if header:
            data = [[Paragraph(str(x), hstyle) if isinstance(x, str) else x for x in data[0]]] + data[1:]
        t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
        style = [
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"), ("FONTSIZE", (0, 0), (-1, -1), fs),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BFBFBF")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ]
        if header:
            style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3864")),
                      ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                      ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
        t.setStyle(TableStyle(style))
        return t

    P = lambda text, st=body: Paragraph(text, st)
    story = []

    # ---------------- Page 1 ----------------
    story += [P("ClaimFast: Prompt Strategy Evaluation for Motor Claim Triage", h1),
              P("Case Study 1, MCA Semester I, Agentic AI. Model: openai/gpt-oss-20b on Groq. Repository: Aloy17/ClaimFast.", small)]
    story += [P("1. Problem statement", h2),
              P("A motor insurer receives free-text First Notice of Loss (FNOL) messages in English and Hinglish. "
                "Each message must be classified into one of six claim types (own_damage, third_party_liability, theft, total_loss, glass_only, not_a_claim), "
                "given an urgency (high, medium, low), and mapped to seven extraction fields (policy number, incident date, location, vehicle registration, "
                "estimated amount in INR, injury reported, police report filed). Missing values must be null, and false is allowed only when the message denies the condition. "
                "The question is which prompting strategy gives the best balance of accuracy, reliability, cost and speed on the same model."),
              P("2. Data and methodology", h2),
              P("<b>Dataset.</b> 80 synthetic, anonymised messages: 20 development records (<i>data/dev.jsonl</i>) used to write prompts and pick few-shot examples, and 60 held-out test records "
                "(<i>data/test.jsonl</i>) used only for evaluation. Test claim types: third_party_liability 13, own_damage 11, total_loss 10, theft 9, glass_only 9, not_a_claim 8. "
                "Urgency: medium 31, low 17, high 12. Edge cases cover Hinglish, multiple incidents, forwarded messages, missing fields and ambiguous theft. "
                "Rules are written in <i>data/labelling_guide.md</i>. Labels were written by one annotator; the independent 20-record review is not yet complete (see Limitations)."),
              P("<b>Strategies.</b> All use the same system prompt definitions, schema and validator."),
              ]
    strat_rows = [["ID", "Strategy", "Examples", "What differs"],
                  ["A", "Zero-shot", "0", "Instructions and definitions only"],
                  ["B1", "Few-shot", "3", "DEV001, DEV008, DEV003 added to the prompt"],
                  ["B2", "Few-shot", "5", "Adds DEV010, DEV005"],
                  ["B3", "Few-shot", "8", "Adds DEV006, DEV014, DEV017"],
                  ["C", "Reasoning-guided", "0", "Explicit decision checks and a short evidence summary before the JSON"],
                  ["D", "Not implemented", "-", "Optional in the assignment; not built"]]
    story += [table(strat_rows, [1.2 * cm, 3.2 * cm, 1.8 * cm, 11 * cm]), Spacer(1, 6)]
    story += [P("<b>Experiment settings.</b> Temperature 0, reasoning effort low, maximum 2,000 completion tokens, one model for every run. "
                "5 strategies x 3 runs x 60 test records = 900 requests, run on 2026-11-06. Raw responses, token counts and latency are saved in "
                "<i>outputs/final_benchmark/</i>. All 900 returned valid JSON that passed validation (0 API errors, 0 parse errors). Few-shot examples come only from the development set."),
              P("<b>Metrics.</b> Per-field exact match; accuracy, macro precision, recall and F1 for claim_type and urgency; micro F1 over the seven extraction fields; "
                "record exact match (all nine fields correct); recall for injury_reported = true and for urgency = high; pairwise agreement across the three runs; "
                "latency, tokens and cost. Cost uses Groq list prices of $0.075 per million input tokens and $0.30 per million output tokens (Groq model page, read 2026-11-13). "
                "Metrics are computed offline by <i>analysis.py</i> from the saved log.")]

    # ---------------- Page 2: results ----------------
    story += [PageBreak(), P("3. Results", h2)]
    hdr = ["Strategy", "claim_type acc", "claim_type macro F1", "urgency acc", "urgency macro F1", "Extraction micro F1", "Record exact (strict)", "Record exact (location normalised)"]
    rows = [hdr]
    for s in STRATS:
        m = ps[s]
        rows.append([NAMES[s], pct(m["classification"]["claim_type"]["accuracy"]), f"{m['classification']['claim_type']['macro_f1']:.3f}",
                     pct(m["classification"]["urgency"]["accuracy"]), f"{m['classification']['urgency']['macro_f1']:.3f}",
                     f"{m['extraction_micro']['f1']:.3f}", pct(m["record_exact_match"]["accuracy"]), pct(norm[s])])
    story += [table(rows, [2.7 * cm, 1.9 * cm, 2.0 * cm, 1.8 * cm, 2.0 * cm, 2.2 * cm, 1.9 * cm, 2.4 * cm], fs=7),
              P("Each figure pools the three runs (180 predictions per strategy). Strict record exact match is low for every strategy because the gold <i>location</i> label "
                "often keeps words such as 'near' or 'outside the gym in' while the model returns the place name (30 of 60 test records). "
                "The normalised column treats those as equal; it changes no other field.", small)]

    story += [P("Per-field exact match", h2)]
    short = ["claim_type", "urgency", "policy", "date", "location", "reg no", "amount", "injury", "police"]
    frows = [["Strategy"] + short]
    for s in STRATS:
        frows.append([NAMES[s]] + [pct(ps[s]["field_metrics"][f]["exact_match"], 0) for f in FIELDS])
    story += [table(frows, [2.7 * cm] + [1.65 * cm] * 9, fs=7),
              P("Location exact match (50 to 60%) is depressed by the label-format issue above. Policy number is lowest for B1 and B2 (85 to 87%) because the models return null on "
                "not_a_claim messages that do state a policy number; B3 recovers it (98%). Incident date is 100% for every strategy.", small)]

    story += [P("Reliability, speed and cost", h2)]
    crows = [["Strategy", "Valid JSON", "Injury recall", "High-urgency recall", "Mean latency (s)", "p95 latency (s)", "Mean prompt tok.", "Mean output tok.", "Cost per 1,000 claims"]]
    for s in STRATS:
        m = ps[s]
        rel, perf = m["reliability"], m["performance"]
        cost = perf["cost_estimate_usd"]["observed_cost_usd"] / m["record_exact_match"]["count"] * 1000
        crows.append([NAMES[s], pct(rel["json_valid"]["rate"], 0),
                      f"{pct(rel['injury_recall']['recall'], 0)} ({rel['injury_recall']['true_positive']}/{rel['injury_recall']['positive_support']})",
                      pct(rel["high_urgency_recall"], 0), f"{perf['latency_seconds']['mean']:.2f}", f"{perf['latency_seconds']['p95']:.1f}",
                      f"{perf['prompt_tokens']['mean']:.0f}", f"{perf['completion_tokens']['mean']:.0f}", f"${cost:.3f}"])
    story += [table(crows, [2.5 * cm, 1.5 * cm, 2.0 * cm, 2.0 * cm, 1.8 * cm, 1.7 * cm, 1.8 * cm, 1.8 * cm, 2.0 * cm], fs=7),
              P("Latency is wall-clock time on a shared free-tier endpoint and varies between calls; B2 is faster than A on average even though its prompt is nearly twice as long, "
                "which shows the timings carry network and queue noise. Treat latency differences under about 2 seconds as unreliable. Token counts and cost are exact.", small)]

    # ---------------- Page 3: interpretation ----------------
    story += [P("4. Interpretation", h2)]
    a, b1, b2, b3, c = (ps[s]["classification"]["claim_type"]["accuracy"] for s in STRATS)
    story += [
        P(f"<b>Zero-shot against few-shot.</b> Few-shot examples help claim_type: {pct(a)} (A), {pct(b1)} (B1), {pct(b2)} (B2), {pct(b3)} (B3). "
          f"Urgency improves more: {pct(ps['A']['classification']['urgency']['accuracy'])} for A against {pct(ps['B2']['classification']['urgency']['accuracy'])} for B2 and "
          f"{pct(ps['B3']['classification']['urgency']['accuracy'])} for B3. The zero-shot prompt marks 41 of 93 medium cases as high; the examples remove nearly all of that."),
        P("<b>Do the gains flatten?</b> Not cleanly. claim_type rises from k=0 to k=3, dips at k=5, and rises again at k=8, so the curve is not monotonic and has not flattened by k=8. "
          "With 60 records, one record is 1.7 percentage points per run, and the dip at k=5 is one to two records. The run-to-run spread is small (standard deviation under 1.5 points), "
          "but the test set is small, so differences of 2 to 3 points between neighbouring strategies should not be treated as proven. "
          "The clear signal is the large first step (A to B1 and B2) on urgency and the best result at k=8."),
        P(f"<b>Reasoning-guided (C).</b> C reaches {pct(c)} claim_type and {pct(ps['C']['classification']['urgency']['accuracy'])} urgency without any examples, better than A on both "
          f"but behind B3. It uses about 2.1 times the output tokens of A (282 against 134), has the highest mean latency (9.2 s) and costs about as much as B3 per 1,000 claims. "
          "It did not fix the policy-number or injury-denial errors."),
        P("<b>Consistency.</b> Pairwise agreement between runs for the same strategy and record is "
          f"{pct(report['run_agreement']['exact_record_agreement'])} overall; every field is at least 98% (incident date, registration and amount 100%). Record exact match varies by about "
          "3 points or less between runs for any strategy, and B1 is identical across runs. At temperature 0 the outputs are close to deterministic; "
          "the remaining variation comes from the provider, not from the prompt."),
        P("<b>Safety-critical recall.</b> Injury recall (injury_reported = true) is 100% for every strategy (27 of 27 positive predictions across three runs), and high-urgency recall is 100% for every strategy. "
          "No injured claim was missed or downgraded in any of the 900 runs. The cost of the weaker strategies is over-escalation (A raised 41 medium cases to high), not under-triage.")]

    # trade-off figure
    story += [Image(str(FIG_DIR / "tradeoff.png"), width=10.5 * cm, height=5.25 * cm),
              P("Figure 1. claim_type accuracy against cost per 1,000 claims. B3 is the most accurate and about 1.8 times the cost of A.", small)]

    # ---------------- Page 4: confusion matrices ----------------
    story += [PageBreak(), P("5. Confusion matrices", h2),
              P("Counts pool the three runs. Zero-shot (A), the best strategy (B3), and reasoning-guided (C) are shown; matrices for every strategy are in <i>outputs/final_benchmark/analysis/analysis.json</i>.")]
    imgs = [
        [Image(str(FIG_DIR / f"cm_claim_{s}.png"), width=5.45 * cm, height=4.7 * cm)
         for s in ("A", "B3", "C")],
        [Image(str(FIG_DIR / f"cm_urgency_{s}.png"), width=5.45 * cm, height=4.7 * cm)
         for s in ("A", "B3", "C")],
    ]
    story += [Table(imgs, colWidths=[5.7 * cm] * 3), Spacer(1, 4)]
    ca = ps["A"]["classification"]["claim_type"]["per_class"]
    story += [P(f"<b>Reading the matrices.</b> Under A, 14 of 39 third_party_liability predictions and 3 of 30 total_loss predictions become own_damage, so own_damage precision is "
                f"{pct(ca['own_damage']['precision'], 0)} and third_party_liability recall is {pct(ca['third_party_liability']['recall'], 0)}. B3 removes all of those. "
                "The only claim_type errors left in B3 are two predictions of TEST029 (a missing vehicle) as not_a_claim, a case the labelling guide itself flags as ambiguous. "
                "On urgency, A puts 41 medium cases into high, while B3 leaves 4 medium as high and 2 as low. No high case was missed in either strategy.")]

    # ---------------- Page 5: subgroup analysis ----------------
    story += [
        PageBreak(),
        P("6. Subgroup analysis", h2),
        P("These slices examine three specific edge-case groups. Counts below pool three runs; predictions are repeated evaluations of the same messages, not additional independent test messages.", small),
    ]

    subgroup_data = [
        ("Ambiguous theft", "3 messages; 9 predictions per strategy",
         [
             ["Strategy", "Claim acc.", "Urgency acc.", "Extract. F1", "Exact records"],
             ["A", "100.0%", "100.0%", "0.800", "3/9"],
             ["B1", "66.7%", "66.7%", "0.556", "0/9"],
             ["B2", "66.7%", "66.7%", "0.588", "0/9"],
             ["B3", "77.8%", "77.8%", "0.759", "1/9"],
             ["C", "100.0%", "100.0%", "0.800", "3/9"],
         ]),
        ("Glass-only", "9 messages; 27 predictions per strategy",
         [
             ["Strategy", "Claim acc.", "Urgency acc.", "Extract. F1", "Exact records"],
             ["A", "100.0%", "100.0%", "0.882", "9/27"],
             ["B1", "100.0%", "100.0%", "0.902", "12/27"],
             ["B2", "100.0%", "100.0%", "0.889", "8/27"],
             ["B3", "100.0%", "100.0%", "0.914", "12/27"],
             ["C", "100.0%", "100.0%", "0.875", "9/27"],
         ]),
        ("Multi-incident", "1 message; 3 predictions per strategy",
         [
             ["Strategy", "Claim acc.", "Urgency acc.", "Extract. F1", "Exact records"],
             ["A", "100.0%", "100.0%", "0.821", "0/3"],
             ["B1", "100.0%", "100.0%", "0.833", "0/3"],
             ["B2", "100.0%", "100.0%", "0.769", "0/3"],
             ["B3", "100.0%", "100.0%", "0.923", "0/3"],
             ["C", "100.0%", "100.0%", "0.769", "0/3"],
         ]),
    ]

    for title, sample, data in subgroup_data:
        story += [
            P(title, h2),
            P(sample, small),
            table(data, [3.0 * cm, 2.4 * cm, 2.4 * cm, 2.5 * cm, 2.5 * cm], fs=7),
            Spacer(1, 4),
        ]

    story += [
        P("<b>Ambiguous theft.</b> A and C classify all three messages correctly. B1 and B2 repeatedly classify TEST029 as not_a_claim and assign low rather than medium urgency. B3 improves on B1/B2 but still makes errors. A and C tie on extraction F1 and exact records; C is slower on average.", body),
        P("<b>Glass-only.</b> All strategies classify claim type and urgency correctly. B3 has the highest extraction micro F1 (0.914), narrowly ahead of B1 (0.902). Increasing the number of examples does not produce a monotonic improvement: B2 is below B1 on extraction F1 and exact records.", body),
        P("<b>Multi-incident.</b> All strategies classify the single message correctly, but none achieves a fully exact record. B3 has the highest extraction F1 (0.923) in this slice. For TEST058, several strategies carry the first incident's Rs 12,000 windshield estimate into the second incident, while B1 correctly returns null for that amount. This is one illustrative case, not evidence of general strategy superiority.", body),
        P("<b>Interpretation limits.</b> The multi-incident subgroup contains only one unique message, so its results are descriptive only. The other subgroups are also small. Classification macro precision, recall and F1 calculated over the full global class list are not useful for these slices because many classes are absent; subgroup claim-type and urgency accuracy are reported instead. Extraction micro F1 aggregates the extraction fields and should be read alongside exact-record counts.", small),
    ]

    # ---------------- Page 6: error analysis ----------------
    story += [PageBreak(), P("7. Error analysis", h2),
              P("The full sheet is <i>outputs/final_benchmark/analysis/error_analysis.xlsx</i>. It lists every distinct field failure per strategy with its cause, severity, "
                "mitigation, a check of the extracted value against the source message, and whether the gold label is flagged. A: 31 rows, B1: 24, B2: 24, B3: 14 (all that exist), C: 21. "
                "No API or format failure occurred, and every extracted policy number, registration, date and amount appears in its source message, so there are no invented values.")]
    er = [["Category", "A", "B1", "B2", "B3", "C", "Cause and mitigation"],
          ["Urgency set to high for a destroyed vehicle", "14", "6", "0", "2", "6", "Model treats 'destroyed' as unsafe. Guide: high only for injury or a stated unsafe or undrivable vehicle. Keep the rule, add a total_loss example with medium."],
          ["Policy or registration null on not_a_claim", "5", "12", "13", "3", "7", "Prompt line says every extraction field is null for not_a_claim. Reword to 'extract values the message states'."],
          ["Third-party signal missed", "5", "2", "4", "0", "2", "Injured passenger or damaged other car not treated as third party. Add examples."],
          ["Explicit denial or confirmation missed", "3", "1", "1", "3", "5", "'Nobody injured', 'no FIR yet' returned as null. Add denial examples, including Hinglish."],
          ["Unsupported false", "0", "1", "3", "3", "0", "'Only glass is damaged' read as no injury. Reinforce 'false only if explicitly denied'."],
          ["Wrong incident used", "1", "0", "1", "1", "1", "TEST058: Rs 12,000 windshield estimate from the withdrawn first incident. Add a multi-incident example."],
          ["Suspected theft as not_a_claim", "0", "2", "2", "2", "0", "TEST029. Guide marks it ambiguous. Add a missing-vehicle example."],
          ["Claim type confused (theft or total loss)", "2", "0", "0", "0", "0", "A only: parts stolen read as own_damage; 'beyond repair' missed."]]
    er = [[Paragraph(str(x), cell) for x in row] for row in er]
    story += [table(er, [4.0 * cm, 0.8 * cm, 0.8 * cm, 0.8 * cm, 0.8 * cm, 0.8 * cm, 9.6 * cm], fs=7),
              Spacer(1, 4),
              P("<b>Model mistakes against label and metric issues.</b> The location mismatch (30 records) is a label convention, not a model error; it is listed separately and excluded from the failure count. "
                "TEST029 and TEST051 are borderline gold labels. Urgency over-escalation on total loss is a model-versus-guide disagreement that a reader could defend either way, so it is rated low severity. "
                "Severity counts across strategies: High 3 rows (all TEST029 claim_type in B1, B2, B3), Medium 22, Low 89. "
                "The most consequential real defect is the not_a_claim extraction rule, which comes from the prompt wording and is cheap to fix.")]

    # ---------------- Page 6: recommendation ----------------
    story += [PageBreak(), P("8. Recommendation", h2),
              P("<b>Use B3 (few-shot, k=8).</b> It has the highest claim_type accuracy (98.9%), the best strategy-level record exact match (44.4% strict, 81.7% location normalised), the best extraction F1 (0.914) "
                "and the fewest failures (14 rows, 13 distinct records). Injury and high-urgency recall are 100%. It costs about $0.19 per 1,000 claims, 1.8 times zero-shot, and uses about 2,057 prompt tokens per call. "
                "At that price the extra accuracy is worth paying for in a triage setting."),
              P("<b>When to choose otherwise.</b> B2 (k=5) has the best urgency accuracy (98.3%) at $0.157 per 1,000 claims, but loses policy-number recall on not_a_claim messages. "
                "A is the cheapest ($0.104) and still never misses an injury, but over-escalates urgency and misses third-party liability, so it needs human review of every high-urgency item. "
                "C is not recommended: it costs about as much as B3, is slowest, and is less accurate."),
              P("<b>Before deployment.</b> (1) Reword the not_a_claim instruction and rerun B3 on the 60 test records. (2) Add examples for passenger injury, missing vehicles and multi-incident messages. "
                "(3) Route messages the model labels theft or not_a_claim with a missing vehicle to a person. (4) Normalise the gold location convention before comparing location scores."),
              P("9. Limitations", h2),
              P("1. 60 test records is small; one record moves a score by 1.7 points per run, so close results are not conclusive. "
                "2. The messages are synthetic and short. Real claims are longer, noisier and include attachments. "
                "3. One annotator wrote the gold labels. The independent review of 20 labels is prepared (worksheet and comparison script in the repository) but has not been completed, so inter-annotator agreement is not reported. "
                "4. A single model (gpt-oss-20b), a single temperature and low reasoning effort were tested; results may differ for other models or settings. "
                "5. Latency was measured on a shared endpoint and is noisy. Cost is projected from list prices on 2026-11-13 and may change. "
                "6. Few-shot examples were added in a fixed order, so the effect of each example cannot be separated from the effect of count. "
                "7. Prompt D (optional) was not implemented. "
                "8. Several gold labels (TEST029, TEST047, TEST051, urgency for destroyed vehicles) are judgement calls under the guide; different rules would shift the scores."),
              P("Reproduce: <i>python analysis.py --input-cost-per-million 0.075 --output-cost-per-million 0.30</i>, then <i>python scripts/build_error_analysis.py</i> and <i>python scripts/build_report.py</i>. No step calls the API.", small)]

    doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                            title="ClaimFast Evaluation Report", author="Ryane Bose")

    def footer(canvas, d):
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        canvas.drawString(1.7 * cm, 0.8 * cm, "ClaimFast evaluation report")
        canvas.drawRightString(A4[0] - 1.7 * cm, 0.8 * cm, f"Page {d.page}")

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", PDF)


if __name__ == "__main__":
    main()
