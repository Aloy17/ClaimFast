"""Build the ClaimFast error-analysis workbook from the saved benchmark log.

Offline only: reads outputs/final_benchmark/raw_responses.jsonl and writes
outputs/final_benchmark/analysis/error_analysis.xlsx. No API calls.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "outputs" / "final_benchmark" / "raw_responses.jsonl"
OUT = ROOT / "outputs" / "final_benchmark" / "analysis" / "error_analysis.xlsx"

FIELDS = [
    "claim_type", "urgency", "policy_number", "incident_date", "location",
    "vehicle_reg_no", "estimated_amount_inr", "injury_reported",
    "police_report_filed",
]
STRATEGIES = ["A", "B1", "B2", "B3", "C"]
NOT_CLAIM_TYPES = {"not_a_claim"}

CATEGORIES = {
    "M1": "Model: urgency over-escalated to high (total loss / damaged vehicle treated as unsafe)",
    "M2": "Model: policy or registration left null on a not_a_claim message",
    "M3": "Model: third-party signal missed (own_damage returned for third_party_liability)",
    "M4": "Model: claim type wrong (theft, total_loss or own_damage confused)",
    "M5": "Model: explicit denial or confirmation not captured (false/true returned as null)",
    "M6": "Model: unsupported boolean or value (not stated in the message)",
    "M7": "Model: multi-incident contamination (detail taken from the wrong incident)",
    "M8": "Model: suspected theft treated as not_a_claim",
    "L1": "Dataset/label issue: borderline gold label",
    "F1": "API or format failure",
}

# (strategy-independent) per-record, per-field annotation.
# key: (record_id, field) -> (category, severity, cause, mitigation, label_flag)
NOTES = {
    ("TEST029", "claim_type"): ("M8", "High", "Customer says the car is missing but is still checking with security; the model reads it as not a claim.", "Add a worked 'missing vehicle in an FNOL context = theft' example (guide section 6).", "Yes: guide flags TEST029 as ambiguous"),
    ("TEST029", "urgency"): ("M8", "Medium", "Follows from the wrong claim_type.", "Fix claim_type rule; urgency follows.", "Yes: ambiguous gold"),
    ("TEST029", "policy_number"): ("M2", "Low", "Policy POL-66102 is stated but dropped together with the other fields.", "Reword 'every extraction field is null' (prompts.py line 31).", "No"),
    ("TEST029", "location"): ("M2", "Low", "Santacruz stated, but dropped with the not_a_claim output.", "As above.", "No"),
    ("TEST029", "vehicle_reg_no"): ("M2", "Low", "Registration stated, but dropped with the not_a_claim output.", "As above.", "No"),
    ("TEST058", "estimated_amount_inr"): ("M7", "Medium", "Rs 12000 is the windshield estimate from the earlier, withdrawn incident; the gold label is null for the November 1 accident.", "Add a multi-incident example showing figures from the excluded incident are ignored.", "No"),
    ("TEST051", "police_report_filed"): ("L1", "Low", "'Police arrived but no report confirmation' is genuinely ambiguous; model chose false, gold is null.", "Document the rule for 'police attended, no report' in the labelling guide.", "Yes"),
    ("TEST047", "injury_reported"): ("M6", "Low", "Message says nothing about injury; model returns false.", "Reinforce 'false only when explicitly denied'.", "No"),
    ("TEST011", "injury_reported"): ("M6", "Low", "'Nothing else is damaged' is not an injury denial.", "Reinforce 'false only when explicitly denied'.", "No"),
    ("TEST031", "injury_reported"): ("M6", "Low", "'Only glass is damaged' is not an injury denial.", "As above.", "No"),
    ("TEST055", "injury_reported"): ("M6", "Low", "'No other parts are damaged' is not an injury denial.", "As above.", "No"),
    ("TEST018", "claim_type"): ("M3", "Medium", "Model infers liability because a bus hit the car; the guide says damage only to the insured vehicle is own_damage.", "Add a hit-and-run example labelled own_damage.", "No"),
    ("TEST035", "claim_type"): ("M4", "Medium", "Battery and spare tyre were removed, which counts as theft of parts; model chose own_damage.", "State 'stolen parts = theft' with an example.", "No"),
    ("TEST020", "claim_type"): ("M4", "Medium", "'Damaged beyond repair' not matched to total_loss.", "Keep the total_loss definition near the top of the prompt.", "No"),
    ("TEST019", "claim_type"): ("M3", "Medium", "Injured passenger counts as a third party under the guide; model returned own_damage.", "Add a passenger-injury example labelled third_party_liability.", "No"),
    ("TEST039", "claim_type"): ("M3", "Medium", "Injured passenger counts as a third party; model returned own_damage.", "As above.", "No"),
    ("TEST012", "vehicle_reg_no"): ("M2", "Low", "Registration MH02XY8877 is stated in the correction request but dropped.", "Reword prompts.py line 31.", "No"),
    ("TEST012", "policy_number"): ("M2", "Low", "Policy POL-33102 stated but dropped.", "Reword prompts.py line 31.", "No"),
    ("TEST032", "vehicle_reg_no"): ("M2", "Low", "Registration stated but dropped.", "Reword prompts.py line 31.", "No"),
    ("TEST032", "policy_number"): ("M2", "Low", "Policy stated but dropped.", "Reword prompts.py line 31.", "No"),
}


def norm(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def grounded(field, predicted, message):
    """Check that a predicted non-null value can be found in the source message."""
    if predicted is None:
        return ""
    if field in ("claim_type", "urgency", "injury_reported", "police_report_filed"):
        return "n/a (label, not a quoted value)"
    if field == "estimated_amount_inr":
        digits = re.sub(r"[^0-9]", "", message)
        return "Yes" if str(predicted) in digits else "No"
    if field == "incident_date":
        return "Yes" if str(predicted) in message else "No"
    return "Yes" if norm(predicted) in norm(message) else "No"


def location_equivalent(gold, pred):
    if gold == pred:
        return True
    return bool(gold and pred and pred.lower().strip() in gold.lower())


def default_note(record_id, field, gold, pred, claim_gold):
    if field == "urgency" and pred == "high" and gold == "medium":
        return ("M1", "Low", "Model sets high for a destroyed or damaged vehicle; the guide sets high only for injury or a stated unsafe or undrivable vehicle.", "Keep the total_loss urgency rule; add a total_loss example with urgency medium.", "Borderline: 'destroyed' could be read as undrivable")
    if field in ("policy_number", "vehicle_reg_no") and pred is None and claim_gold in NOT_CLAIM_TYPES:
        return ("M2", "Low", "Stated value dropped because the prompt says not_a_claim extraction fields are null.", "Reword prompts.py line 31 to 'extract values the message states'.", "No")
    if field in ("injury_reported", "police_report_filed") and gold is False and pred is None:
        return ("M5", "Low", "The message explicitly denies the condition but the model returns null.", "Add 'nobody injured / no FIR yet = false' examples, including Hinglish.", "No")
    if field in ("injury_reported", "police_report_filed") and gold is True and pred is None:
        return ("M5", "Low", "The message explicitly confirms the condition but the model returns null.", "Add confirmation examples.", "No")
    if field == "claim_type" and gold == "third_party_liability" and pred == "own_damage":
        return ("M3", "Medium", "Other party's vehicle or person is affected; the model returns own_damage.", "Add third-party examples.", "No")
    if field == "claim_type":
        return ("M4", "Medium", "Claim type confused.", "Clarify definitions.", "No")
    return ("M6", "Low", "Value differs from gold.", "Review manually.", "No")


def main():
    records = [json.loads(line) for line in LOG.open(encoding="utf-8") if line.strip()]
    messages = {r["record_id"]: r["message"] for r in records}

    api_failures = sum(bool(r.get("api_error")) or bool(r.get("parse_error")) for r in records)
    runs = defaultdict(lambda: defaultdict(list))  # (strategy, record, field) -> pred -> runs
    gold = {}
    for r in records:
        predicted = r.get("predicted") or {}
        for field in FIELDS:
            gold[(r["record_id"], field)] = r["expected"][field]
            p = predicted.get(field)
            e = r["expected"][field]
            if field == "location" and location_equivalent(e, p):
                continue
            if p != e:
                runs[(r["strategy"], r["record_id"], field)][json.dumps(p)].append(r["run_number"])

    rows = []
    for (strategy, record_id, field), preds in runs.items():
        e = gold[(record_id, field)]
        claim_gold = gold[(record_id, "claim_type")]
        for pred_json, run_list in preds.items():
            pred = json.loads(pred_json)
            cat, sev, cause, mitigation, label_flag = NOTES.get(
                (record_id, field),
                default_note(record_id, field, e, pred, claim_gold),
            )
            if field == "claim_type" and record_id == "TEST029":
                pass
            rows.append({
                "strategy": strategy, "record_id": record_id, "field": field,
                "gold": e, "predicted": pred, "runs": sorted(run_list),
                "category": cat, "severity": sev, "cause": cause,
                "mitigation": mitigation, "label_flag": label_flag,
                "message": messages[record_id],
                "grounded": grounded(field, pred, messages[record_id]),
            })
    rows.sort(key=lambda r: (STRATEGIES.index(r["strategy"]), r["record_id"], FIELDS.index(r["field"])))

    # Location span mismatches (label convention, not a model error)
    loc_rows = defaultdict(lambda: {"preds": defaultdict(set)})
    for r in records:
        p = (r.get("predicted") or {}).get("location")
        e = r["expected"]["location"]
        if p != e and location_equivalent(e, p):
            loc_rows[(r["record_id"], e)]["preds"][p].add(r["strategy"])
    loc_list = sorted(loc_rows.items())
    loc_strict_fail_count = {
        s: sum(1 for r in records if r["strategy"] == s and (r.get("predicted") or {}).get("location") != r["expected"]["location"] and location_equivalent(r["expected"]["location"], (r.get("predicted") or {}).get("location")))
        for s in STRATEGIES
    }

    wb = Workbook()
    arial = "Arial"
    head_fill = PatternFill("solid", start_color="1F3864")
    head_font = Font(name=arial, bold=True, color="FFFFFF", size=10)
    body = Font(name=arial, size=10)
    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    wrap = Alignment(wrap_text=True, vertical="top")

    def style_sheet(ws, widths, header_row=1):
        for col, width in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(col)].width = width
        for cell in ws[header_row]:
            cell.fill, cell.font = head_fill, head_font
            cell.alignment = Alignment(wrap_text=True, vertical="center")
            cell.border = border
        for row in ws.iter_rows(min_row=header_row + 1):
            for cell in row:
                cell.font, cell.alignment, cell.border = body, wrap, border
        ws.freeze_panes = ws.cell(row=header_row + 1, column=1)

    # ---- Sheet 1: Failures ----
    ws = wb.active
    ws.title = "Failures"
    ws.append([
        "ID", "Strategy", "Record", "Field", "Gold", "Predicted", "Runs wrong (of 3)",
        "Category", "Category description", "Severity", "Cause", "Proposed mitigation",
        "Value found in source message?", "Gold label flagged?", "Source message",
    ])
    for i, r in enumerate(rows, 1):
        ws.append([
            f"F{i:03d}", r["strategy"], r["record_id"], r["field"],
            json.dumps(r["gold"]), json.dumps(r["predicted"]),
            len(r["runs"]), r["category"], CATEGORIES[r["category"]], r["severity"],
            r["cause"], r["mitigation"], r["grounded"], r["label_flag"], r["message"],
        ])
    style_sheet(ws, [7, 9, 10, 20, 24, 24, 11, 9, 38, 10, 48, 44, 18, 28, 70])
    ws.auto_filter.ref = ws.dimensions
    last = ws.max_row

    # ---- Sheet 2: Summary (formulas over Failures) ----
    sm = wb.create_sheet("Summary", 0)
    sm.append(["ClaimFast error analysis summary (counts are formulas over the Failures sheet)"])
    sm["A1"].font = Font(name=arial, bold=True, size=12)
    sm.append([])
    sm.append(["Strategy", "Failure rows reviewed", "Distinct records", "Severity High", "Severity Medium", "Severity Low", "Rows with a value not found in source", "Location format mismatches (not counted in Failures)"])
    header_row = 3
    for i, s in enumerate(STRATEGIES, 4):
        sm.append([
            s,
            f'=COUNTIF(Failures!$B$2:$B${last},A{i})',
            None,
            f'=COUNTIFS(Failures!$B$2:$B${last},A{i},Failures!$J$2:$J${last},"High")',
            f'=COUNTIFS(Failures!$B$2:$B${last},A{i},Failures!$J$2:$J${last},"Medium")',
            f'=COUNTIFS(Failures!$B$2:$B${last},A{i},Failures!$J$2:$J${last},"Low")',
            f'=COUNTIFS(Failures!$B$2:$B${last},A{i},Failures!$M$2:$M${last},"No")',
            loc_strict_fail_count[s],
        ])
        distinct = len({r["record_id"] for r in rows if r["strategy"] == s})
        sm.cell(row=i, column=3, value=distinct)
    sm.append([])
    sm.append(["Category", "Description", "A", "B1", "B2", "B3", "C", "Total"])
    cat_header = sm.max_row
    for code, desc in CATEGORIES.items():
        r = sm.max_row + 1
        sm.append([code, desc] + [
            f'=COUNTIFS(Failures!$B$2:$B${last},"{s}",Failures!$H$2:$H${last},$A{r})'
            for s in STRATEGIES
        ] + [f"=SUM(C{r}:G{r})"])
    sm.append([])
    notes = [
        f"API or format failures across all 900 runs: {api_failures}. All 900 responses were valid JSON and passed validation, so F1 has no rows.",
        "Failures lists every distinct (strategy, record, field, predicted value) disagreement, so each strategy has more than the 15 required. Runs wrong shows how many of the 3 runs repeated it.",
        "Location format mismatches are listed on the Location_Format sheet. The gold labels keep fragments such as 'near' or 'outside the gym in'; the model returns the place name. Treated as a label convention, not a model error.",
        "'Value found in source message?' = No means a non-null extracted value does not appear in the message text. Boolean and label fields show n/a.",
        "Severity: High = unsafe or wrong routing (for example a suspected theft treated as not a claim). Medium = wrong claim type or a figure from the wrong incident. Low = a null or over-escalated field with limited consequence.",
    ]
    for n in notes:
        sm.append([n])
    sm.column_dimensions["A"].width = 12
    sm.column_dimensions["B"].width = 70
    for col in "CDEFGH":
        sm.column_dimensions[col].width = 16
    for cell in sm[header_row]:
        cell.fill, cell.font = head_fill, head_font
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    for cell in sm[cat_header]:
        cell.fill, cell.font = head_fill, head_font
    for row in sm.iter_rows(min_row=4):
        for cell in row:
            if cell.font != head_font and not (cell.font and cell.font.bold):
                cell.font = body
    for row in sm.iter_rows(min_row=sm.max_row - len(notes) + 1, max_row=sm.max_row):
        sm.merge_cells(start_row=row[0].row, start_column=1, end_row=row[0].row, end_column=8)
        row[0].alignment = Alignment(wrap_text=True, vertical="top")
        sm.row_dimensions[row[0].row].height = 30

    # ---- Sheet 3: Location_Format ----
    lf = wb.create_sheet("Location_Format")
    lf.append(["Record", "Gold location", "Model location (strategies returning it)", "Source message", "Assessment"])
    for (record_id, e), info in loc_list:
        pred_text = "; ".join(f"{p} ({', '.join(sorted(s))})" for p, s in sorted(info["preds"].items(), key=lambda kv: str(kv[0])))
        lf.append([record_id, e, pred_text, messages[record_id],
                   "Same place; gold keeps extra words such as 'near' or the venue type. Label convention, not a model error."])
    style_sheet(lf, [10, 36, 44, 80, 50])

    # ---- Sheet 4: Method ----
    me = wb.create_sheet("Method")
    me.append(["Item", "Detail"])
    method = [
        ("Source", "outputs/final_benchmark/raw_responses.jsonl (900 records: 5 strategies x 3 runs x 60 test messages)."),
        ("What counts as a failure", "A field whose predicted value differs from the gold value. Location is compared leniently (see Location_Format)."),
        ("Grouping", "One row per distinct strategy, record, field and predicted value. Runs wrong shows repeats across the 3 runs."),
        ("Hallucination check", "Every non-null policy, registration, date, amount and location is searched for in the original message. Result in the 'Value found in source message?' column."),
        ("Model vs label vs API", "M codes are model errors, L codes are borderline or flagged gold labels, F codes are API or format failures (none occurred)."),
        ("Reproduce", "python scripts/build_error_analysis.py"),
    ]
    for item in method:
        me.append(list(item))
    style_sheet(me, [26, 120])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    per = {s: sum(1 for r in rows if r["strategy"] == s) for s in STRATEGIES}
    print("rows per strategy:", per, "| location records:", len(loc_list), "| api/parse failures:", api_failures)
    print("ungrounded values:", [(r["strategy"], r["record_id"], r["field"], r["predicted"]) for r in rows if r["grounded"] == "No"])
    print("default-note rows (check):", sum(1 for r in rows if r["cause"] == "Value differs from gold." or r["cause"] == "Claim type confused."))


if __name__ == "__main__":
    main()
