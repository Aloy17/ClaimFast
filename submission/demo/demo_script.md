# ClaimFast three-minute demo script

Setup: activate the virtual environment, confirm `.env` contains your key, and open a terminal with a large font.

## 1. Introduce (20 seconds)
Say what ClaimFast does: reads a motor claim message and returns claim type, urgency and seven extracted fields as JSON.

## 2. One message through A, B3 and C (90 seconds)
```bash
python scripts/demo_strategies.py --message "Kal shaam meri car MH02ST3344 ko Navi Mumbai mein ek bike ne side se hit kiya. Bike rider ko chot nahi lagi, lekin uski bike damage hui. Policy POL-55109. Estimate 32,000 rupees. Police complaint file ki hai."
```
Point out: claim_type for each strategy (gold is third_party_liability), the null incident date (relative date "kal"), latency and tokens per strategy.

## 3. An injury case (30 seconds)
```bash
python scripts/demo_strategies.py --message "My car MH01EF9900 skidded on a wet road near Borivali on 2026-10-24 and hit a wall. My friend in the passenger seat has a shoulder injury. The car is still drivable. Policy POL-11890. Repair cost Rs 85000. Police report not filed."
```
Point out: urgency high, injury_reported true, third_party_liability for the injured passenger.

## 4. Results (40 seconds)
Show the results table and the claim_type confusion matrices in `submission/ClaimFast_Evaluation_Report.pdf`. State the recommendation (B3) and that B1, B2 and the 900-request benchmark are covered in the report.

Screenshots alternative: capture the terminal after steps 2 and 3, and pages 2 and 4 of the report.
