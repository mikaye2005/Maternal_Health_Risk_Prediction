# Dataset Selection Note

Two classification datasets were assessed before the final capstone dataset was confirmed.

| Candidate | Source and reuse terms | Size and variables | Target | Suitability | Limitations | Decision |
|---|---|---|---|---|---|---|
| UCI Maternal Health Risk Dataset | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/863/maternal+health+risk), DOI `10.24432/C5DP5D`, CC BY 4.0 | UCI metadata: 1,013 instances and six predictors. The downloaded CSV preserved in this project has 1,014 rows, six measurements and one target. | `RiskLevel`: Low, Mid or High Risk | Matches the accepted maternal-health classification proposal; has a clear target, more than 500 records, six meaningful predictors, a data dictionary and public-sharing permission. | Small, heavily duplicated, contains contradictory labels, has limited demographic/context variables and is not Kenyan clinical data. | **Selected.** It supports a complete, reproducible classification capstone while making the data limitations visible. |
| Zindi Financial Inclusion in Africa | [Zindi learning competition](https://zindi.africa/competitions/financial-inclusion-in-africa/data); competition terms restrict redistribution of the supplied data | Training file: 23,524 rows, 12 predictors plus the target; demographic and financial-access fields from Kenya, Rwanda, Tanzania and Uganda | `bank_account`: Yes or No | Larger African dataset with richer demographic subgroup variables and a clear binary target. | Does not match the accepted maternal-health proposal; competition data cannot be republished in a public repository, and changing domains would discard the approved MamaCare scope. | **Not selected.** Stronger sample size and subgroup coverage did not outweigh the scope and redistribution constraints. |

## Final rationale

The UCI dataset was retained because it aligns with the approved MamaCare objective and permits public attribution and reuse. Selection does not imply clinical adequacy. The final workflow therefore uses measurement-signature-separated evaluation, reports duplicates and contradictory labels, disaggregates High Risk recall by age group and limits the interface to an academic demonstration.
