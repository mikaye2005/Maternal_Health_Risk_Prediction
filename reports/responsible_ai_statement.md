# Responsible AI Statement

## Who does this system affect?
MamaCare affects several groups. The first group is the learner, assessor or health worker who views a model output and may be influenced by the displayed category. The second group is pregnant people whose measurements are classified, even when they never see the model directly. The third group includes younger mothers and other populations that may be underrepresented, inconsistently labelled or absent from the training data. Families can also be affected because the wording of a result may create reassurance or anxiety. The public dataset is documented as coming from maternal-health facilities in rural Bangladesh, while the demonstration may be viewed in Kenya. The represented population and the likely audience are therefore not the same.

## Worst plausible outcome
The most serious plausible outcome is false reassurance. An actual High Risk record could be assigned Low Risk or Mid Risk and a user could wrongly delay professional assessment. In the untouched test set, the model identified 81.8% of High Risk records, which also means some High Risk cases were missed. The application repeatedly states that it is not a diagnosis, treatment tool, triage system or emergency assessment. These warnings reduce misuse risk but cannot make an unvalidated model clinically safe.

## Who bears the cost of errors?
The main cost is borne by the pregnant person whose record is misclassified, not by the model developer. A false negative can delay assessment, while a false positive can create anxiety and unnecessary referral. Mid Risk recall is only 14.9%, showing that the model is especially unreliable for the middle class. The dataset also contains 35 identical measurement signatures with different labels. Records with conflicting signatures had substantially higher error rates, demonstrating that some failures come from the available data rather than model choice alone.

## Redress mechanism
There is no formal redress mechanism because MamaCare is an academic capstone prototype. Any future real-world study would need human review of every output, a visible way to challenge or correct a result, documentation of overrides and a clear statement that the model is only one source of information. A correction request should be acknowledged within one working day and receive measurement correction plus human reassessment within two working days. This route must never replace urgent care. No person should be denied care, reassurance or referral solely because of this output.

## Monitoring plan
A deployed research version would require monthly monitoring of Weighted F1, Macro F1, High Risk recall, Mid Risk recall, missing-input performance, data drift and subgroup recall. The research system should be withdrawn from use if High Risk recall is below 0.75 in a review containing at least 50 actual High Risk records, if an age group with at least 20 actual High Risk records performs more than 0.10 below overall recall, or if material input drift is confirmed. Smaller samples must be labelled inconclusive rather than safe. Monitoring should also record human overrides and investigate confidently wrong predictions.

## Fairness criterion and findings
The selected fairness criterion is Equal Opportunity for High Risk identification across age groups. It is appropriate because the primary safety question is whether actual High Risk records are correctly identified at similar rates for different age groups. In the current test split, High Risk recall ranges from 37.5% to 95.5%, a gap of 58.0%. The youngest group performs worst. Sample sizes are limited, but the gap is too large to claim equal performance.

## Datasheet summary
The dataset contains six routine measurements and a three-class target label. It does not include Kenyan clinical validation data, pregnancy stage, county, facility context, medical history, socioeconomic conditions or sufficient demographic information for a complete fairness audit. The project is therefore appropriate only as a transparent educational demonstration.
