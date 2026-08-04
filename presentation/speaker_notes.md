# MamaCare Presentation Speaker Notes

## Slide 1 - Problem and Context
Good morning. My project is MamaCare, a maternal-health risk classification capstone. The system uses six routine measurements and classifies a record as Low Risk, Mid Risk or High Risk. The target is RiskLevel, and the primary metric was fixed as Weighted F1 before final testing. The dataset is from UCI and is documented as coming from rural Bangladesh, so this is not Kenyan clinical validation. The project is an academic demonstration, not a diagnosis or treatment system.

## Slide 2 - Data and Approach
The dataset contains 1,014 downloaded rows, six inputs and one target. A major issue is duplication: 562 rows are exact duplicates. More importantly, 35 identical measurement signatures have conflicting labels, covering 215 rows. To prevent inflated performance, I grouped identical signatures so they could not cross train, validation and test sets. I compared a majority baseline, traditional models, XGBoost, two neural networks and a soft-voting ensemble. Models were selected using validation Weighted F1.

## Slide 3 - Results
The selected model was ShallowMLP. On the untouched test set, Weighted F1 was 0.562, Macro F1 was 0.574, and High Risk recall was 0.818. The model clearly beat the majority baseline. However, Mid Risk recall was only 0.149. I am reporting that weakness directly because a model should not be presented as successful only through its strongest metric. The confusion matrix shows that many Mid Risk records were assigned to Low Risk.

## Slide 4 - Limitations and Responsible Use
For fairness, I used Equal Opportunity for High Risk identification across age groups. High Risk recall varied substantially, with a gap of 0.580 between the best and worst groups. The youngest group performed worst. The dataset is small, duplicated, narrow and not locally validated. The worst plausible harm is false reassurance after a missed High Risk case. A real study would require local data, human review, a redress mechanism, monitoring and clinical governance.

## Slide 5 - Conclusion and Demonstration
MamaCare is complete as a portfolio artefact: it has a backend pipeline, an executed notebook, tests, reports, a Responsible AI statement, slides and a polished frontend. The strongest technical contribution is the measurement-signature split because it prevents duplicate leakage. The selected model found useful signal, but weak Mid Risk performance and age-group differences prevent clinical claims. During the demo, I will load a demonstration case, submit the assessment, explain the three model scores and local influences, and then show the model evidence and responsible-use tabs.

## Final line
MamaCare demonstrates that good data science is not only about a high score. It is also about honest evaluation, clear limitations and responsible communication of who may be affected by model errors.
