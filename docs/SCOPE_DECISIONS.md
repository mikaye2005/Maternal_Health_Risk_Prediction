# Final Scope Decisions

This note records two transparent changes from the [preserved original proposal](references/Original_MamaCare_Project_Proposal.pdf). They prevent the final report from claiming work that was not delivered.

## Neural-network comparison

The proposal described Keras networks with dropout and batch normalisation. The final capstone instead compares two scikit-learn feed-forward multilayer perceptrons: a shallow `(32,)` network and a deeper `(64, 32, 16)` network. Both use ReLU activation, L2 regularisation and early stopping, and the project exports their training-loss and internal-validation curves.

These models provide the required honest shallow-versus-deep comparison, but they are an approximation of the proposed Keras experiment. They do **not** implement dropout or batch normalisation. This narrower scope avoids adding a large deep-learning runtime to a small tabular dataset where extra depth did not improve validation Weighted F1.

## Clinical-threshold features

The proposal also mentioned blood-pressure risk flags and glucose-risk bands. They were excluded from the final feature set because the project does not have a clinically reviewed threshold specification for this dataset or evidence that a single threshold system transfers safely from rural Bangladesh to Kenya.

The retained engineered variables are arithmetic or descriptive transformations only:

- `PulsePressure = SystolicBP - DiastolicBP`
- `MeanArterialPressure = (SystolicBP + 2*DiastolicBP)/3`
- `AgeBand` for `<=19`, `20-34`, `35-49` and `>=50`

The interface labels observed minima and maxima as dataset bounds, not definitions of safe, normal or abnormal health.
