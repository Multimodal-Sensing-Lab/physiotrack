| Evaluation area   | Metric                       |     Value | Unit       |
|:------------------|:-----------------------------|----------:|:-----------|
| Dataset           | Test videos                  |  212      | videos     |
| Dataset           | Person sequences             |  687      | sequences  |
| Eye Openness      | Availability                 |   78.64   | %          |
| Eye Openness      | ROC AUC                      |    0.6631 |            |
| Eye Openness      | Blink-frame median openness  |    0.3057 |            |
| Eye Openness      | Non-blink median openness    |    0.476  |            |
| Blink Detection   | Precision                    |   30.4    | %          |
| Blink Detection   | Recall                       |   24.15   | %          |
| Blink Detection   | F1-score                     |   26.92   | %          |
| Blink Detection   | Mean matched temporal IoU    |    0.6811 |            |
| Blink Events      | Ground-truth blinks          | 7564      | events     |
| Blink Events      | Predicted blinks             | 6010      | events     |
| Blink Events      | Blink-count MAE per sequence |    6.1194 | blinks     |
| Blink Events      | Blink-rate MAE               |   10.7998 | blinks/min |
| Blink Events      | Mean blink-duration error    |    0.1018 | s          |
| Runtime           | Processing time              |   96.48   | min        |