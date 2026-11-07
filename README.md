# Steel Plate Fault Classification

Live app: https://plate-defect-triage.streamlit.app/

## a. Problem statement

The goal is to predict the type of surface fault on a steel plate (7 types) from measurements such as the fault's size, position, outline and brightness. I trained 5 classification models on the same data, compared them, and built a Streamlit app that evaluates the saved models on an uploaded test CSV.

## b. Dataset description

Steel Plates Faults dataset from the UCI Machine Learning Repository (Buscema, Terzi and Tastle, 2010). DOI: 10.24432/C5J88N

- 1941 rows, 27 numeric features, no missing values
- Target: `fault_type` with 7 classes
- Train/test split: 80/20 stratified, giving 1552 train and 389 test rows (test rows are saved in `test_data.csv`)

The features describe each fault's position and size (X/Y min and max, pixel area, perimeters), its brightness (luminosity sum, min, max and index), the plate (steel type, thickness, conveyor length), and some shape indices (edges, empty, square, orientation, log of area, etc.).

Class counts:

| Fault | Count |
|---|---|
| Other_Faults | 673 |
| Bumps | 402 |
| K_Scatch | 391 |
| Z_Scratch | 190 |
| Pastry | 158 |
| Stains | 72 |
| Dirtiness | 55 |

The classes are imbalanced, so I picked hyperparameters using MCC and used macro averaging for precision, recall and F1. A few columns are very skewed, so I applied a Yeo-Johnson transform for Logistic Regression, Naive Bayes and SVM, and robust scaling for KNN. `TypeOfSteel_A400` (just 1 - A300) and `Y_Maximum` (almost the same as `Y_Minimum`) are dropped. I also tried 7 extra features, such as elongation, edge per pixel and mean brightness, and let grid search decide for each model whether to use them.

## c. GitHub Repository Link

https://github.com/synF5/plate-defect-triage

```
app.py              streamlit app
requirements.txt
test_data.csv       test set used in the app
lab_run.ipynb       notebook used on the BITS virtual lab
model/              saved models (.joblib) + scorecard.json
plate_triage/       code for loading data, features, models, metrics and training
data/raw/           original UCI files
```

The preprocessing is part of each saved pipeline, so nothing else needs to be loaded.

To run locally:

```
pip install -r requirements.txt
python -m plate_triage.fit_all     # retrains and saves the models (optional)
streamlit run app.py
```

## d. Models used

Each model was tuned with grid search (5-fold stratified CV) on the training data. The results below are on the 389 test rows.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.735 | 0.940 | 0.727 | 0.744 | 0.735 | 0.662 |
| Decision Tree | 0.746 | 0.872 | 0.783 | 0.769 | 0.774 | 0.671 |
| KNN | 0.733 | 0.951 | 0.782 | 0.764 | 0.770 | 0.654 |
| Naive Bayes | 0.594 | 0.890 | 0.629 | 0.661 | 0.620 | 0.514 |
| Support Vector Machine | 0.743 | 0.959 | 0.755 | 0.800 | 0.766 | 0.686 |

AUC is macro one-vs-rest. Precision, recall and F1 are macro averages. Naive Bayes is the Gaussian version.

Train, CV and test MCC, to check for overfitting:

| ML Model Name | Train MCC | CV MCC | Test MCC | 95% CI (test) |
|---|---|---|---|---|
| Logistic Regression | 0.682 | 0.640 | 0.662 | 0.606 - 0.715 |
| Decision Tree | 0.949 | 0.654 | 0.671 | 0.614 - 0.726 |
| KNN | 1.000 | 0.666 | 0.654 | 0.598 - 0.709 |
| Naive Bayes | 0.549 | 0.543 | 0.514 | 0.454 - 0.569 |
| Support Vector Machine | 0.810 | 0.658 | 0.686 | 0.632 - 0.735 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Most stable model: train and test MCC are very close (0.682 vs 0.662). Very good on K_Scatch and Stains (F1 0.93) but weak on Dirtiness (F1 0.50), so a linear boundary seems too simple for some classes. |
| Decision Tree | Best accuracy (0.746), precision and F1, and the best on Z_Scratch (F1 0.93). It overfits though (train MCC 0.949 vs test 0.671), and it has the lowest AUC (0.872), probably because the leaf probabilities are coarse. |
| KNN | Had the best CV score (0.666) and did well on the small classes (Dirtiness F1 0.87, Stains precision 1.00). Weak on Pastry (recall 0.41). Train MCC of 1.0 is expected with distance weighting, since each point matches itself. |
| Naive Bayes | Worst model (MCC 0.514). It predicts too many Bumps and Pastry: 73 of the 135 Other_Faults rows went to those two classes. Many features are correlated (e.g. Pixels_Areas, LogOfAreas, SigmoidOfAreas), which breaks the independence assumption. |
| Support Vector Machine | Best MCC (0.686), AUC (0.959) and recall (0.800). Balanced class weights helped the small classes (Z_Scratch recall 0.95), but Pastry precision dropped to 0.39 because many Other_Faults rows were predicted as Pastry. |
| Overall, winner for your dataset? | SVM, but only by a small margin. Its confidence interval overlaps with LR, Decision Tree and KNN, and compared to the Decision Tree each model gets about 45 test rows right that the other gets wrong. Naive Bayes is clearly the weakest. |

Other things I noticed:

- Most errors come from Bumps, Other_Faults and Pastry being mixed up (90 of SVM's 100 errors). Other_Faults is a "misc" class, so it's hard to separate.
- The extra features were only picked for Logistic Regression and the Decision Tree.
- When at least 4 of the 5 models agree with the SVM, accuracy is 88% (68% of rows). The other rows are only 45% accurate. I used this idea in the app's triage tab.
- I picked the winner based on the test set, so its score is a bit optimistic. Also, small classes like Dirtiness have only 11 test rows.

## Streamlit app

Features:

- Upload a test CSV (sidebar), or use the bundled `test_data.csv`. A sample file can be downloaded from the sidebar.
- Model selection dropdown
- Shift report tab: the 6 metrics, confusion matrix and classification report
- Model line-up tab: all 5 models compared on the same data
- Consensus triage tab: rows the models agree on are auto-cleared, and the rest are flagged for checking

The uploaded CSV must contain all 27 feature columns and `fault_type`. The app only loads the saved models from `model/` and doesn't retrain anything.
