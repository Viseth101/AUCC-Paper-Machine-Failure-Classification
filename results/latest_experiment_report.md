# Experiment Summary Report (2026-11-17 22:41:41)

```text
================================================================================
EXPERIMENT RUN LOG — 2026-11-17 22:41:41
================================================================================

1. MASTER CONFIGURATION SETTINGS:
   - RANDOM_STATE           : 2077
   - NUMBER_OF_FEATURES     : 4
   - RFE_SELECTOR_MODEL     : Logistic Regression
   - SFS_SELECTOR_MODEL     : Decision Tree
   - PCA_VARIANCE           : 0.95 (4 components kept)
   - CV_FOLDS               : 5
   - UNIQUE_FINALIST_MODELS : False

2. FEATURES SELECTED BY EACH METHOD:
   - Baseline (All 6)       : Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]
   - ANOVA                  : Air temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]
   - Pearson Correlation    : Torque [Nm], Tool wear [min], Air temperature [K], Rotational speed [rpm]
   - Adapted Lasso (L1)     : Torque [Nm], Rotational speed [rpm], Air temperature [K], Process temperature [K]
   - RFE (Logistic Regression) : Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm]
   - Forward SFS (Decision Tree) : Air temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]
   - Backward SFS (Decision Tree): Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm]
   - PCA (95% var)        : 4 orthogonal components

3. BEST MODEL PER FEATURE-SELECTION METHOD (VALIDATION SCREENING):
   Feature selection             Model  Validation failure F1  Validation failure recall  Validation failure precision  Validation missed failures  Validation false alarms
      Drop Type only Gradient Boosting               0.650000                   0.509804                      0.896552                          25                        3
No feature selection Gradient Boosting               0.641026                   0.490196                      0.925926                          26                        2
       Adapted Lasso     Random Forest               0.583333                   0.411765                      1.000000                          30                        0
  Backward selection     Random Forest               0.583333                   0.411765                      1.000000                          30                        0
                 RFE     Random Forest               0.583333                   0.411765                      1.000000                          30                        0
               ANOVA     Decision Tree               0.426966                   0.372549                      0.500000                          32                       19
 Pearson correlation     Decision Tree               0.426966                   0.372549                      0.500000                          32                       19
   Forward selection     Decision Tree               0.426966                   0.372549                      0.500000                          32                       19
                 PCA Gradient Boosting               0.388889                   0.274510                      0.666667                          37                        7

4. TOP 3 FINALISTS AFTER 5-FOLD CV HYPERPARAMETER TUNING (VALIDATION):
   Feature selection             Model  Validation F1  Validation recall  Validation precision  Validation missed failures  Validation false alarms
No feature selection Gradient Boosting       0.712644           0.607843              0.861111                          20                        5
No feature selection     Random Forest       0.675000           0.529412              0.931034                          24                        2
      Drop Type only Gradient Boosting       0.674699           0.549020              0.875000                          23                        4

5. FINAL TEST COMPARISON (SINGLE WINNER VS. STACKING):
   Feature selection                  Model                                                                                                 Features  Test F1  Test recall  Test precision  Test balanced accuracy  Test missed failures  Test false alarms
No feature selection      Gradient Boosting Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min] 0.729412     0.607843        0.911765                0.802886                    20                  3
No feature selection Stacking (Tuned Top 3) Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min] 0.738095     0.607843        0.939394                0.803231                    20                  2

6. SELECTED WINNER & TUNED HYPERPARAMETERS:
   - Winning Model          : Gradient Boosting
   - Winning Feature Method : No feature selection (Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min])
   - Best Hyperparameters   : {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200, 'subsample': 1.0}

7. FINAL UNSEEN TEST SET PERFORMANCE (SINGLE WINNER VS. STACKING):
   Feature selection                  Model                                                                                                 Features  Test F1  Test recall  Test precision  Test balanced accuracy  Test missed failures  Test false alarms
No feature selection      Gradient Boosting Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min] 0.729412     0.607843        0.911765                0.802886                    20                  3
No feature selection Stacking (Tuned Top 3) Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min] 0.738095     0.607843        0.939394                0.803231                    20                  2
================================================================================

   - Test Failure F1        : 0.7294
   - Test Failure Recall    : 0.6078
   - Test Failure Precision : 0.9118
   - Test Balanced Accuracy : 0.8029
   - Test Missed Failures   : 20
   - Test False Alarms      : 3
================================================================================


```
