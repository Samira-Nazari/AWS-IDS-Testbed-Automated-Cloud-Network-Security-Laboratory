# Improvement Report: TST_Complete_project_V9 vs TST_Complete_project_V9

This document explains the main code changes made in `TST_Complete_project_V9`, why they were made, and how they improved the model compared with `TST_Complete_project_V9`.

Note: the filename is kept as `improvemnet.md` because that was the requested name.

## 1. Summary Of The Problem In V9

`TST_Complete_project_V9` had two major model-quality problems:

1. The model often collapsed to one class.
   - In one run, validation accuracy was about `0.049140`, with predictions collapsing to class 4.
   - In a later run after training shuffle changes, validation accuracy became about `0.499924`, but this was still not a real solution because the model was mostly predicting class 0, the majority class.

2. Feature selection was computed but not actually used in the final saved window data.
   - V9 saved selected feature names.
   - But the final `X_windows.npy` still used all original features.
   - This meant the model was not really trained on the selected feature subset.

The main goal in V9 was to make the feature pipeline honest, reduce label imbalance problems, and evaluate the model with metrics that reveal class collapse.

## 2. Result Comparison

### V9 Baseline Behavior

From the V9 training logs:

| Metric | V9 Result |
|---|---:|
| Final train accuracy | about `0.499804` |
| Final validation accuracy | about `0.499924` |
| Main issue | majority-class prediction |
| Predicted behavior | mostly class 0 |

This `0.499924` validation accuracy was misleading because class 0 made up about 50 percent of the validation set.

### V9 Best Validation Result

After the V9 changes:

| Metric | V9 Validation Result |
|---|---:|
| Best validation accuracy | `0.879240` |
| Best validation macro-F1 | `0.876927` |
| Best validation balanced accuracy | `0.948946` |
| Best epoch | epoch 2 |

The validation prediction distribution included all 5 classes, so the model was no longer collapsed to one class.

### V9 Final Test Result

From `outputs/results/evaluation_results.json`:

| Metric | V9 Test Result |
|---|---:|
| Accuracy | `0.925110` |
| Weighted precision | `0.958994` |
| Weighted recall | `0.925110` |
| Weighted F1 | `0.934880` |
| Macro precision | `0.852109` |
| Macro recall | `0.961176` |
| Macro F1 | `0.883356` |
| Balanced accuracy | `0.961176` |

This is much better than the V9 majority-class behavior.

## 3. Pipeline Shape Comparison

### V9 Window Data

Before the V9 feature pipeline fix, the saved window data had this shape:

```text
X_windows: (43841, 20, 48)
```

This means:

1. Window length was 20.
2. The model used 48 features.
3. Feature selection metadata existed, but the final windows still used all features.

### V9 Window Data

After the V9 fix:

```text
X_windows: (43851, 10, 29)
```

This means:

1. Window length is 10.
2. The model uses 29 selected features.
3. The final training data now matches the selected feature list.

This was one of the most important improvements.

## 4. Step-By-Step Code Changes And Reasons

### 4.1 Feature Selection Now Actually Controls The Final Training Data

File changed:

```text
features/feature_engineering.py
```

V9 behavior:

```python
X_windows_best = self.create_sliding_window(X, self.best_window_size, self.best_step_size)
```

Problem:

`X` contained all features, not only selected features. So feature selection did not affect the final model input.

V9 behavior:

```python
X_selected = df[selected_feature_names].values
X_windows_best = self.create_sliding_window(
    X_selected,
    self.best_window_size,
    self.best_step_size
)
```

Reason:

The model should train only on the selected important features. If feature selection says 29 features are important, the final `X_windows.npy` should contain 29 features, not all 48.

Effect:

The final input changed from:

```text
(43841, 20, 48)
```

to:

```text
(43851, 10, 29)
```

This reduced noisy or less useful features and made the training data match the feature engineering result.

### 4.2 Raw Timestamp Was Removed Before Model Training

File changed:

```text
features/feature_engineering.py
```

V9 behavior:

```python
df = df.drop(columns=[
    "Source IP",
    "Destination IP",
    "SimillarHTTP",
    "Flow ID",
    "unique_id"
], errors="ignore")
```

Problem:

`Timestamp` could remain in the numeric feature set. Since timestamp is a raw time value, it can dominate learning or encode dataset-order information instead of real traffic behavior.

V9 behavior:

```python
df = df.drop(columns=[
    "Source IP",
    "Destination IP",
    "SimillarHTTP",
    "Flow ID",
    "unique_id",
    "Timestamp"
], errors="ignore")
```

Reason:

For a classification model, the model should learn from traffic features, not from raw chronological position.

Effect:

The model learned more meaningful traffic patterns and avoided possible time-based leakage or distribution bias.

### 4.3 Selected Feature Names Are Kept In Stable Original Column Order

File changed:

```text
features/feature_engineering.py
```

V9 behavior:

```python
selected_feature_names = list(set(feature_names[best_selected_features % len(feature_names)]))
```

Problem:

Using `set(...)` removes duplicates, but it also destroys column order. That can make metadata unstable and harder to interpret.

V9 behavior:

```python
selected_feature_set = set(feature_names[best_selected_features % len(feature_names)])
selected_feature_names = [col for col in feature_names if col in selected_feature_set]
```

Reason:

The selected features should stay in the same order as the original DataFrame columns.

Effect:

The saved feature list is deterministic and easier to compare, debug, and reproduce.

### 4.4 Added Feature Selection Safety Checks

File changed:

```text
features/feature_engineering.py
```

V9 added:

```python
if best_selected_features is None or best_importances is None:
    raise ValueError("Feature selection failed: no valid window configuration was found.")

if not selected_feature_names:
    raise ValueError("Feature selection failed: no input features were selected.")
```

Reason:

If feature selection fails, the script should stop clearly instead of silently creating invalid model input.

Effect:

The pipeline is safer and easier to debug.

### 4.5 Training Uses Weighted Cross Entropy

File changed:

```text
training/trainer.py
```

V9 behavior:

```python
criterion = nn.CrossEntropyLoss()
```

Problem:

The data is imbalanced:

```text
Class 0: about 50 percent
Class 1: about 33 percent
Class 2: about 8 percent
Class 3: about 5 percent
Class 4: about 5 percent
```

Plain cross entropy can reward the model too much for predicting the majority class.

V9 behavior:

```python
labels = train_loader.dataset.tensors[1].detach().cpu().numpy()
classes, counts = np.unique(labels, return_counts=True)
weights = np.sqrt(counts.max() / counts)

class_weights = torch.ones(int(classes.max()) + 1, dtype=torch.float32)
class_weights[classes.astype(int)] = torch.tensor(weights, dtype=torch.float32)
class_weights = class_weights.to(self.device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
```

Reason:

Minority classes need more influence during training. The square-root weighting is softer than full inverse-frequency weighting, so it helps minority classes without making training too unstable.

Effect:

The model stopped predicting only class 0 and started predicting all 5 classes.

Example V9 validation prediction distribution:

```text
Val pred distribution:
{0: 2537, 1: 2136, 2: 1252, 3: 326, 4: 324}
```

This is much healthier than predicting only class 0.

### 4.6 Validation Now Tracks Macro-F1 And Balanced Accuracy

File changed:

```text
training/trainer.py
```

V9 added:

```python
val_macro_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
val_weighted_f1 = f1_score(val_labels, val_preds, average="weighted", zero_division=0)
val_balanced_acc = balanced_accuracy_score(val_labels, val_preds)
```

Reason:

Accuracy alone was misleading in V9. A model predicting class 0 could get about 50 percent accuracy without learning the real multi-class task.

Macro-F1 treats all classes more equally. Balanced accuracy averages recall across classes.

Effect:

The training log now shows whether the model is learning all classes, not just the majority class.

### 4.7 Training Logs True And Predicted Label Distributions

File changed:

```text
training/trainer.py
```

V9 added:

```python
true_classes, true_counts = np.unique(val_labels, return_counts=True)
pred_classes, pred_counts = np.unique(val_preds, return_counts=True)
true_distribution = dict(zip(true_classes.astype(int), true_counts.astype(int)))
pred_distribution = dict(zip(pred_classes.astype(int), pred_counts.astype(int)))
```

Reason:

This immediately reveals class collapse.

Effect:

When V9 collapsed, it was visible only by reading sample predictions. In V9, collapse would be obvious from:

```text
Val pred distribution
```

This helped confirm that V9 was predicting all 5 classes.

### 4.8 Best Model Is Selected By Macro-F1 Instead Of Accuracy

File changed:

```text
training/trainer.py
```

V9 behavior:

```python
if val_acc > best_val_acc:
    save_model(...)
```

Problem:

Accuracy can prefer majority-class behavior in imbalanced datasets.

V9 behavior:

```python
if val_macro_f1 > best_val_macro_f1:
    self.save_model(f"best_model_epoch_{epoch}.pt")
    self.save_model("best_model.pt")
```

Reason:

Macro-F1 is a better checkpoint metric for imbalanced multi-class classification.

Effect:

The saved `best_model.pt` came from epoch 2, which had the best macro-F1:

```text
Best validation macro-F1: 0.876927
```

This avoids using the final model if later epochs overfit.

### 4.9 Early Stopping Now Actually Stops Training

File changed:

```text
training/trainer.py
```

V9 problem:

The patience counter increased, but training did not break out of the loop.

V9 behavior:

```python
if patience_counter >= max_patience:
    logger.info(
        f"  Early stopping triggered after {patience_counter} epochs without macro-F1 improvement."
    )
    break
```

Reason:

After validation stops improving, continuing training can overfit and waste time.

Effect:

Training stopped at epoch 7:

```text
Early stopping triggered after 5 epochs without macro-F1 improvement.
```

The best model was still saved from epoch 2.

### 4.10 Evaluation Reports Macro And Balanced Metrics

File changed:

```text
evaluation/evaluator.py
```

V9 evaluation reported weighted metrics, but not enough imbalance-aware metrics.

V9 added:

```python
metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
metrics["macro_precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
metrics["macro_recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
```

Reason:

Weighted metrics can still look good when majority classes dominate. Macro metrics reveal minority-class performance more clearly.

Effect:

The final report now includes:

```text
Macro F1:          0.883356
Balanced Accuracy: 0.961176
```

These are better indicators for the imbalanced DDoS classification task.

### 4.11 Evaluation Loads The Best Model Instead Of Always Loading The Final Model

File changed:

```text
evaluation/evaluator.py
```

V9 behavior:

```python
model_path = Path(MODELS_DIR) / "final_model.pt"
```

Problem:

The final epoch is not always the best model. In V9, validation was best at epoch 2, while training continued until epoch 7.

V9 behavior:

```python
best_model_path = Path(MODELS_DIR) / "best_model.pt"
final_model_path = Path(MODELS_DIR) / "final_model.pt"
model_path = best_model_path if best_model_path.exists() else final_model_path
```

Reason:

Evaluation should use the best validation model, not necessarily the last trained model.

Effect:

The test result came from `best_model.pt`, giving a stronger and more reliable evaluation.

### 4.12 Training Hyperparameters Were Adjusted

File changed:

```text
config/config.py
```

V9:

```python
EPOCHS = 5
LEARNING_RATE = 1e-3
```

V9:

```python
EPOCHS = 20
LEARNING_RATE = 5e-4
```

Reason:

The model was given more possible epochs, but with a smaller learning rate for more stable optimization. Early stopping prevents unnecessary long training.

Effect:

The model improved quickly and early stopping stopped the run when macro-F1 stopped improving.

## 5. Final Confusion Matrix Interpretation

V9 test confusion matrix:

```text
[[2852    0    4  391   42]
 [  25 2117   14    0    0]
 [   9    0  502    0    0]
 [   0    0    4  298    0]
 [   0    0    0    4  321]]
```

Strong points:

1. Class 1 is almost perfect.
2. Class 2 is very strong.
3. Class 4 is very strong.
4. Class 3 recall is very high.
5. The model predicts all 5 classes.

Remaining weakness:

Class 3 precision is low:

```text
Class 3 precision: 0.430014
Class 3 recall:    0.986755
```

Reason:

The model predicts class 3 too often. The largest error is:

```text
391 true class-0 samples predicted as class 3
```

This may be caused by the class-3 weight being too strong.

## 6. Recommended Next Improvement

The next experiment should soften the class weights.

Current V9 uses square-root inverse-frequency weights:

```python
weights = np.sqrt(counts.max() / counts)
```

Suggested next test:

```python
weights = (counts.max() / counts) ** 0.25
```

Reason:

This will reduce overprediction of minority classes, especially class 3, while still giving minority classes more importance than class 0.

Expected effect:

1. Class 3 precision should improve.
2. Class 3 recall may decrease slightly.
3. Macro-F1 may improve if false positives decrease more than recall decreases.

## 7. Reproducible V9 Pipeline

To reproduce the V9 results:

```bash
cd /usagers/sanazb/Projects/TST/TST_Complete_project_V9

python preprocessing/preprocessor.py
python features/feature_engineering.py
python utils/helpers.py
python training/trainer.py
python evaluation/evaluator.py
```

Important expected shapes:

```text
After feature engineering:
X_windows: (43851, 10, 29)

After splitting:
X_train: (30693, 10, 29)
X_val:   (6575, 10, 29)
X_test:  (6583, 10, 29)
```

## 8. Main Takeaway

The biggest improvement came from fixing the feature pipeline:

1. V9 computed selected features but trained on all features.
2. V9 trains on the selected features only.
3. V9 removes raw timestamp from model input.
4. V9 uses class-aware training and macro-F1 checkpointing.
5. V9 evaluates with macro and balanced metrics.

This changed the model from majority-class behavior to real 5-class classification, improving validation accuracy from about `0.499924` to `0.879240`, and producing a final test accuracy of `0.925110` with macro-F1 of `0.883356`.

---

# Version 6 Improvements Compared With Version 5

Version 6 keeps the same CIC-DDoS2019 TST classification goal as Version 5, but the code was changed to make the pipeline more leakage-safe and more reliable for imbalanced classes.

## 1. Leakage-Safe Feature Engineering

In Version 5, feature engineering and window creation were still closely tied to the full dataset workflow.

In Version 6, `features/feature_engineering.py` uses:

```python
engineer_features_no_leakage(...)
```

The improved order is:

1. Split the dataframe per class in chronological order.
2. Fit RandomForest feature selection on the training split only.
3. Fit `StandardScaler` on the selected training features only.
4. Transform validation and test features using the training scaler.
5. Create train, validation, and test windows separately.

This prevents validation/test data from influencing feature selection or scaling.

## 2. Split Arrays Are Created Inside Feature Engineering

In Version 5, the standalone pipeline used:

```text
features/feature_engineering.py
utils/helpers.py
```

Feature engineering created window arrays, then helper code created the train/validation/test split.

In Version 6, feature engineering directly saves:

```text
outputs/data/X_train.npy
outputs/data/X_val.npy
outputs/data/X_test.npy
outputs/data/y_train.npy
outputs/data/y_val.npy
outputs/data/y_test.npy
```

So `utils/helpers.py` is no longer required as a normal standalone pipeline step.

## 3. Standardization Moved After The Split

In Version 5, preprocessing was responsible for standardization.

In Version 6, preprocessing is called with:

```python
standardize=False
```

Then scaling happens inside leakage-safe feature engineering after the split. This means the scaler is fitted only on training data, which is the correct evaluation setup.

## 4. New Metadata Files

Version 6 saves extra metadata in `outputs/data/`:

```text
standard_scaler.pkl
label_classes.pkl
window_config.pkl
```

These files store the train-fitted scaler, the training label order, and the selected best window/step configuration.

## 5. Label Encoding Is Train-Based

Version 6 fits the `LabelEncoder` using training labels and saves it to:

```text
outputs/models/label_encoder.pkl
```

Evaluation loads this encoder and transforms test labels before computing metrics. This keeps label IDs consistent between training and testing.

## 6. Training Uses Weighted Cross Entropy

Version 6 computes class weights from the training label distribution:

```python
weights = np.sqrt(counts.max() / counts)
```

These weights are used in:

```python
nn.CrossEntropyLoss(weight=class_weights)
```

This helps the model pay more attention to minority classes instead of optimizing mostly for the majority class.

## 7. Best Model Is Selected By Macro-F1

Version 6 saves the best checkpoint when validation macro-F1 improves:

```text
best_model.pt
best_model_epoch_*.pt
```

Macro-F1 is more useful than accuracy for this project because the CIC-DDoS2019 classes are imbalanced.

## 8. Early Stopping Added

Version 6 stops training after 5 epochs without validation macro-F1 improvement.

This avoids continuing training after the main minority-class-aware validation metric stops improving.

## 9. Evaluation Metrics Expanded

Version 6 evaluation reports:

```text
accuracy
balanced_accuracy
weighted precision / recall / F1
macro precision / recall / F1
per-class precision / recall / F1
confusion matrix
true label distribution
predicted label distribution
```

This makes minority-class problems easier to identify. For example, high accuracy can hide weak recall for a small class, while macro-F1 and balanced accuracy show that weakness.

## 10. Main Pipeline Updated

Version 6 `main.py` follows the improved end-to-end order:

1. Load selected CSV files.
2. Preprocess without global standardization.
3. Run leakage-safe feature engineering.
4. Save split arrays and metadata.
5. Fit and save the label encoder.
6. Create the TST model from the actual input dimensions.
7. Train with weighted loss and macro-F1 checkpointing.
8. Evaluate on the held-out test split.

## 11. Device Handling Clarified

Version 6 still reads `USE_GPU` from `config/config.py`, but the actual runtime device is selected with:

```python
device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
```

So the config can print `cuda`, while the actual script can still run on `cpu` if CUDA is unavailable in the active environment.

## 12. Main Takeaway

The most important Version 6 improvement over Version 5 is the leakage-safe training pipeline.

Version 6 makes sure feature selection, scaling, label encoding, checkpointing, and evaluation are all based on the correct split boundaries. This gives more trustworthy test results and makes macro-F1, balanced accuracy, and per-class metrics more meaningful for the imbalanced DDoS classification task.

---

# Additional Improvement Report: TST_Complete_project_V9 vs TST_Complete_project_V9

This section documents the new changes added in `TST_Complete_project_V9` compared with `TST_Complete_project_V9`.

## 1. Main Difference

The most important V9 improvement is that the pipeline is now more leakage-safe.

In V9, the code created one full window dataset first, then split the window arrays into train, validation, and test sets. This meant feature selection and some preprocessing decisions could be influenced by data that later became validation or test data.

In V9, the dataframe is split first, then feature selection, scaling, and window creation happen using the correct split boundaries.

## 2. Feature Engineering Changed From Global To Split-Aware

File changed:

```text
features/feature_engineering.py
```

V9 used:

```python
engineer_features(...)
```

V9 adds and uses:

```python
engineer_features_no_leakage(...)
```

The new V9 function does this:

1. Splits the dataframe per class in chronological order.
2. Fits RandomForest feature selection on the training split only.
3. Fits `StandardScaler` on selected training features only.
4. Transforms validation and test data with the training scaler.
5. Creates train, validation, and test windows separately.

Reason:

Validation and test data should not influence feature selection, scaling, or window configuration. V9 keeps those decisions tied to the training split, so final evaluation is more trustworthy.

## 3. Preprocessing No Longer Standardizes Globally

File changed:

```text
preprocessing/preprocessor.py
```

V9 preprocessing standardized features before the final train/validation/test split.

V9 adds a `standardize` option:

```python
preprocess_data(..., standardize=False)
```

Reason:

Scaling before splitting can leak validation/test statistics into training. V9 delays standardization until after the split, then fits the scaler only on selected training features.

## 4. Split Outputs Are Created Inside Feature Engineering

In V9, feature engineering produced:

```text
X_windows.npy
y_windows.npy
```

Then `utils/helpers.py` split those windows.

In V9, feature engineering directly saves:

```text
X_train.npy
X_val.npy
X_test.npy
y_train.npy
y_val.npy
y_test.npy
```

Reason:

The split must happen before leakage-sensitive steps. So V9 moves the split earlier and makes feature engineering responsible for the final split window arrays.

## 5. New Metadata Files

V9 saves additional metadata:

```text
outputs/data/standard_scaler.pkl
outputs/data/label_classes.pkl
outputs/data/window_config.pkl
```

These files store:

1. The scaler fitted on training data.
2. The label order used during leakage-safe feature engineering.
3. The selected best window size and step size.

This makes the pipeline easier to reproduce and debug.

## 6. Main Pipeline Uses The New Leakage-Safe Flow

File changed:

```text
main.py
```

V9 called:

```python
feature_input = _prepare_feature_engineering_input(preprocessed_data)
X_windows, y_windows, selected_features, columns_to_drop = engineer_features(...)
X_train, X_val, X_test, y_train, y_val, y_test = split_per_class_time_series(...)
```

V9 calls:

```python
result = engineer_features_no_leakage(
    preprocessed_data,
    WINDOW_SIZES,
    STEP_SIZES,
    target_column="Label",
    valid_size=VALID_SIZE,
    test_size=TEST_SIZE,
    rf_n_estimators=RF_N_ESTIMATORS,
    cv_splits=CV_SPLITS,
)
```

Reason:

This makes `main.py` follow the corrected order:

1. Load data.
2. Clean/drop columns without global standardization.
3. Split chronologically per class.
4. Select features using training data only.
5. Scale using training data only.
6. Create split-specific windows.
7. Train and evaluate.

## 7. Config Values Are Now Used More Directly

V9 uses these config values inside leakage-safe feature engineering:

```python
RF_N_ESTIMATORS = 100
CV_SPLITS = 5
VALID_SIZE = 0.2
TEST_SIZE = 0.2
```

Reason:

The RandomForest feature selector, cross-validation splits, validation size, and test size are now explicitly controlled by `config/config.py` instead of being hidden inside standalone helper logic.

## 8. Standalone Stage Order Changed

In V9, the normal standalone flow included:

```text
python utils/helpers.py
```

In V9, `features/feature_engineering.py` creates the train/validation/test arrays directly, so `utils/helpers.py` is no longer a required standalone pipeline stage.

The V9 standalone order is:

```text
python data/data_loader.py
python preprocessing/preprocessor.py
python features/feature_engineering.py
python training/trainer.py
python evaluation/evaluator.py
```

## 9. Expected Quality Improvement

V9 may not always produce higher raw accuracy than V9, because it removes leakage that can make results look better than they really are.

The improvement is reliability:

1. Validation and test metrics are more honest.
2. Feature selection is based only on training data.
3. Scaling is based only on training data.
4. Window generation respects split boundaries.
5. Saved metadata makes the experiment easier to repeat.

## 10. Final Takeaway

V9 improved model behavior and imbalance-aware training.

V9 improves experimental correctness.

The key upgrade is not just another model tweak. It is a safer data pipeline that prevents validation/test information from influencing feature selection and scaling before evaluation.
