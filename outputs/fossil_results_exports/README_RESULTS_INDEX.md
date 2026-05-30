# Fossil Results Export Index

## CSV Files

- `csv/all_tests.csv`: every recorded test run, including exploratory tests and preprocessing experiment tests.
- `csv/model_comparison_initial_vs_updated.csv`: main comparison table for the five preprocessing models, showing initial 10-image test versus updated 38-image test.
- `csv/per_class_updated_38_image_test.csv`: coral-vs-shell performance for each model on the updated 38-image wild test.
- `csv/incorrect_predictions_updated_38_image_test.csv`: every wrong prediction on the updated 38-image wild test, with true class, predicted class, and confidence.
- `csv/training_history_all_models.csv`: epoch-by-epoch training and validation metrics for all saved models.
- `csv/notes.csv`: definitions and interpretation notes.

## Visuals

- `visuals/01_accuracy_initial_vs_updated.png`: bar chart comparing initial wild-test accuracy against updated 38-image wild-test accuracy.
- `visuals/02_correct_vs_incorrect_updated_38.png`: stacked bar chart showing correct versus incorrect predictions on the updated test set.
- `visuals/03_per_class_accuracy_updated_38.png`: line chart comparing coral and shell accuracy per model on the updated test set.

## Key Interpretation

- The updated 38-image wild test is more reliable than the initial 10-image wild test.
- Current best model on the updated test is `original_enhanced`: 34/38 correct, 89.5% accuracy.
- The `original_enhanced` model was trained on original plus enhanced JPEG versions, not enhanced-only images.
