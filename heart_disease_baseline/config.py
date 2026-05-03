"""Stores dataset paths, feature lists, split settings, and the model selection rule."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "UCI_Heart_Disease_Dataset_Combined.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

TARGET_COLUMN = "HeartDisease"
FEATURE_COLUMNS = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
]
CATEGORICAL_COLUMNS = [
    "Sex",
    "ChestPainType",
    "FastingBS",
    "RestingECG",
    "ExerciseAngina",
]
NUMERIC_COLUMNS = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "MaxHR",
    "Oldpeak",
]

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
TUNING_REFIT_METRIC = "recall"

# Heart disease screening should prioritize catching positive cases.
# The best model is therefore selected by minimizing false negatives first,
# then maximizing recall, with broader quality metrics only used as tie-breakers.
MODEL_SELECTION_PRIORITY = [
    "false_negatives",
    "recall",
    "f1",
    "accuracy",
]
