import os
import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(BASE_DIR, "models")


# ============================================================
# LOAD RANDOM FOREST
# ============================================================

rf_model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "random_forest_model.pkl"
    )
)


# ============================================================
# LOAD XGBOOST
# ============================================================

xgb_model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "xgboost_model.pkl"
    )
)

xgb_preprocessor = joblib.load(
    os.path.join(
        MODEL_DIR,
        "xgboost_preprocessor.pkl"
    )
)


# ============================================================
# LOAD EXTRA TREES
# ============================================================

extra_model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "extra_trees_model.pkl"
    )
)

extra_preprocessor = joblib.load(
    os.path.join(
        MODEL_DIR,
        "extra_trees_preprocessor.pkl"
    )
)


# ============================================================
# LOAD STAGE ENCODER
# ============================================================

stage_encoder = None

stage_encoder_path = os.path.join(
    MODEL_DIR,
    "stage_encoder.pkl"
)

if os.path.exists(stage_encoder_path):

    stage_encoder = joblib.load(
        stage_encoder_path
    )


# ============================================================
# GET EXPECTED COLUMNS FROM RF PIPELINE
# ============================================================

def get_pipeline_columns(model):

    if not hasattr(model, "named_steps"):
        return None

    if "preprocessor" not in model.named_steps:
        return None

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    if hasattr(
        preprocessor,
        "feature_names_in_"
    ):

        return list(
            preprocessor.feature_names_in_
        )

    return None


# ============================================================
# PREPARE RANDOM FOREST INPUT
# ============================================================

def prepare_rf_input(patient):

    patient_df = pd.DataFrame(
        [patient]
    )

    expected_columns = get_pipeline_columns(
        rf_model
    )

    if expected_columns is not None:

        missing = [
            col
            for col in expected_columns
            if col not in patient_df.columns
        ]

        if missing:

            raise ValueError(
                "Random Forest missing columns: "
                + ", ".join(missing)
            )

        patient_df = patient_df[
            expected_columns
        ]

    return patient_df


# ============================================================
# PREPARE XGBOOST INPUT
# ============================================================

def prepare_xgb_input(patient):

    patient_df = pd.DataFrame(
        [patient]
    )

    # --------------------------------------------------------
    # Get the columns expected by the XGBoost preprocessor
    # --------------------------------------------------------

    if hasattr(
        xgb_preprocessor,
        "feature_names_in_"
    ):

        expected_columns = list(
            xgb_preprocessor.feature_names_in_
        )

        missing = [
            col
            for col in expected_columns
            if col not in patient_df.columns
        ]

        if missing:

            raise ValueError(
                "XGBoost missing columns: "
                + ", ".join(missing)
            )

        patient_df = patient_df[
            expected_columns
        ]

    # --------------------------------------------------------
    # Transform categorical variables
    # --------------------------------------------------------

    xgb_input = xgb_preprocessor.transform(
        patient_df
    )

    return xgb_input


# ============================================================
# PREPARE EXTRA TREES INPUT
# ============================================================

def prepare_extra_input(patient):

    patient_df = pd.DataFrame(
        [patient]
    )

    # --------------------------------------------------------
    # Get expected columns
    # --------------------------------------------------------

    if hasattr(
        extra_preprocessor,
        "feature_names_in_"
    ):

        expected_columns = list(
            extra_preprocessor.feature_names_in_
        )

        missing = [
            col
            for col in expected_columns
            if col not in patient_df.columns
        ]

        if missing:

            raise ValueError(
                "Extra Trees missing columns: "
                + ", ".join(missing)
            )

        patient_df = patient_df[
            expected_columns
        ]

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    extra_input = extra_preprocessor.transform(
        patient_df
    )

    return extra_input


# ============================================================
# DECODE STAGE
# ============================================================

def decode_stage(value):

    # Already a string
    if isinstance(value, str):

        return value

    # Use saved encoder
    if stage_encoder is not None:

        try:

            return stage_encoder.inverse_transform(
                [int(value)]
            )[0]

        except Exception:
            pass

    # Fallback
    stage_mapping = {
        0: "Stage I",
        1: "Stage II",
        2: "Stage III",
        3: "Stage IV"
    }

    return stage_mapping.get(
        int(value),
        str(value)
    )


# ============================================================
# GET CLASS LABELS
# ============================================================

def get_class_labels(model):

    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    if hasattr(
        model,
        "named_steps"
    ):

        final_model = list(
            model.named_steps.values()
        )[-1]

        if hasattr(
            final_model,
            "classes_"
        ):

            classes = final_model.classes_

        else:

            return None

    # --------------------------------------------------------
    # Normal model
    # --------------------------------------------------------

    elif hasattr(
        model,
        "classes_"
    ):

        classes = model.classes_

    else:

        return None

    labels = []

    for value in classes:

        labels.append(
            decode_stage(value)
        )

    return labels


# ============================================================
# RANDOM FOREST PREDICTION
# ============================================================

def predict_random_forest(patient):

    patient_df = prepare_rf_input(
        patient
    )

    # Pipeline does preprocessing internally
    prediction_raw = rf_model.predict(
        patient_df
    )[0]

    prediction = decode_stage(
        prediction_raw
    )

    probabilities = rf_model.predict_proba(
        patient_df
    )[0]

    confidence = float(
        max(probabilities) * 100
    )

    labels = get_class_labels(
        rf_model
    )

    probability_dict = {}

    if labels is not None:

        probability_dict = {
            label: float(prob * 100)
            for label, prob in zip(
                labels,
                probabilities
            )
        }

    return {
        "stage": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "probability_dict": probability_dict
    }


# ============================================================
# XGBOOST PREDICTION
# ============================================================

def predict_xgboost(patient):

    # IMPORTANT:
    # XGBoost is NOT a pipeline.
    # Therefore preprocess first.

    xgb_input = prepare_xgb_input(
        patient
    )

    prediction_raw = xgb_model.predict(
        xgb_input
    )[0]

    prediction = decode_stage(
        prediction_raw
    )

    probabilities = xgb_model.predict_proba(
        xgb_input
    )[0]

    confidence = float(
        max(probabilities) * 100
    )

    labels = get_class_labels(
        xgb_model
    )

    probability_dict = {}

    if labels is not None:

        probability_dict = {
            label: float(prob * 100)
            for label, prob in zip(
                labels,
                probabilities
            )
        }

    return {
        "stage": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "probability_dict": probability_dict
    }


# ============================================================
# EXTRA TREES PREDICTION
# ============================================================

def predict_extra_trees(patient):

    # IMPORTANT:
    # Extra Trees uses its saved preprocessor.

    extra_input = prepare_extra_input(
        patient
    )

    prediction_raw = extra_model.predict(
        extra_input
    )[0]

    prediction = decode_stage(
        prediction_raw
    )

    probabilities = extra_model.predict_proba(
        extra_input
    )[0]

    confidence = float(
        max(probabilities) * 100
    )

    labels = get_class_labels(
        extra_model
    )

    probability_dict = {}

    if labels is not None:

        probability_dict = {
            label: float(prob * 100)
            for label, prob in zip(
                labels,
                probabilities
            )
        }

    return {
        "stage": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "probability_dict": probability_dict
    }


# ============================================================
# PREDICT USING ALL THREE MODELS
# ============================================================

def predict_all_models(patient):

    results = {}

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    results["Random Forest"] = (
        predict_random_forest(patient)
    )

    # --------------------------------------------------------
    # XGBOOST
    # --------------------------------------------------------

    results["XGBoost"] = (
        predict_xgboost(patient)
    )

    # --------------------------------------------------------
    # EXTRA TREES
    # --------------------------------------------------------

    results["Extra Trees"] = (
        predict_extra_trees(patient)
    )

    return results