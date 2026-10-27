import streamlit as st

import os
import tempfile
import time
import matplotlib.pyplot as plt
import pandas as pd

from utils.segmentation import segment_pancreas
from utils.feature_extraction import extract_features
from utils.prediction import predict_all_models
from utils.risk_score import calculate_risk_score


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Pancreatic Cancer Analysis System",
    page_icon="🩺",
    layout="wide"
)


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🩺 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "🧠 CT Analysis",
        "📋 Clinical Prediction",
        "📄 Final Report"
    ]
)


# =====================================================
# HOME PAGE
# =====================================================

if page == "🏠 Home":

    st.title("🩺 Pancreatic Cancer Analysis System")

    st.markdown(
        "### AI-assisted pancreatic cancer analysis using CT imaging and clinical data."
    )

    st.markdown("---")

    # =================================================
    # WELCOME SECTION
    # =================================================

    st.markdown("""
    ## Welcome

    This system combines **Deep Learning** and **Machine Learning**
    to assist in pancreatic cancer analysis.

    The platform performs pancreas segmentation from CT scans,
    extracts imaging biomarkers, evaluates imaging risk, and
    predicts the clinical stage using multiple machine learning models.
    """)

    st.markdown("---")

    # =================================================
    # MAIN FEATURES
    # =================================================

    st.subheader("System Features")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("### 🧠 CT Analysis")

        st.success("Deep Learning")

        st.write(
            "Upload a CT scan and perform automatic pancreas segmentation."
        )

        st.write("• CT scan preprocessing")
        st.write("• 3D U-Net segmentation")
        st.write("• Imaging biomarker extraction")

    with col2:

        st.markdown("### 📋 Clinical Prediction")

        st.info("Machine Learning")

        st.write(
            "Enter patient clinical information to predict the cancer stage."
        )

        st.write("• Clinical risk factors")
        st.write("• Multiple ML models")
        st.write("• Stage prediction")
        st.write("• Confidence score")

    with col3:

        st.markdown("### 📄 Final Report")

        st.warning("Analysis Summary")

        st.write(
            "View the combined imaging and clinical analysis results."
        )

        st.write("• Imaging biomarkers")
        st.write("• Imaging risk assessment")
        st.write("• Predicted cancer stage")
        st.write("• Prediction confidence")

    st.markdown("---")

    # =================================================
    # GET STARTED
    # =================================================

    st.subheader("Get Started")

    st.info(
        "Use the navigation menu on the left to begin the analysis."
    )

    st.write(
        "**Recommended sequence:** CT Analysis → Clinical Prediction → Final Report"
    )


# =====================================================
# CT ANALYSIS
# =====================================================

elif page == "🧠 CT Analysis":

    st.title("🧠 CT Image Analysis")

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload CT Scan (.nii / .nii.gz)",
        type=["nii", "nii.gz"]
    )

    if uploaded_file is not None:

        extension = (
            ".nii.gz"
            if uploaded_file.name.endswith(".nii.gz")
            else ".nii"
        )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as tmp_file:

            tmp_file.write(uploaded_file.read())

            temp_path = tmp_file.name

        st.success("✅ CT Scan Uploaded Successfully!")

        try:

            start = time.time()

            with st.spinner(
                "Running pancreas segmentation..."
            ):

                image, mask = segment_pancreas(temp_path)

            end = time.time()

            st.success("✅ Segmentation Completed")

            st.info(
                f"Processing Time : {end-start:.2f} seconds"
            )

            st.write("Image Shape :", image.shape)
            st.write("Mask Shape :", mask.shape)

            # -------------------------------------
            # Display Images
            # -------------------------------------

            st.markdown("---")

            st.subheader("Segmentation Results")

            slice_idx = image.shape[2] // 2

            col1, col2 = st.columns(2)

            with col1:

                st.write("### Original CT")

                fig, ax = plt.subplots(figsize=(5, 5))

                ax.imshow(
                    image[:, :, slice_idx],
                    cmap="gray"
                )

                ax.axis("off")

                st.pyplot(fig)

                plt.close(fig)

            with col2:

                st.write("### Predicted Mask")

                fig, ax = plt.subplots(figsize=(5, 5))

                ax.imshow(
                    mask[:, :, slice_idx],
                    cmap="viridis"
                )

                ax.axis("off")

                st.pyplot(fig)

                plt.close(fig)

            # -------------------------------------
            # Feature Extraction
            # -------------------------------------

            st.markdown("---")

            st.subheader("Imaging Biomarkers")

            features = extract_features(
                image,
                mask
            )

            st.write("Features returned:")
            st.write(features)

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Volume",
                    f"{features['Volume']:.2f}"
                )

                st.metric(
                    "Maximum Intensity",
                    f"{features['Maximum Intensity']:.4f}"
                )

            with c2:

                st.metric(
                    "Mean Intensity",
                    f"{features['Mean Intensity']:.4f}"
                )

                st.metric(
                    "Minimum Intensity",
                    f"{features['Minimum Intensity']:.4f}"
                )

            with c3:

                st.metric(
                    "Standard Deviation",
                    f"{features['Standard Deviation']:.4f}"
                )

            # -------------------------------------
            # Imaging Risk Assessment
            # -------------------------------------

            score, level, interpretation = calculate_risk_score(
                features
            )

            st.session_state["risk_score"] = score
            st.session_state["risk_level"] = level
            st.session_state["risk_interpretation"] = interpretation

            st.markdown("---")

            st.subheader("Imaging Risk Assessment")

            c1, c2 = st.columns(2)

            with c1:

                st.metric(
                    "Risk Score",
                    f"{score}/100"
                )

            with c2:

                if level == "Low Risk":

                    st.success(level)

                elif level == "Moderate Risk":

                    st.warning(level)

                else:

                    st.error(level)

            st.info(interpretation)

            # -------------------------------------
            # Store for Final Report
            # -------------------------------------

            st.session_state["ct_image"] = image
            st.session_state["ct_mask"] = mask
            st.session_state["imaging_features"] = features

            st.session_state["risk_score"] = score
            st.session_state["risk_level"] = level
            st.session_state["risk_interpretation"] = interpretation

        except Exception as e:

            st.error("Segmentation Failed")

            st.exception(e)

        finally:

            if os.path.exists(temp_path):

                os.remove(temp_path)


# =====================================================
# CLINICAL PREDICTION
# =====================================================

elif page == "📋 Clinical Prediction":

    st.title("📋 Clinical Stage Prediction")

    st.markdown("---")

    st.info(
        "Enter the patient's clinical information to predict "
        "the pancreatic cancer stage using three ML models."
    )

    st.subheader("Patient Information")

    col1, col2 = st.columns(2)

    # =================================================
    # COLUMN 1
    # =================================================

    with col1:

        country = st.selectbox(
            "Country",
            [
                "India",
                "United States",
                "Canada",
                "China",
                "Australia",
                "Germany",
                "South Africa",
                "United Kingdom",
                "Brazil"
            ]
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=50
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        smoking = st.selectbox(
            "Smoking History",
            [0, 1]
        )

        obesity = st.selectbox(
            "Obesity",
            [0, 1]
        )

        diabetes = st.selectbox(
            "Diabetes",
            [0, 1]
        )

        chronic = st.selectbox(
            "Chronic Pancreatitis",
            [0, 1]
        )

        family = st.selectbox(
            "Family History",
            [0, 1]
        )

        hereditary = st.selectbox(
            "Hereditary Condition",
            [0, 1]
        )

    # =================================================
    # COLUMN 2
    # =================================================

    with col2:

        jaundice = st.selectbox(
            "Jaundice",
            [0, 1]
        )

        abdominal = st.selectbox(
            "Abdominal Discomfort",
            [0, 1]
        )

        backpain = st.selectbox(
            "Back Pain",
            [0, 1]
        )

        weightloss = st.selectbox(
            "Weight Loss",
            [0, 1]
        )

        type2 = st.selectbox(
            "Development of Type2 Diabetes",
            [0, 1]
        )

        alcohol = st.selectbox(
            "Alcohol Consumption",
            [0, 1]
        )

        physical = st.selectbox(
            "Physical Activity Level",
            [
                "Low",
                "Medium",
                "High"
            ]
        )

        diet = st.selectbox(
            "Diet Processed Food",
            [
                "Low",
                "Medium",
                "High"
            ]
        )

        healthcare = st.selectbox(
            "Access to Healthcare",
            [
                "Low",
                "Medium",
                "High"
            ]
        )

    st.markdown("---")

    # =================================================
    # PREDICT
    # =================================================

    if st.button(
        "🩺 Predict Cancer Stage",
        use_container_width=True
    ):

        patient = {

            "Country": country,

            "Age": age,

            "Gender": gender,

            "Smoking_History": smoking,

            "Obesity": obesity,

            "Diabetes": diabetes,

            "Chronic_Pancreatitis": chronic,

            "Family_History": family,

            "Hereditary_Condition": hereditary,

            "Jaundice": jaundice,

            "Abdominal_Discomfort": abdominal,

            "Back_Pain": backpain,

            "Weight_Loss": weightloss,

            "Development_of_Type2_Diabetes": type2,

            "Alcohol_Consumption": alcohol,

            "Physical_Activity_Level": physical,

            "Diet_Processed_Food": diet,

            "Access_to_Healthcare": healthcare

        }

        try:

            with st.spinner(
                "Running three ML models..."
            ):

                results = predict_all_models(patient)

            # Store results

            st.session_state["model_results"] = results

            # =================================================
            # MODEL RESULTS
            # =================================================

            st.success(
                "✅ Prediction completed successfully!"
            )

            st.markdown("---")

            st.subheader("🤖 Model Predictions")

            model_col1, model_col2, model_col3 = st.columns(3)

            # -------------------------------------------------
            # RANDOM FOREST
            # -------------------------------------------------

            with model_col1:

                st.markdown("### 🌲 Random Forest")

                st.metric(
                    "Predicted Stage",
                    results["Random Forest"]["stage"]
                )

                st.metric(
                    "Confidence",
                    f"{results['Random Forest']['confidence']:.2f}%"
                )

            # -------------------------------------------------
            # XGBOOST
            # -------------------------------------------------

            with model_col2:

                st.markdown("### 🚀 XGBoost")

                st.metric(
                    "Predicted Stage",
                    results["XGBoost"]["stage"]
                )

                st.metric(
                    "Confidence",
                    f"{results['XGBoost']['confidence']:.2f}%"
                )

                st.success(
                    "🏆 Best Performing Model"
                )

            # -------------------------------------------------
            # EXTRA TREES
            # -------------------------------------------------

            with model_col3:

                st.markdown("### 🌳 Extra Trees")

                st.metric(
                    "Predicted Stage",
                    results["Extra Trees"]["stage"]
                )

                st.metric(
                    "Confidence",
                    f"{results['Extra Trees']['confidence']:.2f}%"
                )

            # =================================================
            # PERFORMANCE COMPARISON
            # =================================================

            st.markdown("---")

            st.subheader("📊 Model Performance Comparison")

            comparison_data = pd.DataFrame({

                "Model": [
                    "Random Forest",
                    "XGBoost",
                    "Extra Trees"
                ],

                "Accuracy": [
                    90.74,
                    91.31,
                    87.92
                ],

                "Macro F1": [
                    90.75,
                    91.32,
                    87.93
                ]

            })

            st.dataframe(
                comparison_data,
                use_container_width=True,
                hide_index=True
            )

            # =================================================
            # FINAL PREDICTION
            # =================================================

            st.markdown("---")

            st.subheader("🏆 Final Stage Prediction")

            final_stage = results["XGBoost"]["stage"]

            final_confidence = results["XGBoost"]["confidence"]

            st.success(
                f"Predicted Cancer Stage: **{final_stage}**"
            )

            st.metric(
                "XGBoost Confidence",
                f"{final_confidence:.2f}%"
            )

            st.caption(
                "XGBoost was selected as the final model because "
                "it achieved the highest validation accuracy "
                "(91.31%) among the three evaluated models."
            )

            # Store final prediction

            st.session_state["predicted_stage"] = final_stage

            st.session_state["confidence"] = final_confidence

        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.exception(e)


# =====================================================
# FINAL REPORT
# =====================================================

elif page == "📄 Final Report":

    st.title("📄 Final Patient Report")

    st.markdown("---")

    if "imaging_features" not in st.session_state:

        st.warning(
            "⚠ Please complete CT Analysis first."
        )

        st.stop()

    if "predicted_stage" not in st.session_state:

        st.warning(
            "⚠ Please complete Clinical Prediction first."
        )

        st.stop()

    features = st.session_state["imaging_features"]

    stage = st.session_state["predicted_stage"]

    confidence = st.session_state["confidence"]

    risk_score = st.session_state.get(
        "risk_score",
        0
    )

    risk_level = st.session_state.get(
        "risk_level",
        "Not Available"
    )

    interpretation = st.session_state.get(
        "risk_interpretation",
        "Risk assessment not available."
    )

    st.header("Patient Summary")

    st.success(
        "Analysis Completed Successfully"
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    # =====================================================
    # LEFT COLUMN
    # =====================================================

    with col1:

        st.subheader("Imaging Biomarkers")

        st.metric(
            "Volume",
            f"{features['Volume']:.2f}"
        )

        st.metric(
            "Mean Intensity",
            f"{features['Mean Intensity']:.4f}"
        )

        st.metric(
            "Standard Deviation",
            f"{features['Standard Deviation']:.4f}"
        )

        st.metric(
            "Maximum Intensity",
            f"{features['Maximum Intensity']:.4f}"
        )

        st.metric(
            "Minimum Intensity",
            f"{features['Minimum Intensity']:.4f}"
        )

        st.markdown("---")

        st.subheader("Imaging Risk Assessment")

        st.metric(
            "Risk Score",
            f"{risk_score}/100"
        )

        if risk_level == "Low Risk":

            st.success(risk_level)

        elif risk_level == "Moderate Risk":

            st.warning(risk_level)

        else:

            st.error(risk_level)

        st.info(interpretation)

    # =====================================================
    # RIGHT COLUMN
    # =====================================================

    with col2:

        st.subheader("Clinical Prediction")

        st.metric(
            "Predicted Stage",
            stage
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

    st.markdown("---")

    st.subheader("Medical Interpretation")

    st.info(
        f"""
### Imaging Analysis

The pancreas was successfully segmented from the uploaded CT scan.

Quantitative imaging biomarkers were extracted from the segmented region.

Imaging Risk Score : **{risk_score}/100**

Risk Category : **{risk_level}**

---

### Clinical Prediction

Predicted Stage : **{stage}**

Prediction Confidence : **{confidence:.2f}%**

---

This report combines Deep Learning–based image analysis with Machine Learning–based clinical stage prediction to support pancreatic disease assessment.

**Note:** The Imaging Risk Score is a rule-based assessment derived from extracted imaging biomarkers and is intended as supportive information. It is not a clinical diagnosis.
"""
    )

    st.markdown("---")

    st.download_button(
        "📥 Download Report",
        data=f"""
===============================
PANCREATIC CANCER REPORT
===============================

IMAGING BIOMARKERS

Volume : {features['Volume']:.2f}

Mean Intensity : {features['Mean Intensity']:.4f}

Standard Deviation : {features['Standard Deviation']:.4f}

Maximum Intensity : {features['Maximum Intensity']:.4f}

Minimum Intensity : {features['Minimum Intensity']:.4f}

-------------------------------

IMAGING RISK

Risk Score : {risk_score}/100

Risk Level : {risk_level}

-------------------------------

CLINICAL PREDICTION

Predicted Stage : {stage}

Confidence : {confidence:.2f}%

===============================
""",
        file_name="patient_report.txt",
        mime="text/plain"
    )