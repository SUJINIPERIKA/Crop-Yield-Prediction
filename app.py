import streamlit as st
import pandas as pd
import joblib
import os

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Crop Yield & Precision Agronomy",
    page_icon="🌾",
    layout="wide"
)

# =========================================================
# CUSTOM STYLE
# =========================================================

st.markdown("""
<style>
    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 17px;
        color: #666;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
    }

    .info-card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        background: #fafafa;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

MODEL_PATH = "models/crop_yield_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Unable to load trained model: {e}")
    st.stop()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🌾 Crop Yield Prediction & Precision Agronomy</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered field monitoring, yield prediction and decision-support system</div>',
    unsafe_allow_html=True
)

st.markdown("""
**Machine Learning:** Random Forest Regressor  
**Dataset:** 10,000 agricultural records  
**Goal:** Predict crop yield and provide field-management insights
""")


# =========================================================
# SIDEBAR INPUTS
# =========================================================

st.sidebar.header("🌱 Field Inputs")

crop = st.sidebar.selectbox(
    "Crop",
    ["Maize", "Barley", "Rice", "Wheat", "Cotton", "Soybean"]
)

region = st.sidebar.selectbox(
    "Region",
    ["Region_A", "Region_B", "Region_C", "Region_D"]
)

soil = st.sidebar.selectbox(
    "Soil Type",
    ["Sandy", "Loam", "Clay", "Silt", "Peat"]
)

soil_ph = st.sidebar.slider(
    "Soil pH",
    4.0, 9.0, 6.5, 0.01
)

rainfall = st.sidebar.slider(
    "Rainfall (mm)",
    100.0, 2000.0, 700.0, 10.0
)

temperature = st.sidebar.slider(
    "Temperature (°C)",
    10.0, 45.0, 30.0, 0.5
)

humidity = st.sidebar.slider(
    "Humidity (%)",
    20.0, 90.0, 55.0, 0.5
)

fertilizer = st.sidebar.slider(
    "Fertilizer Used (kg)",
    0.0, 300.0, 100.0, 1.0
)

irrigation = st.sidebar.selectbox(
    "Irrigation",
    ["Drip", "Sprinkler", "Flood", "Rainfed"]
)

pesticides = st.sidebar.slider(
    "Pesticides Used (kg)",
    0.0, 50.0, 10.0, 0.5
)

planting_density = st.sidebar.slider(
    "Planting Density",
    1.0, 30.0, 15.0, 0.1
)

previous_crop = st.sidebar.selectbox(
    "Previous Crop",
    ["Rice", "Wheat", "Maize", "Barley", "Cotton", "Soybean"]
)


# =========================================================
# INPUT DATAFRAME
# =========================================================

input_data = pd.DataFrame({
    "Crop": [crop],
    "Region": [region],
    "Soil_Type": [soil],
    "Soil_pH": [soil_ph],
    "Rainfall_mm": [rainfall],
    "Temperature_C": [temperature],
    "Humidity_pct": [humidity],
    "Fertilizer_Used_kg": [fertilizer],
    "Irrigation": [irrigation],
    "Pesticides_Used_kg": [pesticides],
    "Planting_Density": [planting_density],
    "Previous_Crop": [previous_crop]
})


# =========================================================
# REAL ML PREDICTION
# =========================================================

yield_pred = float(model.predict(input_data)[0])
yield_pred = round(max(0, yield_pred), 2)


# =========================================================
# FIELD CONDITION STATUS
# =========================================================

if humidity < 40:
    humidity_status = "🔴 Low"
elif humidity < 70:
    humidity_status = "🟢 Suitable"
else:
    humidity_status = "🟡 High"

if rainfall < 400:
    rain_status = "🔴 Low"
elif rainfall <= 1200:
    rain_status = "🟢 Good"
else:
    rain_status = "🟡 High"

if temperature > 35:
    temp_status = "🔴 High"
elif temperature >= 20:
    temp_status = "🟢 Suitable"
else:
    temp_status = "🟡 Low"

if soil_ph < 5.5:
    ph_status = "🔴 Acidic"
elif soil_ph <= 7.5:
    ph_status = "🟢 Suitable"
else:
    ph_status = "🟡 Alkaline"

if fertilizer < 50:
    fertilizer_status = "🔴 Low"
elif fertilizer < 150:
    fertilizer_status = "🟡 Moderate"
else:
    fertilizer_status = "🟢 High"


# =========================================================
# YIELD RISK
# =========================================================

risk_points = sum([
    humidity < 35,
    rainfall < 400 or rainfall > 1200,
    temperature > 35,
    soil_ph < 5.0 or soil_ph > 8.0,
    fertilizer < 40
])

if risk_points >= 3:
    risk = "High"
elif risk_points >= 1:
    risk = "Medium"
else:
    risk = "Low"


# =========================================================
# TOP METRICS
# =========================================================

st.divider()

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "🌾 Predicted Yield",
    f"{yield_pred} t/ha"
)

c2.metric(
    "⚠️ Yield Risk",
    risk
)

c3.metric(
    "🧪 Soil pH",
    f"{soil_ph:.2f}"
)

c4.metric(
    "🌱 Crop",
    crop
)


# =========================================================
# FIELD HEALTH + INPUT OVERVIEW
# =========================================================

st.divider()

left, right = st.columns([1.15, 1])

with left:

    st.subheader("🌱 Field Health")

    health = pd.DataFrame({
        "Parameter": [
            "Soil pH",
            "Rainfall",
            "Temperature",
            "Humidity",
            "Fertilizer"
        ],
        "Status": [
            ph_status,
            rain_status,
            temp_status,
            humidity_status,
            fertilizer_status
        ]
    })

    st.table(health)


with right:

    st.subheader("📊 Field Input Overview")

    factors = pd.DataFrame({
        "Feature": [
            "Rainfall",
            "Temperature",
            "Humidity",
            "Fertilizer",
            "Pesticides"
        ],
        "Value": [
            rainfall,
            temperature,
            humidity,
            fertilizer,
            pesticides
        ]
    })

    st.bar_chart(
        factors.set_index("Feature")
    )


# =========================================================
# PRECISION AGRONOMY
# =========================================================

st.divider()

st.subheader("🎯 Precision Agronomy Recommendations")

r1, r2, r3 = st.columns(3)

with r1:

    st.markdown("### 💧 Water Management")

    if rainfall < 400:
        st.error(
            "Low rainfall detected. Consider supplemental irrigation."
        )
    elif rainfall > 1200:
        st.warning(
            "High rainfall detected. Monitor drainage and waterlogging."
        )
    else:
        st.success(
            "Rainfall level is within a suitable range."
        )


with r2:

    st.markdown("### 🧪 Soil & Nutrients")

    if soil_ph < 5.5:
        st.warning(
            "Soil is acidic. Consider soil-test-based pH management."
        )
    elif soil_ph > 7.5:
        st.warning(
            "Soil is alkaline. Monitor nutrient availability."
        )
    elif fertilizer < 50:
        st.warning(
            "Fertilizer level is low. Consider soil-test-based nutrient management."
        )
    else:
        st.success(
            "Soil and nutrient conditions are suitable."
        )


with r3:

    st.markdown("### ⚠️ Risk Alert")

    if risk == "High":
        st.error(
            "High-risk field conditions detected."
        )
    elif risk == "Medium":
        st.warning(
            "Some field conditions need monitoring."
        )
    else:
        st.success(
            "Overall field conditions look suitable."
        )


# =========================================================
# PREDICTION DETAILS
# =========================================================

st.divider()

st.subheader("🔍 Prediction Details")

st.write(
    "The predicted yield is generated using the trained "
    "Random Forest regression model."
)

prediction_table = pd.DataFrame({
    "Input": [
        "Crop",
        "Region",
        "Soil Type",
        "Soil pH",
        "Rainfall",
        "Temperature",
        "Humidity",
        "Fertilizer",
        "Irrigation",
        "Pesticides",
        "Planting Density",
        "Previous Crop"
    ],
    "Value": [
        crop,
        region,
        soil,
        soil_ph,
        rainfall,
        temperature,
        humidity,
        fertilizer,
        irrigation,
        pesticides,
        planting_density,
        previous_crop
    ]
})

st.dataframe(
    prediction_table,
    width="stretch",
    hide_index=True
)

st.success(
    f"🌾 Estimated Crop Yield: **{yield_pred} tons per hectare**"
)


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.divider()

st.subheader("📊 Machine Learning Model Performance")

METRICS_PATH = "models/model_metrics.pkl"

if os.path.exists(METRICS_PATH):

    metrics = joblib.load(METRICS_PATH)

    p1, p2, p3 = st.columns(3)

    p1.metric(
        "MAE",
        f"{metrics['MAE']:.3f}"
    )

    p2.metric(
        "RMSE",
        f"{metrics['RMSE']:.3f}"
    )

    p3.metric(
        "R² Score",
        f"{metrics['R2']:.3f}"
    )

    st.caption(
        "Lower MAE/RMSE indicate lower prediction error. "
        "Higher R² indicates better model fit."
    )

else:

    st.warning(
        "Model performance file not found. Run train_model.py first."
    )


# =========================================================
# ACTUAL VS PREDICTED
# =========================================================

st.divider()

st.subheader("📈 Actual vs Predicted Yield")

TEST_DATA_PATH = "models/test_predictions.csv"

if os.path.exists(TEST_DATA_PATH):

    test_results = pd.read_csv(TEST_DATA_PATH)

    # Use first 100 samples for dashboard visualization
    chart_data = test_results.head(100).copy()

    # Create clean comparison dataframe
    comparison = pd.DataFrame({
        "Actual Yield": chart_data["Actual"].values,
        "Predicted Yield": chart_data["Predicted"].values
    })

    st.line_chart(
        comparison,
        width="stretch"
    )

    # Error analysis
    chart_data["Absolute Error"] = (
        chart_data["Actual"] -
        chart_data["Predicted"]
    ).abs()

    e1, e2, e3 = st.columns(3)

    e1.metric(
        "Mean Error",
        f"{chart_data['Absolute Error'].mean():.2f}"
    )

    e2.metric(
        "Minimum Error",
        f"{chart_data['Absolute Error'].min():.2f}"
    )

    e3.metric(
        "Maximum Error",
        f"{chart_data['Absolute Error'].max():.2f}"
    )

    st.caption(
        "Comparison between actual and Random Forest predicted "
        "yield values for test samples."
    )

else:

    st.warning(
        "Test prediction file not found."
    )


# =========================================================
# EXPLAINABLE AI - FEATURE IMPORTANCE
# =========================================================

st.divider()

st.subheader("🧠 Explainable AI — What Drives the Prediction?")

st.caption(
    "Feature importance shows which input variables the Random Forest "
    "model relied on most when generating predictions. "
    "Importance does not imply causation."
)

IMPORTANCE_PATH = "models/feature_importance.pkl"

if os.path.exists(IMPORTANCE_PATH):

    importance = joblib.load(IMPORTANCE_PATH)

    importance = importance.copy()

    # Top 10 features
    top10 = importance.sort_values(
        "Importance",
        ascending=False
    ).head(10)

    # Chart requires ascending order for clean display
    chart_importance = top10.sort_values(
        "Importance",
        ascending=True
    )

    st.bar_chart(
        chart_importance.set_index("Feature")["Importance"],
        width="stretch"
    )

    st.write("### 🏆 Top Influential Features")

    display_importance = top10.copy()

    display_importance["Importance"] = (
        display_importance["Importance"].round(4)
    )

    st.dataframe(
        display_importance,
        width="stretch",
        hide_index=True
    )

    # Highlight strongest feature
    strongest_feature = top10.iloc[0]["Feature"]
    strongest_value = top10.iloc[0]["Importance"]

    st.info(
        f"🔎 **Most influential feature:** {strongest_feature} "
        f"with importance score {strongest_value:.4f}."
    )

else:

    st.warning(
        "Feature importance file not found."
    )


# =========================================================
# PROJECT SUMMARY
# =========================================================

st.divider()

st.subheader("🌾 Project Summary")

s1, s2, s3 = st.columns(3)

with s1:

    st.markdown("### 📥 Input")

    st.write(
        "Crop, region, soil properties, weather conditions, "
        "fertilizer, irrigation and crop-management information."
    )


with s2:

    st.markdown("### 🤖 AI Model")

    st.write(
        "Random Forest Regression trained on "
        "10,000 agricultural records."
    )


with s3:

    st.markdown("### 📤 Output")

    st.write(
        "Predicted crop yield, field risk assessment, "
        "field health and precision agronomy recommendations."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.info(
    "🌾 Model: Random Forest Regressor | "
    "Training dataset: 10,000 records | "
    "Target: Yield_ton_per_ha"
)