# Crop Yield Prediction & Precision Agronomy

A Streamlit prototype dashboard for crop-yield prediction and precision-agriculture recommendations.

## Features
- Crop and field input dashboard
- Prototype yield prediction
- Yield-risk classification
- Soil, weather and vegetation health indicators
- Irrigation recommendation
- Nutrient recommendation
- Explainable feature-contribution chart
- Historical yield trend

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Next step for the academic project

Replace the transparent prototype formula in `app.py` with a trained ML model such as Random Forest or XGBoost.

Recommended final pipeline:

Data → Cleaning → Feature Engineering → ML Model → Yield Prediction → Explainability → Precision Agronomy Recommendations
