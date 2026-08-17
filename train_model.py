import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# PATHS
# =========================================================

DATA_PATH = r"C:\Users\sujin\Downloads\archive\crop_yield_dataset.csv"

MODEL_PATH = "models/crop_yield_model.pkl"
METRICS_PATH = "models/model_metrics.pkl"
IMPORTANCE_PATH = "models/feature_importance.pkl"
TEST_DATA_PATH = "models/test_predictions.csv"


# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# =========================================================
# FEATURES AND TARGET
# =========================================================

target = "Yield_ton_per_ha"

X = df.drop(columns=[target])
y = df[target]


# =========================================================
# IDENTIFY COLUMNS
# =========================================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


# =========================================================
# PREPROCESSING
# =========================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])


# =========================================================
# RANDOM FOREST
# =========================================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


# =========================================================
# COMPLETE PIPELINE
# =========================================================

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# TRAIN
# =========================================================

print("\nTraining Random Forest...")

pipeline.fit(X_train, y_train)

print("Training completed!")


# =========================================================
# PREDICTION
# =========================================================

y_pred = pipeline.predict(X_test)


# =========================================================
# MODEL PERFORMANCE
# =========================================================

mae = mean_absolute_error(y_test, y_pred)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R2   : {r2:.3f}")


# =========================================================
# SAVE MODEL
# =========================================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    pipeline,
    MODEL_PATH
)

print("\nModel saved successfully!")
print("Location:", MODEL_PATH)


# =========================================================
# SAVE METRICS
# =========================================================

metrics = {
    "MAE": round(mae, 3),
    "RMSE": round(rmse, 3),
    "R2": round(r2, 3)
}

joblib.dump(
    metrics,
    METRICS_PATH
)


# =========================================================
# SAVE TEST PREDICTIONS
# =========================================================

test_predictions = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

test_predictions.to_csv(
    TEST_DATA_PATH,
    index=False
)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

rf_model = pipeline.named_steps["model"]
preprocessor_fitted = pipeline.named_steps["preprocessor"]

feature_names = preprocessor_fitted.get_feature_names_out()

importance_values = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

feature_importance["Feature"] = (
    feature_importance["Feature"]
    .str.replace("num__", "", regex=False)
    .str.replace("cat__", "", regex=False)
)

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

# Keep top 15 features
feature_importance = feature_importance.head(15)

joblib.dump(
    feature_importance,
    IMPORTANCE_PATH
)


print("\nModel analysis files saved:")
print("-", METRICS_PATH)
print("-", TEST_DATA_PATH)
print("-", IMPORTANCE_PATH)

print("\nTop Features:")
print(feature_importance.head(10).to_string(index=False))