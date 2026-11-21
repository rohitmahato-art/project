import os

import gradio as gr
import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "house_price_xgb_model.joblib")
model = joblib.load(MODEL_PATH)

# The exact 93 training columns (and their order) are stored inside the model,
# so we read them from there instead of hard-coding them.
FEATURES = list(model.get_booster().feature_names)

# Set to True ONLY if you trained on a log-transformed price (np.log / np.log1p).
LOG_TARGET = False
CURRENCY = "₹"  # change/remove if your price column is in another unit

# Categorical columns were one-hot encoded (pd.get_dummies, drop_first=True), so
# one category per column was dropped. Picking the "reference" option leaves all
# of that column's dummies at 0, which is exactly how the model saw it in training.
# (label, column prefix, reference-category label, default choice)
CATEGORICAL = [
    ("Status", "Status_", "Other status (reference)", None),
    ("Transaction", "Transaction_", "New Property (reference)", "Resale"),
    ("Furnishing", "Furnishing_", "Furnished (reference)", None),
    ("Facing", "facing_", "East (reference)", None),
    ("Overlooking", "overlooking_", "Garden/Park (reference)", None),
    ("Ownership", "Ownership_", "Co-operative Society (reference)", "Freehold"),
    ("Parking type", "parking_type_", "Covered (reference)", None),
    ("Location", "location_grouped_", "Other city (reference)", "mumbai"),
]

NUMERIC = [
    # (column, label, default, minimum)
    ("Area_sqft", "Area (sq ft)", 1000, 1),
    ("Bathroom", "Bathrooms", 2, 0),
    ("Balcony", "Balconies", 1, 0),
    ("floor_num", "Floor number", 2, 0),
    ("total_floors", "Total floors in building", 5, 1),
    ("parking_count", "Parking spots", 1, 0),
]


def categories_for(prefix):
    return [c[len(prefix):] for c in FEATURES if c.startswith(prefix)]


def build_row(numeric_values, categorical_values):
    row = {c: 0.0 for c in FEATURES}
    for (col, *_), val in zip(NUMERIC, numeric_values):
        row[col] = float(val or 0)
    for (_, prefix, reference, _), val in zip(CATEGORICAL, categorical_values):
        col = prefix + str(val)
        if val != reference and col in row:
            row[col] = 1.0
    return pd.DataFrame([row], columns=FEATURES).astype("float32")


def predict(*values):
    n = len(NUMERIC)
    X = build_row(values[:n], values[n:])
    pred = float(model.predict(X)[0])
    if LOG_TARGET:
        import numpy as np

        pred = float(np.expm1(pred))
    return f"{CURRENCY}{max(pred, 0):,.0f}"


inputs = []
for col, label, default, minimum in NUMERIC:
    inputs.append(gr.Number(label=label, value=default, minimum=minimum, precision=0))

for label, prefix, reference, default in CATEGORICAL:
    choices = [reference] + categories_for(prefix)
    value = default if default in choices else reference
    inputs.append(gr.Dropdown(choices=choices, value=value, label=label))

demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=gr.Textbox(label="Estimated price"),
    title="House Price Predictor",
    description="Enter the property details to get an estimated price from the XGBoost model.",
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
