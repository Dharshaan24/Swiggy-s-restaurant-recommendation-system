import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
import joblib

# -------------------------------
# 1. Load Cleaned Data
# -------------------------------
df = pd.read_csv("D:\downloads\Swiggy’s Restaurant Recommendation System\cleaned_data.csv")

# -------------------------------
# 2. One-Hot Encode City Column
# -------------------------------
city_dummies = pd.get_dummies(df["city"], prefix="city")

# -------------------------------
# 3. Multi-Hot Encode Cuisine Column
# -------------------------------
# Split cuisines: "North Indian,Chinese" → ["north indian","chinese"]
def split_cuisine(x):
    if pd.isna(x):
        return []
    return [item.strip().lower() for item in str(x).split(",")]

cuisine_list = df["cuisine"].apply(split_cuisine)

mlb = MultiLabelBinarizer()
cuisine_encoded = pd.DataFrame(
    mlb.fit_transform(cuisine_list),
    columns=[f"cuisine_{c}" for c in mlb.classes_]
)

# -------------------------------
# 4. Select Numerical Columns
# -------------------------------
numeric_cols = df[["rating", "rating_count", "cost"]]

# -------------------------------
# 5. Scale Numerical Features
# -------------------------------
scaler = MinMaxScaler()
scaled_numeric = pd.DataFrame(
    scaler.fit_transform(numeric_cols),
    columns=numeric_cols.columns
)

# -------------------------------
# 6. Combine Everything into One Encoded Dataset
# -------------------------------
encoded_df = pd.concat([scaled_numeric, city_dummies, cuisine_encoded], axis=1)

# -------------------------------
# 7. Save Outputs (Very Important)
# -------------------------------
encoded_df.to_csv("encoded_data.csv", index=False)

# Save the encoder and scaler to reuse in Streamlit
joblib.dump(
    {
        "mlb": mlb,               # MultiLabel Encoder for cuisine
        "scaler": scaler,         # Scaler for numeric columns
        "city_columns": list(city_dummies.columns)  # For future encoding
    },
    "encoder.pkl"
)

print("Preprocessing Completed!")
print("Files saved: encoded_data.csv and encoder.pkl")
