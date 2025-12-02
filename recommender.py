import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import joblib

class Recommender:

    def __init__(self):
        # Load required files
        self.cleaned = pd.read_csv("D:\downloads\Swiggy’s Restaurant Recommendation System\cleaned_data.csv")
        self.encoded = pd.read_csv("D:\downloads\Swiggy’s Restaurant Recommendation System\encoded_data.csv")
        enc = joblib.load("encoder.pkl")

        # Load encoding tools
        self.mlb = enc["mlb"]
        self.scaler = enc["scaler"]
        self.city_columns = enc["city_columns"]

        # Save encoded columns
        self.columns = self.encoded.columns.tolist()


    # --------------------------------------------
    # Convert user input into encoded numeric vector
    # --------------------------------------------
    def encode_user_input(self, city, cuisine, rating, cost):

        # Create empty vector with all encoded columns
        v = pd.Series(0, index=self.columns, dtype=float)

        # ---- FIXED SCALER WARNING ----
        # Prepare dataframe with correct feature names
        num_df = pd.DataFrame(
            [[rating, 0, cost]],
            columns=["rating", "rating_count", "cost"]
        )

        # Scale numeric values using same scaler as training
        scaled = self.scaler.transform(num_df)[0]

        v["rating"] = scaled[0]
        v["rating_count"] = scaled[1]   # Always zero for user input
        v["cost"] = scaled[2]

        # ---- Encode City ----
        city_col = f"city_{city}"
        if city_col in v.index:
            v[city_col] = 1.0

        # ---- Encode Cuisine ----
        cuisines = [c.strip().lower() for c in cuisine.split(",") if c.strip()]

        if cuisines:
            enc = self.mlb.transform([cuisines])[0]
            for label, value in zip(self.mlb.classes_, enc):
                col_name = f"cuisine_{label}"
                if col_name in v.index:
                    v[col_name] = value

        return v.values


    # --------------------------------------------
    # METHOD 1 — COSINE SIMILARITY RECOMMENDER
    # --------------------------------------------
    def recommend_cosine(self, user_vector, top_k=10):
        sims = cosine_similarity([user_vector], self.encoded.values)[0]
        top_idx = sims.argsort()[::-1][:top_k]

        results = self.cleaned.iloc[top_idx].copy()
        results["similarity"] = sims[top_idx]

        return results


    # --------------------------------------------
    # METHOD 2 — KMEANS CLUSTERING RECOMMENDER
    # --------------------------------------------
    def fit_kmeans(self, n_clusters=10):
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.kmeans.fit(self.encoded.values)

    def recommend_kmeans(self, user_vector, top_k=10):
        cluster = self.kmeans.predict([user_vector])[0]
        idxs = np.where(self.kmeans.labels_ == cluster)[0]

        sims = cosine_similarity([user_vector], self.encoded.values[idxs])[0]
        top_local = sims.argsort()[::-1][:top_k]

        return self.cleaned.iloc[idxs[top_local]]
