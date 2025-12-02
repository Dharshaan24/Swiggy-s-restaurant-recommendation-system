import streamlit as st
import pandas as pd
import joblib
from recommender import Recommender

# Title
st.title("🍽️ Swiggy Restaurant Recommendation System")

# Load recommender
rec = Recommender()

# Load cleaned data for city dropdown
df_clean = pd.read_csv("cleaned_data.csv")
cities = sorted(df_clean["city"].dropna().unique())

# -------------------------------
# USER INPUT SIDEBAR
# -------------------------------
st.sidebar.header("User Preferences")

city = st.sidebar.selectbox(
    "Select City",
    cities
)

cuisine = st.sidebar.text_input(
    "Enter Cuisine (e.g., Chinese, Fast Food, Pizza)"
)

rating = st.sidebar.slider(
    "Minimum Rating",
    0.0, 5.0, 3.5
)

cost = st.sidebar.slider(
    "Approx. Cost for Two",
    50, 2000, 300
)

top_k = st.sidebar.slider(
    "Number of Recommendations",
    1, 20, 5
)

algo = st.sidebar.radio(
    "Choose Recommendation Method",
    ("Cosine Similarity", "KMeans")
)

# -------------------------------
# BUTTON — GENERATE RESULTS
# -------------------------------
if st.sidebar.button("Get Recommendations"):
    
    st.subheader("Top Recommended Restaurants 🍛")

    # Encode user input
    user_vec = rec.encode_user_input(city, cuisine, rating, cost)

    # Choose algorithm
    if algo == "Cosine Similarity":
        results = rec.recommend_cosine(user_vec, top_k)
    else:
        try:
            rec.fit_kmeans(n_clusters=10)
            results = rec.recommend_kmeans(user_vec, top_k)
        except:
            st.error("KMeans training failed.")

    # Display results
    for idx, row in results.iterrows():
        st.write(f"### 🍴 {row['name']} ({row['city']})")
        st.write(f"⭐ Rating: {row['rating']}")
        st.write(f"💰 Cost: ₹{row['cost']}")
        st.write(f"📍 Address: {row['address']}")
        st.write(f"[🔗 View on Swiggy]({row['link']})")
        st.write("---")
