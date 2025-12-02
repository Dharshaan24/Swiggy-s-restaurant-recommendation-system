import pandas as pd

# Read your dataset
df = pd.read_csv("D:\\downloads\\Swiggy’s Restaurant Recommendation System\\swiggy.csv")

# -------------------------------
# 1. Remove missing restaurant names
# -------------------------------
df = df.dropna(subset=["name"])

# -------------------------------
# 2. Clean Rating column
# -------------------------------
df["rating"] = df["rating"].replace("--", None)
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["rating"] = df["rating"].fillna(df["rating"].median())

# -------------------------------
# 3. Clean Rating Count column
# -------------------------------
def clean_rating_count(x):
    x = str(x)

    if "Too Few" in x:
        return 0

    # Handle values like "1K+ ratings"
    if "K" in x:
        try:
            num = float(x.split("K")[0])
            return int(num * 1000)
        except:
            return 0

    # Handle values like "50+ ratings"
    if "+" in x:
        try:
            return int(x.split("+")[0])
        except:
            return 0

    try:
        return int(x)
    except:
        return 0

df["rating_count"] = df["rating_count"].apply(clean_rating_count)
df["rating_count"] = df["rating_count"].fillna(0)

# -------------------------------
# 4. Clean Cost column
# -------------------------------
def clean_cost(x):
    x = str(x).replace("₹", "").replace(" ", "")
    return pd.to_numeric(x, errors="coerce")

df["cost"] = df["cost"].apply(clean_cost)
df["cost"] = df["cost"].fillna(df["cost"].median())

# -------------------------------
# 5. Reset index
# -------------------------------
df = df.reset_index(drop=True)

# -------------------------------
# 6. Save cleaned data
# -------------------------------
df.to_csv("cleaned_data.csv", index=False)

print("Cleaning Completed!")
print("Cleaned file saved as: cleaned_data.csv")
