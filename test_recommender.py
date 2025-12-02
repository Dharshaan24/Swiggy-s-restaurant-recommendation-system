from recommender import Recommender

rec = Recommender()

city = "Abohar"
cuisine = "Chinese"
rating = 4.0
cost = 300

user_vec = rec.encode_user_input(city, cuisine, rating, cost)

# COSINE RECOMMENDATION
results = rec.recommend_cosine(user_vec, top_k=5)

print(results)
