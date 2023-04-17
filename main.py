import pandas as pd
import numpy as np
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import LabelEncoder


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    user_est_true = dict()
    for uid, _, true_r, est, _ in predictions:
        current = user_est_true.get(uid, [])
        current.append((est, true_r))
        user_est_true[uid] = current

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
    return precisions, recalls

def get_top_n(predictions, n=5):
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid not in top_n:
            top_n[uid] = [(iid, est)]  # Remove the true_r here
        else:
            top_n[uid].append((iid, est))  # Remove the true_r here
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n


def f1_score(precisions, recalls):
    f1_scores = dict()
    for uid in precisions:
        precision = precisions[uid]
        recall = recalls[uid]
        if precision + recall == 0:
            f1_scores[uid] = 0
        else:
            f1_scores[uid] = 2 * (precision * recall) / (precision + recall)
    return f1_scores

def average_metric(metric_dict):
    return sum(metric_dict.values()) / len(metric_dict)

# Load the dataset
data = pd.read_csv("beer_reviews.csv")

# Preprocess the data
data.dropna(subset=['review_profilename', 'beer_name', 'beer_style'], inplace=True)
data['beer_abv'].fillna(data['beer_abv'].mean(), inplace=True)

# Create a dataset with user, beer, and review_overall columns
ratings_data = data[['review_profilename', 'beer_name', 'review_overall']]

# Instantiate a Surprise Reader and load the dataset
reader = Reader(rating_scale=(1, 5))
surprise_data = Dataset.load_from_df(ratings_data, reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(surprise_data, test_size=0.2, random_state=42)

# Train the SVD model for collaborative filtering
algo = SVD()
algo.fit(trainset)

# Test the SVD model
predictions = algo.test(testset)

# Calculate the evaluation metrics
accuracy_score = accuracy.rmse(predictions)
precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=3.5)
precision = average_metric(precisions)
recall = average_metric(recalls)

# Calculate F1 score for each user and average F1 score
f1_scores = f1_score(precisions, recalls)
avg_f1_score = average_metric(f1_scores)

print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", avg_f1_score)
print("Collaborative Filtering Accuracy (RMSE): ", accuracy_score)

# Content-based filtering
beer_data = data[['beer_name', 'beer_style']].drop_duplicates(subset='beer_name')
beer_data.reset_index(drop=True, inplace=True)  # Add this line to reset the index
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(beer_data['beer_style'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(beer_data.index, index=beer_data['beer_name'])  # Update this line to use the new index



def content_based_recommendations(beer_name, n_recommendations=2):  # Change n_recommendations to 2
    # Check if the beer is in the indices
    if beer_name not in indices:
        return []
    # Get the index of the beer
    idx = indices[beer_name]
    # Check if the index is within the bounds of the cosine similarity matrix
    if idx >= len(cosine_sim):
        return []
    # Get the pairwise similarity scores of all beers with the given beer
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the beers based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the n most similar beers
    sim_scores = sim_scores[1:n_recommendations + 1]
    # Get the beer indices
    beer_indices = [i[0] for i in sim_scores]
    # Return the top n most similar beers
    return beer_data['beer_name'].iloc[beer_indices].values



# Get user input
user_profile = input("Enter a profile name: ")

# Get the top 5 beer recommendations for the user using collaborative filtering
user_predictions = [pred for pred in predictions if pred[0] == user_profile]
top_beers_collab = get_top_n(user_predictions, n=5)

# If the user has no collaborative filtering recommendations, use the top-rated beers instead
if user_profile not in top_beers_collab:
    top_beers_collab[user_profile] = ratings_data.groupby('beer_name')['review_overall'].mean().sort_values(ascending=False).head(5).reset_index().apply(lambda x: (x['beer_name'], x['review_overall']), axis=1).tolist()

# Get content-based recommendations for each beer
top_beers_content = []
for beer, _ in top_beers_collab[user_profile]:
    top_beers_content.extend(content_based_recommendations(beer))

# Combine collaborative and content-based recommendations
top_beers_hybrid = list(set(top_beers_content))

# Add more recommendations if necessary
while len(top_beers_hybrid) < 10:
    top_beer = ratings_data.groupby('beer_name')['review_overall'].mean().sort_values(ascending=False).reset_index().loc[len(top_beers_hybrid), 'beer_name']
    if top_beer not in top_beers_hybrid:
        top_beers_hybrid.append(top_beer)

print("\nTop 10 beer recommendations for you (In order from most to least):")
for i, beer in enumerate(top_beers_hybrid, 1):
    print(f"{i}. {beer}")

