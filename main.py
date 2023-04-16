import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
data = pd.read_csv("beer_reviews.csv")

# Clean and preprocess the data
data.dropna(subset=['brewery_name', 'review_profilename', 'beer_name'], inplace=True)
data['beer_abv'].fillna(data['beer_abv'].mean(), inplace=True)

# Select relevant features
features = ['brewery_name', 'review_aroma', 'review_appearance', 'beer_style', 'review_palate', 'review_taste',
            'beer_abv']
X = data[features]
y = data['beer_name']

# Encode categorical data
categorical_features = ['brewery_name', 'beer_style']
encoders = {}

for feature in categorical_features:
    encoder = LabelEncoder()
    X.loc[:, feature] = encoder.fit_transform(X[feature])
    encoders[feature] = encoder

# Normalize the data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


# Function to generate three random beer names with good review_overall values so user can recognize them easier
def get_random_beers(num_beers=3, min_review_overall=4.0):
    good_beers = data[data['review_overall'] >= min_review_overall]
    random_indices = random.sample(range(len(good_beers)), num_beers)
    beer_choices = good_beers.iloc[random_indices]['beer_name'].values
    return beer_choices


# Function to ask the user to select two beers
def get_user_choices():
    beer_choices = get_random_beers()
    selected_beers = []

    while len(selected_beers) < 2:
        print("\nChoose a beer from the following list (or type 'none' if you haven't heard of any):")
        for i, beer in enumerate(beer_choices):
            print(f"{i + 1}. {beer}")

        choice = input("Enter the number of the beer or 'none': ")
        if choice.lower() == 'none':
            beer_choices = get_random_beers()
        elif choice.isdigit() and 1 <= int(choice) <= len(beer_choices):
            selected_beers.append(beer_choices[int(choice) - 1])
            if len(selected_beers) < 2:
                print(f"\nYou selected {selected_beers[0]}. Now pick another beer.")
                beer_choices = get_random_beers()
        else:
            print("Invalid input. Please try again.")

    return selected_beers


# Get user choices
chosen_beers = get_user_choices()

# Get the chosen beers' data
chosen_beers_data = data[data['beer_name'].isin(chosen_beers)]

# Calculate the average for 'beer_abv'
avg_features = pd.Series(dtype=object)
avg_features['beer_abv'] = chosen_beers_data['beer_abv'].mean()

# Get the most common brewery from the chosen beers
avg_features['brewery_name'] = chosen_beers_data['brewery_name'].mode()[0]

# Add dummy values for the other features
avg_features['review_aroma'] = data['review_aroma'].mean()
avg_features['review_appearance'] = data['review_appearance'].mean()
avg_features['beer_style'] = encoders['beer_style'].transform([data['beer_style'].mode()[0]])[0]
avg_features['review_palate'] = data['review_palate'].mean()
avg_features['review_taste'] = data['review_taste'].mean()

# Encode the brewery_name
avg_features['brewery_name'] = encoders['brewery_name'].transform([avg_features['brewery_name']])[0]

# Normalize the average features
avg_features_df = pd.DataFrame([avg_features], columns=features)
avg_features_transformed = scaler.transform(avg_features_df)

# Get the nearest neighbors
distances, neighbors = knn.kneighbors(avg_features_transformed, return_distance=True)

# Sort neighbors by distances
sorted_neighbors = sorted(zip(neighbors[0], distances[0]), key=lambda x: x[1])

# Print recommendations
print("\nBeers you may also like (strongest recommendation first):")
for i, (idx, distance) in enumerate(sorted_neighbors, 1):
    # Calculate accuracy from distance
    accuracy = 1 / (1 + distance)
    accuracy_percentage = accuracy * 100
    print(f"{i}. {data.iloc[idx]['beer_name']} (Accuracy: {accuracy_percentage:.2f}%)")

