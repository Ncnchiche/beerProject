<h1 align="center">Beer Recommendation Project</h1>


## What it does

This beer recommendation system is a Python script that uses a K-Nearest Neighbors (KNN) classifier to suggest beers that a user might enjoy based on their preferences.

When the code is executed, the user is prompted to choose two beers they enjoy from a list of options. The program then calculates the average values of the chosen beers' features, encodes and normalizes them, and passes them to the trained KNN model. The KNN model then finds the k nearest neighbors to the average features and recommends those beers to the user. The program also includes a function to generate three random beers with good overall review ratings to help the user choose their initial beer selections.

Overall, the beer recommendation system is a fun and interactive way for beer lovers to discover new beers they may enjoy based on their preferences.


## How it works?
This program is a beer recommendation system based on user input of two beers they enjoy. It uses a dataset of beer reviews to train a K-Nearest Neighbors (KNN) machine learning model that can predict which beers the user may also like. The KNN model takes into account various features of beers, such as the brewery name, beer style, alcohol content (ABV), and other review metrics like taste and aroma. 



## How Was it Made?

The beer recommendation system was made using Python programming language and several libraries which are included in the requirments(Pandas, NumPy, scikit-learn, and random)

### Requirements

- Python 3.6 or newer
- pandas
- numpy
- scikit-learn

## Installation

1. Download or clone the repository.

2. Add Data Set To your repository.
[beer_reviews.csv](https://data.world/socialmediadata/beeradvocate/workspace/file?filename=beer_reviews.csv)

2. Create a virtual environment (optional, but recommended):

3. Activate and install required libraries using requirments.txt
```python
pip install -r requirements.txt
```

4. Run the main.py file and have fun
