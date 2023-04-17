<h1 align="center">Beer Recommender System</h1>


## What is this?

This beer recommender system combines collaborative filtering and content-based filtering techniques to provide personalized beer recommendations for beer lovers, helping them discover new beers based on their preferences.


## How it works?

The recommender system utilizes two techniques:

-   Collaborative Filtering: Using the SVD (Singular Value Decomposition) algorithm from the Surprise library, the system predicts the ratings a user would give to different beers based on their past ratings and the ratings of similar users.

-   Content-Based Filtering: Using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer from scikit-learn, the system finds similar beers based on their styles and recommends beers with similar styles to the ones the user has rated highly.

The final recommendations are a combination of both collaborative filtering and content-based filtering results, providing a more comprehensive and personalized list of beer suggestions.



## How Was it Made?

The beer recommendation system was made using Python programming language and several libraries which are included in the requirments(Pandas, NumPy, scikit-learn, random, surprise, etc)

### Requirements:

-Python 3
-pandas
-numpy
-scikit-learn
-Surprise

## How to run

1. Download or clone the repository.

https://github.com/Ncnchiche/beerProject


    git clone [link]


2. Add Data Set To your repository:

- [beer_reviews.csv](https://data.world/socialmediadata/beeradvocate/workspace/file?filename=beer_reviews.csv)

3. Create a virtual environment (optional, but recommended):

    1. cd into directory
    
    2. Create venv:
            
            python -m venv venv
     
    3. Activate virtual environment
            
        -To Activate on Windows:

            .\venv\Scripts\activate


        -To Activate on Mac:

            source venv/bin/activate

            
4. Activate and install required libraries using requirments.txt
    ```
    pip install -r requirements.txt
    ```

5. Run the main.py file and have fun :)
    ```
    python3 main.py
    ```

6. When prompted for Profile name(you can find them on CSV) and it will give you top 10 recommendations

 ## Demo from Command Line
 
![My Image](samplePicture.png)
