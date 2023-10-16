# Movie Recommendation System

This Python script implements a movie recommendation system. It allows users to input the name of a movie, and it provides recommendations based on that input. The recommendations are generated using TF-IDF (Term Frequency-Inverse Document Frequency) to find similar movie titles. Additionally, users can provide a movie ID to find movies that are similar based on user ratings.

The code uses Pandas for data manipulation, Scikit-learn for TF-IDF vectorization, and cosine similarity for recommendation calculations.

## Usage

1 - **Requirements:**

   Make sure you have the following dependencies installed:
   - Python 3.x
   - pandas
   - scikit-learn
   - requests
   - numpy

   You can click [here](https://files.grouplens.org/datasets/movielens/ml-25m.zip) to download the dataset used, and is expected to be in CSV files ('data/movies.csv' and 'data/ratings.csv').

2 - **Executing the script:**

   "All you need to do to run the script is to execute:"

   ```
   python movie_recommendation.py
   ```

3 - **To get movie recomendations:**

At first, you should provide the movie's name to receive recommendations. If you wish, you can also provide the movie's ID to get similar films, as shown in the image below:

![code output](https://github.com/mthwk/mlops2023/blob/74468e199e619f3a1c04881b2214888ad0a0e298/Python_Essentials_for_MLOps/Project_01/images/output.png)


## Using Pylint

Pylint is a static code analysis tool for Python that checks code for errors, enforces a coding standard, and provides code quality and style feedback. To use Pylint, you should run the following command::

```
pylint movie_recommendation.py
```

The script movie_recommendation.py received a linting score of 8.97.

![Pylint first result](https://github.com/mthwk/mlops2023/blob/74468e199e619f3a1c04881b2214888ad0a0e298/Python_Essentials_for_MLOps/Project_01/images/pylint.png)

After reducing some lines and adding comments to explain sections of the code, the maximum score was achieved, as shown in the following image.

![Pylint final result[(https://github.com/mthwk/mlops2023/blob/3336b0db03a4916462b17b3a7fa3fd0639cc22a8/Python_Essentials_for_MLOps/Project_01/images/pylint2.png)
## References
[Dataquest - Build a Movie Recommendation System in Python](https://github.com/dataquestio/project-walkthroughs/blob/master/movie_recs/movie_recommendations.ipynb)
