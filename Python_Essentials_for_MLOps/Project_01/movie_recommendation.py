import logging
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(filename='movie_recommendation.log',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def read_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        logging.error("File not found: %s", file_path)
        print("File not found!")
        return None

def clean_title(title):
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    return title

def get_recommendations(movies_df, title, vectorizer, tfidf, num_recommendations=5):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argsort(similarity)[::-1][:num_recommendations]
    results = movies_df.iloc[indices]
    return results

def find_similar_movies(ratings_df, movie_id, movies_df):
    similar_users = ratings_df[(ratings_df["movieId"] == movie_id) &
                               (ratings_df["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings_df[(ratings_df["userId"].isin(similar_users)) &
                                   (ratings_df["rating"] > 4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
    similar_user_recs = similar_user_recs[similar_user_recs > .10]
    all_users = ratings_df[(ratings_df["movieId"].isin(similar_user_recs.index)) &
                           (ratings_df["rating"] > 4)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)
    return rec_percentages.head(10).merge(
        movies_df, left_on="movieId", right_on="movieId")[["score", "title", "genres"]]

def user_interaction(movies_df, ratings_df, vectorizer, tfidf):
    while True:
        user_input = input("Enter the name of a movie (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        recommendations = get_recommendations(movies_df, user_input, vectorizer, tfidf)
        if not recommendations.empty:
            print(f"\nMovie recommendations based on {user_input}:")
            print(recommendations[['title', 'genres']])
        else:
            print("No matching movies found.\n")
        logging.info("User input: %s", user_input)
        movie_id_input = input("Enter the movie ID to find similar movies (or 'skip' to continue): ")
        if movie_id_input.lower() == 'skip':
            continue
        try:
            movie_id = int(movie_id_input)
            similar_movies = find_similar_movies(ratings_df, movie_id, movies_df)
            movie_title = movies_df[movies_df["movieId"] == movie_id]["title"].values[0]
            print(f"\nMovies similar to {movie_title}:")
            print(similar_movies[["title", "genres"]])
            print("\n")
            logging.info("Movie ID input: %s", movie_id_input)
        except ValueError:
            print("Invalid movie ID. Please enter a valid movie ID or 'skip'.\n")
            logging.error("Invalid movie ID input: %s", movie_id_input)

if __name__ == "__main__":
    movies_data = read_data("data/movies.csv")
    ratings_data = read_data("data/ratings.csv")
    movies_data["clean_title"] = movies_data["title"].apply(clean_title)
    vect = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vect.fit_transform(movies_data["clean_title"])
    user_interaction(movies_data, ratings_data, vect, tfidf_matrix)
