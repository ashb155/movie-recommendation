import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Load the MovieLens dataset
@st.cache_data
def load_data():
    try:
        ratings = pd.read_csv(r"datasets1/u.data",
                              sep='\t',
                              names=['user_id', 'item_id', 'rating', 'timestamp'])
        movies = pd.read_csv(r"datasets1/u.item",
                             sep='|',
                             names=['item_id', 'title', 'release_date', 'video_release_date',
                                    'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                                    'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                                    'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                                    'Thriller', 'War', 'Western'],
                             encoding='latin-1')
        return ratings, movies
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None


# Prepare dataset for regression with One-Hot Encoding
def prepare_data(ratings):
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_features = encoder.fit_transform(ratings[['user_id', 'item_id']])

    X = pd.DataFrame(encoded_features)
    y = ratings['rating']

    return X, y, encoder


# Train Linear Regression model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# Predict ratings for unseen movies
def get_recommendations(user_id, movies, model, ratings, encoder):
    all_movies = set(movies['item_id'])
    watched_movies = set(ratings[ratings['user_id'] == user_id]['item_id'])
    unseen_movies = list(all_movies - watched_movies)

    if not unseen_movies:
        return []

    user_movie_pairs = pd.DataFrame({'user_id': [user_id] * len(unseen_movies), 'item_id': unseen_movies})
    encoded_data = encoder.transform(user_movie_pairs)

    predicted_ratings = model.predict(encoded_data)
    predicted_ratings = np.clip(predicted_ratings, 1, 5)

    recommended = [movie for rating, movie in zip(predicted_ratings, unseen_movies) if rating >= 4]

    return recommended[:10]


# Compute model evaluation metrics
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 1, 5)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Convert ratings to binary for precision and recall
    y_pred_binary = (y_pred >= 4).astype(int)
    y_test_binary = (y_test >= 4).astype(int)

    precision = np.sum(y_pred_binary * y_test_binary) / np.sum(y_pred_binary) if np.sum(y_pred_binary) > 0 else 0
    recall = np.sum(y_pred_binary * y_test_binary) / np.sum(y_test_binary) if np.sum(y_test_binary) > 0 else 0
    hit_rate = 1 if np.any(y_pred_binary * y_test_binary) else 0

    # Mean Average Precision (MAP)
    average_precision = np.mean([precision])

    return rmse, precision, recall, hit_rate, average_precision


# Streamlit app
def main():
    st.set_page_config(page_title="Movie Recommender", layout="wide")

    ratings, movies = load_data()
    if ratings is None or movies is None:
        return

    X, y, encoder = prepare_data(ratings)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)

    rmse, precision, recall, hit_rate, average_precision = evaluate_model(model, X_test, y_test)

    user_id = st.number_input("Enter User ID", min_value=1, max_value=943, value=1)

    if st.button("Get Recommendations"):
        with st.spinner("Generating recommendations..."):
            recommendations = get_recommendations(user_id, movies, model, ratings, encoder)

            if recommendations:
                st.success("Here are your recommended movies!")
                for movie_id in recommendations:
                    movie_info = movies[movies['item_id'] == movie_id].iloc[0]
                    st.markdown(f"**{movie_info['title']}**")
            else:
                st.warning("No recommendations found. Try a different user ID.")

            st.markdown("### Model Performance")
            st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
            st.metric("Precision@K", f"{precision:.4f}")
            st.metric("Recall@K", f"{recall:.4f}")
            st.metric("Hit Rate", f"{hit_rate}")
            st.metric("Mean Average Precision (MAP)", f"{average_precision:.4f}")


if __name__ == "__main__":
    main()
