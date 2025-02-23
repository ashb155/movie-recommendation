import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

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

# Create user-item matrix with normalized ratings
def create_user_item_matrix(ratings):
    matrix = ratings.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    user_mean_ratings = matrix.mean(axis=1)
    normalized_matrix = matrix.sub(user_mean_ratings, axis=0)
    return matrix, normalized_matrix

# Create user profile visualization
def create_genre_preference_chart(user_ratings, movies):
    user_movies = movies[movies['item_id'].isin(user_ratings[user_ratings > 0].index)]
    genre_cols = movies.columns[5:]
    genre_counts = user_movies[genre_cols].sum()

    fig = go.Figure(data=[
        go.Bar(
            x=genre_counts.index,
            y=genre_counts.values,
            marker_color='rgb(158,202,225)',
            text=genre_counts.values,
            textposition='auto',
        )
    ])

    fig.update_layout(
        title="Your Ratings by Genre",
        xaxis_tickangle=-45,
        showlegend=False,
        height=400
    )
    return fig

# Create rating distribution chart
def create_rating_distribution(user_ratings):
    ratings_dist = user_ratings[user_ratings > 0].value_counts().sort_index()

    fig = go.Figure(data=[
        go.Bar(
            x=ratings_dist.index,
            y=ratings_dist.values,
            marker_color='rgb(142,124,195)',
            text=ratings_dist.values,
            textposition='auto',
        )
    ])

    fig.update_layout(
        title="Your Rating Distribution",
        xaxis_title="Rating",
        yaxis_title="Number of Movies",
        height=300
    )
    return fig

# Get movie recommendations using collaborative filtering
def get_recommendations(user_id, user_item_matrix, normalized_matrix, movies, n_recommendations=5):
    user_similarity = cosine_similarity(normalized_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=normalized_matrix.index, columns=normalized_matrix.index)

    similar_users = user_similarity_df.loc[user_id].sort_values(ascending=False)[1:11].index
    user_ratings = user_item_matrix.loc[user_id]
    already_rated = user_ratings[user_ratings > 0].index.tolist()

    similar_users_ratings = user_item_matrix.loc[similar_users]
    similar_users_similarities = user_similarity_df.loc[user_id, similar_users]

    weighted_ratings = pd.DataFrame(0, index=similar_users_ratings.columns, columns=['weighted_rating', 'similarity_sum'])

    for user, similarity in zip(similar_users, similar_users_similarities):
        user_rates = similar_users_ratings.loc[user]
        weighted_ratings.loc[user_rates.index, 'weighted_rating'] += user_rates * similarity
        weighted_ratings.loc[user_rates.index, 'similarity_sum'] += np.abs(similarity)

    weighted_ratings['final_score'] = weighted_ratings['weighted_rating'] / weighted_ratings['similarity_sum']
    weighted_ratings = weighted_ratings[~weighted_ratings.index.isin(already_rated)]

    top_recommendations = weighted_ratings.nlargest(n_recommendations, 'final_score')

    recommended_movies = []
    for movie_id, scores in top_recommendations.iterrows():
        movie_info = movies[movies['item_id'] == movie_id].iloc[0]
        genres = [col for col in movies.columns[5:] if movie_info[col] == 1]
        recommended_movies.append({
            'title': movie_info['title'],
            'rating': scores['final_score'],
            'genres': genres
        })

    return recommended_movies, top_recommendations.index.tolist()

# Evaluation metrics
def precision_at_k(recommended, actual, k):
    relevant_recommendations = set(recommended[:k]) & set(actual)
    return len(relevant_recommendations) / k if k > 0 else 0

def recall_at_k(recommended, actual, k):
    relevant_recommendations = set(recommended[:k]) & set(actual)
    return len(relevant_recommendations) / len(actual) if len(actual) > 0 else 0

def hit_rate(recommended, actual):
    return 1 if set(recommended) & set(actual) else 0

def rmse(predicted, actual):
    return np.sqrt(np.mean((predicted - actual) ** 2))

def mean_average_precision(recommended, actual):
    if not actual:
        return 0
    avg_precision = 0
    for k in range(1, len(recommended) + 1):
        avg_precision += precision_at_k(recommended, actual, k)
    return avg_precision / len(recommended)

# Streamlit app
def main():
    st.set_page_config(
        page_title="Enhanced Movie Recommender",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
        }
        .recommendation-card {
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    col1, col2 = st.columns([2, 3])
    with col1:
        st.title("🎬 Movie Mate")
        st.subheader("Your Personal Movie Recommendation Engine")

    # Load data
    ratings, movies = load_data()
    if ratings is None or movies is None:
        return

    user_item_matrix, normalized_matrix = create_user_item_matrix(ratings)

    # Sidebar
    with st.sidebar:
        st.header("👤 User Profile")

        user_id = st.number_input("Enter User ID", min_value=1, max_value=943, value=1)
        n_recommendations = st.slider("Number of recommendations", 5, 20, 10)

        if user_id in user_item_matrix.index:
            user_ratings = user_item_matrix.loc[user_id]
            rated_count = (user_ratings > 0).sum()
            avg_rating = user_ratings[user_ratings > 0].mean()

            st.metric("Movies Rated", rated_count)
            st.metric("Average Rating", f"{avg_rating:.2f}")

            # Rating distribution
            st.plotly_chart(create_rating_distribution(user_ratings), use_container_width=True)

            # Genre preferences
            st.plotly_chart(create_genre_preference_chart(user_ratings, movies), use_container_width=True)

    # Main content
    tabs = st.tabs(["Your Ratings and Recommendations", "About"])

    with tabs[0]:
        action = st.selectbox("Select Action", ["Generate Recommendations", "View Your Ratings"])

        if action == "Generate Recommendations":
            if st.button("Get Recommendations", type="primary"):
                if user_id in user_item_matrix.index:
                    with st.spinner("🎬 Magic in progress... Analyzing your taste in movies..."):
                        recommendations, recommended_ids = get_recommendations(
                            user_id,
                            user_item_matrix,
                            normalized_matrix,
                            movies,
                            n_recommendations
                        )

                        # Get actual favorites for evaluation
                        actual_favorites = user_ratings[user_ratings > 0].index.tolist()

                        # Calculate metrics
                        precision = precision_at_k(recommended_ids, actual_favorites, n_recommendations)
                        recall = recall_at_k(recommended_ids, actual_favorites, n_recommendations)
                        hit = hit_rate(recommended_ids, actual_favorites)
                        predicted_ratings = [movie['rating'] for movie in recommendations]
                        actual_ratings = user_ratings[recommended_ids].values[:len(predicted_ratings)]
                        error = rmse(np.array(predicted_ratings), actual_ratings)
                        map_score = mean_average_precision(recommended_ids, actual_favorites)

                        st.success("🎉 Here are your personalized movie recommendations!")

                        # Display recommendations in an enhanced grid
                        for idx, movie in enumerate(recommendations):
                            with st.container():
                                st.markdown(f"""
                                <div class="recommendation-card">
                                    <h3>{idx + 1}. {movie['title']}</h3>
                                    <p><strong>Match Score:</strong> {'⭐' * int(round(movie['rating']))} ({movie['rating']:.2f})</p>
                                    <p><strong>Genres:</strong> {', '.join(movie['genres'])}</p>
                                </div>
                                """, unsafe_allow_html=True)

                else:
                    st.error("User  ID not found. Please enter a valid User ID.")

        elif action == "View Your Ratings":
            st.header("Your Ratings")
            if rated_count > 0:
                sorted_ratings = user_ratings[user_ratings > 0].sort_values(ascending=False)

                for movie_id in sorted_ratings.index:
                    movie_info = movies[movies['item_id'] == movie_id].iloc[0]
                    genres = [col for col in movies.columns[5:] if movie_info[col] == 1]
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <h4>{movie_info['title']}</h4>
                        <p><strong>Your Rating:</strong> {sorted_ratings[movie_id]}</p>
                        <p><strong>Genres:</strong> {', '.join(genres)}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("You have not rated any movies yet.")

    with tabs[1]:
        st.header("ℹ️ About Movie Mate")
        st.write("""
        Movie Mate is your personal movie recommendation engine powered by advanced collaborative filtering.
        It analyzes your movie ratings and finds patterns among similar users to suggest movies you're likely to enjoy.

        ### How it works:
        1. Enter your User ID
        2. View your movie watching profile
        3. Get personalized recommendations
        4. Explore movies by genre

        ### Dataset
        This system uses the MovieLens 100K dataset, which contains 100,000 ratings from 943 users on 1,682 movies.
        """)

        # Display some statistics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Users", len(ratings['user_id'].unique()))
        col2.metric("Total Movies", len(movies))
        col3.metric("Total Ratings", len(ratings))


if __name__ == "__main__":
    main()