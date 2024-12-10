import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load and prepare data
@st.cache
def load_data():
    # Load preprocessed final dataset (replace 'final_data.csv' with your file path)
    final_data = pd.read_csv('final_data.csv')
    return final_data

final_data = load_data()

# Create TF-IDF embeddings
@st.cache(allow_output_mutation=True)
def prepare_embeddings(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['all_tags'].fillna(''))
    return tfidf, tfidf_matrix

tfidf, tfidf_matrix = prepare_embeddings(final_data)

# Recommendation function
def recommend_movies(query, top_n=5):
    query_vector = tfidf.transform([query])
    query_similarity = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = query_similarity.argsort()[-top_n:][::-1]
    recommendations = final_data.iloc[top_indices][['title', 'all_tags']]
    recommendations['similarity'] = query_similarity[top_indices]
    return recommendations

# Sidebar
st.sidebar.title("🎥 Movie Recommendation System")
st.sidebar.write("Find your next favorite movie!")

# Main Content
st.title("🎥 Welcome to the Movie Recommendation System")
st.write("Enter a description, genre, or movie title to get recommendations!")

# User input
query = st.text_input(
    "Type a movie description or genre (e.g., 'romantic comedy with a twist'):",
    placeholder="e.g., Action-packed superhero movie"
)

# Number of recommendations slider
top_n = st.sidebar.slider("Number of recommendations", min_value=1, max_value=10, value=5)

if query:
    st.write(f"### Recommendations for your query: **{query}**")
    recommendations = recommend_movies(query, top_n=top_n)
    
    for index, row in recommendations.iterrows():
        st.write(f"**{row['title']}** - Similarity: `{row['similarity']:.2f}`")
        with st.expander("Show Tags/Genres"):
            st.write(f"**Tags/Genres:** {row['all_tags']}")
        st.write("---")

# Footer
st.sidebar.write("Developed with ❤️ by Rahul")
st.sidebar.write("Powered by Python & Streamlit")
