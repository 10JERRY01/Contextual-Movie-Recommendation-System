# Contextual-Movie-Recommendation-System
# Project Overview
This project is a Contextual Movie Recommendation System designed to provide personalized movie suggestions based on user queries. It integrates natural language processing, machine learning, and a simple web interface built with Streamlit to deliver a seamless recommendation experience.

# Features
Contextual Recommendations: Suggests movies based on user descriptions or genres.
Interactive Interface: Users can input queries and view recommendations dynamically via a Streamlit-based UI.
Customizable Outputs: Number of recommendations can be adjusted using a slider.
Expandable Details: Tags and genres for each recommended movie are viewable in expandable sections.
Technologies Used
Programming Language: Python
Libraries:
Streamlit: For creating the user interface.
Pandas: For data manipulation and processing.
Scikit-learn: For building TF-IDF vector embeddings and computing cosine similarity.
TfidfVectorizer: For natural language processing (NLP).
Cosine Similarity: For calculating similarity between user queries and movie metadata.
# Dataset
The project utilizes the MovieLens Dataset, which consists of the following files:

tags.csv: Contains user-provided movie tags.
ratings.csv: Includes user ratings for movies.
movies.csv: Lists movies with genres.
links.csv: Provides IMDb and TMDb links for movies.
genome-tags.csv: Contains tag definitions.
genome-scores.csv: Maps tags to movies with relevance scores.
Key Columns
movieId: Unique identifier for movies.
title: Movie title.
genres: Movie genres.
all_tags: Combined tags and genres used for recommendation.
# Data Processing
Merging Data: Combine relevant files to create a single dataset with movie metadata, tags, and genres.
# Data Cleaning:
Handle missing values.
Merge tags and genres into a single column (all_tags) for NLP processing.
# Feature Engineering:
Use TF-IDF Vectorization on the all_tags column to create embeddings.
# Recommendation Engine
Input: User provides a query in natural language (e.g., "action-packed superhero movie").
Processing:
Transform the query using the same TF-IDF vectorizer.
Compute cosine similarity between the query and the movie embeddings.
Output: Return the top N movies with the highest similarity scores.
# Streamlit Application
Key Features
Sidebar:
Application title and description.
Slider to adjust the number of recommendations.
Main Interface:
Input box for user queries.
Display of recommendations with similarity scores.
Expandable sections for movie tags and genres.
How to Run the App
Save the app code as app.py.
Ensure the final_data.csv file is in the same directory or provide its path.
Run the Streamlit application:
streamlit run app.py
Open the browser window or the provided link to interact with the app.
Project Structure
Movie-Recommendation-System/
│
|── tags.csv
│── ratings.csv
│── movies.csv
│── links.csv
│── genome-tags.csv
│── genome-scores.csv
│── main.ipynb
├── final_data.csv      # Preprocessed dataset used by the app
├── app.py              # Streamlit application script
└── README.md           # Project documentation

Future Enhancements
Advanced Embeddings: Use pre-trained embeddings like BERT for improved recommendations.
Real-time Updates: Allow dynamic updates with new user data.
Cloud Deployment: Deploy the app on Streamlit Cloud or AWS for public access.
User Authentication: Add user profiles to store preferences and history.
Enhanced UI/UX: Improve the app's design for a more engaging user experience.
Acknowledgments
MovieLens Dataset: The dataset used in this project is publicly available from MovieLens.
Streamlit: For enabling rapid application development.
Scikit-learn: For providing robust machine learning tools.
