"""
Movie Recommendation System - FastAPI Backend
Project 4: DecodeLabs AI Engineering Internship
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import json

# Initialize FastAPI app
app = FastAPI(title="Movie Recommendation System")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Movie dataset
MOVIES_DATA = [
    {
        "id": 1,
        "title": "The Shawshank Redemption",
        "genre": "Drama",
        "rating": 9.3,
        "description": "Two imprisoned men bond over a number of years finding solace and eventual redemption through acts of common decency."
    },
    {
        "id": 2,
        "title": "The Godfather",
        "genre": "Crime, Drama",
        "rating": 9.2,
        "description": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant youngest son."
    },
    {
        "id": 3,
        "title": "The Dark Knight",
        "genre": "Action, Crime, Drama",
        "rating": 9.0,
        "description": "When the menace known as the Joker emerges from his mysterious past, he wreaks havoc and chaos on the people of Gotham."
    },
    {
        "id": 4,
        "title": "Inception",
        "genre": "Action, Sci-Fi, Thriller",
        "rating": 8.8,
        "description": "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea."
    },
    {
        "id": 5,
        "title": "Pulp Fiction",
        "genre": "Crime, Drama",
        "rating": 8.9,
        "description": "The lives of two mob hitmen, a boxer, a gangster's wife, and a pair of diner bandits intertwine in four tales of violence."
    },
    {
        "id": 6,
        "title": "Forrest Gump",
        "genre": "Drama, Romance",
        "rating": 8.8,
        "description": "The presidencies of Kennedy and Johnson, the Vietnam War, the Watergate scandal and other historical events unfold from the perspective of an Alabama man."
    },
    {
        "id": 7,
        "title": "Gladiator",
        "genre": "Action, Adventure, Drama",
        "rating": 8.5,
        "description": "A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family and sent him into slavery."
    },
    {
        "id": 8,
        "title": "Interstellar",
        "genre": "Adventure, Drama, Sci-Fi",
        "rating": 8.6,
        "description": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival."
    },
    {
        "id": 9,
        "title": "The Matrix",
        "genre": "Action, Sci-Fi",
        "rating": 8.7,
        "description": "A computer programmer discovers that reality as he knows it is a simulation created by machines."
    },
    {
        "id": 10,
        "title": "Avatar",
        "genre": "Action, Adventure, Sci-Fi",
        "rating": 7.8,
        "description": "A paraplegic Marine dispatched to the moon Pandora on a unique mission falls in love with a native Na'vi."
    },
    {
        "id": 11,
        "title": "The Prestige",
        "genre": "Drama, Mystery, Sci-Fi",
        "rating": 8.5,
        "description": "After a tragic accident, two stage magicians engage in an increasingly elaborate battle to create the ultimate illusion."
    },
    {
        "id": 12,
        "title": "Titanic",
        "genre": "Drama, Romance",
        "rating": 7.8,
        "description": "A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic."
    },
    {
        "id": 13,
        "title": "The Avengers",
        "genre": "Action, Adventure, Sci-Fi",
        "rating": 8.0,
        "description": "Earth's mightiest heroes must come together and learn to fight as a team to save the world from destruction."
    },
    {
        "id": 14,
        "title": "Fight Club",
        "genre": "Drama",
        "rating": 8.8,
        "description": "An insomniac office worker and a devil-may-care soapmaker form an underground fight club that evolves into much more."
    },
    {
        "id": 15,
        "title": "The Wolf of Wall Street",
        "genre": "Biography, Comedy, Crime, Drama",
        "rating": 8.2,
        "description": "Based on the true story of Jordan Belfort, from his rise as a charismatic stock-broker to his fall involving crime and corruption."
    },
]

# Convert to DataFrame
df_movies = pd.DataFrame(MOVIES_DATA)

# Prepare TF-IDF vectorizer and fit on movie descriptions
tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df_movies['description'])

# Calculate similarity matrix once
similarity_matrix = cosine_similarity(tfidf_matrix)


@app.get("/")
def read_root():
    """Serve the main HTML page"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(base_dir, "index.html")
    return FileResponse(index_path, media_type="text/html")



@app.get("/api/movies")
def get_all_movies():
    """Get all available movies"""
    return {
        "status": "success",
        "total": len(MOVIES_DATA),
        "movies": MOVIES_DATA
    }


@app.get("/api/movies/{movie_id}")
def get_movie(movie_id: int):
    """Get a specific movie by ID"""
    movie = next((m for m in MOVIES_DATA if m["id"] == movie_id), None)
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    return {"status": "success", "movie": movie}


@app.get("/api/recommend/{movie_id}")
def get_recommendations(movie_id: int, top_n: int = 5):
    """
    Get movie recommendations based on a selected movie
    
    Args:
        movie_id: ID of the movie to base recommendations on
        top_n: Number of recommendations to return (default: 5)
    
    Returns:
        List of recommended movies with similarity scores
    """
    # Find the movie index
    movie_idx = df_movies[df_movies['id'] == movie_id].index
    
    if len(movie_idx) == 0:
        raise HTTPException(status_code=404, detail="Movie not found")
    
    movie_idx = movie_idx[0]
    
    # Get the selected movie
    selected_movie = MOVIES_DATA[movie_idx]
    
    # Get similarity scores for all movies
    sim_scores = list(enumerate(similarity_matrix[movie_idx]))
    
    # Sort by similarity score (descending) and exclude the movie itself
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]
    similarity_scores = [float(i[1]) for i in sim_scores]
    
    # Get recommended movies
    recommended_movies = [MOVIES_DATA[i] for i in movie_indices]
    
    # Add similarity scores to recommendations
    for i, movie in enumerate(recommended_movies):
        movie['similarity_score'] = similarity_scores[i]
    
    return {
        "status": "success",
        "selected_movie": selected_movie,
        "recommendations": recommended_movies,
        "total_recommendations": len(recommended_movies)
    }


@app.get("/api/search")
def search_movies(query: str):
    """
    Search movies by title or genre
    
    Args:
        query: Search query string
    
    Returns:
        List of matching movies
    """
    query_lower = query.lower()
    
    matching_movies = [
        movie for movie in MOVIES_DATA
        if query_lower in movie['title'].lower() or 
           query_lower in movie['genre'].lower()
    ]
    
    return {
        "status": "success",
        "query": query,
        "total_results": len(matching_movies),
        "movies": matching_movies
    }


@app.get("/api/genres")
def get_unique_genres():
    """Get all unique genres"""
    genres_set = set()
    for movie in MOVIES_DATA:
        for genre in movie['genre'].split(", "):
            genres_set.add(genre)
    
    return {
        "status": "success",
        "total_genres": len(genres_set),
        "genres": sorted(list(genres_set))
    }


@app.get("/api/top-rated")
def get_top_rated(limit: int = 10):
    """Get top-rated movies"""
    sorted_movies = sorted(MOVIES_DATA, key=lambda x: x['rating'], reverse=True)
    top_movies = sorted_movies[:min(limit, len(sorted_movies))]
    
    return {
        "status": "success",
        "total": len(top_movies),
        "movies": top_movies
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
