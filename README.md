# decode-labs-project4

A full-stack movie recommendation system built with a FastAPI backend and a responsive HTML/JS frontend. The engine utilizes Content-Based Filtering, TF-IDF Vectorization, and Cosine Similarity to suggest movies based on textual and genre alignment.

project/
├── main.py          # FastAPI Backend (Data Processing, Vectorization, & REST API)
├── index.html       # Frontend UI (Responsive Grid, Live Search, & Tabs)
├── requirements.txt # Project Dependencies (pandas, numpy, scikit-learn, fastapi, uvicorn)
└── README.md        # Project Documentation

> Key API Endpoints
GET /api/movies - Fetch all 15 curated movies.
GET /api/movies/{id} - Fetch specific movie details.
GET /api/recommend/{movie_id}?top_n=5 - Retrieve top-N content-based recommendations.
GET /api/search?query={query} - Search movies by title or genre.
GET /api/genres - List all unique genres.
GET /api/top-rated - Get movies sorted by rating.
