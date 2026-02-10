import streamlit as st
import pandas as pd
import pickle

# --------------------------------------------------
# Configuración de la página
# --------------------------------------------------
st.set_page_config(
    page_title="🎬 Recomendador de Películas",
    layout="centered"
)

st.title("🎬 Recomendador de Películas")
st.write("Selecciona películas que te gustaron y te recomendaré otras similares.")

# --------------------------------------------------
# Cargar modelos y datos
# --------------------------------------------------
@st.cache_resource
def load_models():
    with open("kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("all_genre_cols.pkl", "rb") as f:
        all_genre_cols = pickle.load(f)

    return kmeans, scaler, all_genre_cols


@st.cache_data
def load_movies():
    return pd.read_csv("movies_with_clusters.csv")


kmeans, scaler, all_genre_cols = load_models()
df_movies = load_movies()

# --------------------------------------------------
# Obtener lista de géneros disponibles
# --------------------------------------------------
all_genres = sorted(
    set(
        genre
        for g in df_movies["genres"].dropna()
        for genre in g.split("|")
    )
)

# --------------------------------------------------
# Selección de películas del usuario (hasta 6)
# --------------------------------------------------
st.subheader("🎥 Películas vistas por el usuario")

MAX_MOVIES = 6
user_selections = []
selected_titles = set()  # para evitar repeticiones

for i in range(MAX_MOVIES):
    with st.expander(f"Película {i + 1}", expanded=(i == 0)):

        genre = st.selectbox(
            "Selecciona un género",
            [""] + all_genres,
            key=f"genre_{i}"
        )

        if genre:
            movies_by_genre = (
                df_movies[df_movies["genres"].str.contains(genre, na=False)]
                .loc[~df_movies["title"].isin(selected_titles)]  # evita repeticiones
                ["title"]
                .sort_values()
                .tolist()
            )

            if movies_by_genre:
                movie = st.selectbox(
                    "Selecciona una película",
                    [""] + movies_by_genre,
                    key=f"movie_{i}"
                )

                if movie:
                    rating = st.slider(
                        "Tu valoración",
                        min_value=0.5,
                        max_value=5.0,
                        value=3.0,
                        step=0.5,
                        key=f"rating_{i}"
                    )

                    user_selections.append({
                        "title": movie,
                        "rating": rating
                    })
                    sel
