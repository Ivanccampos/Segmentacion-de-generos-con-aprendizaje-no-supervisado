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
                    selected_titles.add(movie)  # añadimos para no repetir

# --------------------------------------------------
# Número de recomendaciones
# --------------------------------------------------
st.subheader("🎯 Recomendaciones")

top_n = st.slider(
    "Número de recomendaciones",
    min_value=5,
    max_value=20,
    value=10
)

recommend = st.button("🍿 Recomendar películas")

# --------------------------------------------------
# Funciones auxiliares
# --------------------------------------------------
def build_user_df(selections, df_movies):
    rows = []

    for sel in selections:
        row = df_movies[df_movies["title"] == sel["title"]].iloc[0]
        rows.append({
            "movieId": row["movieId"],
            "title": row["title"],
            "genres": row["genres"],
            "rating": sel["rating"]
        })

    return pd.DataFrame(rows)


def build_user_vector(user_df, all_genre_cols):
    avg_rating = user_df["rating"].mean()

    genres_ohe = user_df["genres"].str.get_dummies(sep="|")
    genre_profile = genres_ohe.sum().to_frame().T

    aligned_genres = pd.DataFrame(0, index=[0], columns=all_genre_cols)
    for col in genre_profile.columns:
        if col in aligned_genres.columns:
            aligned_genres[col] = genre_profile[col].values[0]

    user_vector = pd.concat(
        [pd.DataFrame({"rating": [avg_rating]}), aligned_genres],
        axis=1
    )

    return user_vector

# --------------------------------------------------
# Recomendación final
# --------------------------------------------------
if recommend:
    if len(user_selections) == 0:
        st.error("❌ Selecciona al menos una película.")
    else:
        user_df = build_user_df(user_selections, df_movies)
        user_vector = build_user_vector(user_df, all_genre_cols)
        user_scaled = scaler.transform(user_vector)
        cluster = kmeans.predict(user_scaled)[0]

        seen_ids = set(user_df["movieId"])

        # Evitar que se repitan títulos en la recomendación
        recommendations = (
            df_movies[df_movies["cluster_labels"] == cluster]
            .loc[~df_movies["movieId"].isin(seen_ids)]
            .drop_duplicates(subset="title")  # no repetir títulos
            .head(top_n)
        )

        st.subheader("🎬 Películas recomendadas para ti")
        st.dataframe(
            recommendations[["title", "genres"]],
            use_container_width=True
        )
