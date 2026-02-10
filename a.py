import streamlit as st
import pandas as pd
import pickle

# -------------------------
# Cargar artefactos
# -------------------------
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
    # Ajusta el nombre si tu CSV se llama distinto
    return pd.read_csv("movies_with_clusters.csv")


kmeans, scaler, all_genre_cols = load_models()
df_movies = load_movies()

# -------------------------
# UI
# -------------------------
st.title("🎬 Recomendador de Películas")
st.write("Dime algunas películas que te gustaron y te recomiendo otras similares.")

st.subheader("🎥 Películas vistas por el usuario")

user_movies = st.text_area(
    "Introduce películas y ratings (una por línea):",
    value="Toy Story (1995)|4.5\nHeat (1995)|5.0\nGrumpier Old Men (1995)|3.0"
)

top_n = st.slider("Número de recomendaciones", 5, 20, 10)

# -------------------------
# Procesamiento
# -------------------------
def parse_user_input(text, df):
    rows = []
    for line in text.split("\n"):
        try:
            title, rating = line.split("|")
            rating = float(rating)

            movie_row = df[df["title"] == title.strip()]
            if not movie_row.empty:
                rows.append({
                    "movieId": movie_row.iloc[0]["movieId"],
                    "title": title.strip(),
                    "genres": movie_row.iloc[0]["genres"],
                    "rating": rating
                })
        except:
            continue

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


# -------------------------
# Recomendación
# -------------------------
if st.button("🎯 Recomendar películas"):
    user_df = parse_user_input(user_movies, df_movies)

    if user_df.empty:
        st.error("No se pudieron procesar las películas introducidas.")
    else:
        user_vector = build_user_vector(user_df, all_genre_cols)
        user_vector_scaled = scaler.transform(user_vector)

        cluster = kmeans.predict(user_vector_scaled)[0]

        seen_movies = set(user_df["movieId"])

        recommendations = (
            df_movies[df_movies["cluster_labels"] == cluster]
            .loc[~df_movies["movieId"].isin(seen_movies)]
            .groupby("title")
            .size()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index(name="popularity")
        )

        st.subheader("🍿 Recomendaciones para ti")
        st.dataframe(recommendations[["title"]])
