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
    df = pd.read_csv("movies_with_clusters.csv")
    # SOLUCIÓN 1: Eliminar duplicados del CSV original
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)
    return df


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

# Mantenemos un registro de títulos ya seleccionados para que no se repitan en los selectbox
selected_titles = []

for i in range(MAX_MOVIES):
    with st.expander(f"Película {i + 1}", expanded=(i == 0)):

        genre = st.selectbox(
            "Selecciona un género",
            [""] + all_genres,
            key=f"genre_{i}"
        )

        if genre:
            # SOLUCIÓN 2: Filtrar películas por género Y que no hayan sido seleccionadas antes
            movies_by_genre = (
                df_movies[
                    df_movies["genres"].str.contains(genre, na=False) & 
                    ~df_movies["title"].isin(selected_titles)
                ]
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
                    # Añadir a la lista de "ya elegidas" para el siguiente bloque
                    selected_titles.append(movie)

# --------------------------------------------------
# Número de recomendaciones
# --------------------------------------------------
st.subheader("🎯 Ajustes de recomendación")

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

    # Usar un diccionario para asegurar que todas las columnas de géneros existan
    vector_dict = {col: 0 for col in all_genre_cols}
    vector_dict["rating"] = avg_rating

    for genres in user_df["genres"]:
        for g in genres.split("|"):
            if g in vector_dict:
                # Sumamos la presencia del género
                vector_dict[g] += 1

    # Convertir a DataFrame asegurando el orden correcto de las columnas
    user_vector = pd.DataFrame([vector_dict])
    
    # Reordenar columnas para que coincidan exactamente con el scaler
    cols_order = ["rating"] + all_genre_cols
    return user_vector[cols_order]

# --------------------------------------------------
# Recomendación final
# --------------------------------------------------
if recommend:
    if len(user_selections) == 0:
        st.error("❌ Selecciona al menos una película.")
    else:
        with st.spinner('Calculando tus recomendaciones...'):
            user_df = build_user_df(user_selections, df_movies)
            user_vector = build_user_vector(user_df, all_genre_cols)
            
            # Escalar y Predecir Cluster
            user_scaled = scaler.transform(user_vector)
            cluster = kmeans.predict(user_scaled)[0]

            # Filtrar películas que no ha visto
            seen_ids = set(user_df["movieId"])
            
            # Obtener recomendaciones del mismo cluster
            recommendations = (
                df_movies[df_movies["cluster_labels"] == cluster]
                .loc[~df_movies["movieId"].isin(seen_ids)]
                .head(top_n)
            )

            st.subheader("🎬 Películas recomendadas para ti")
            if not recommendations.empty:
                st.dataframe(
                    recommendations[["title", "genres"]],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No encontramos películas adicionales en este cluster.")

# Pie de página
st.divider()
st.caption("Sistema de recomendación basado en K-Means Clustering.")
