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
st.write("Escribe el nombre de las películas que te gustaron y te recomendaré otras similares.")

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
    # Limpiar duplicados por título
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)
    return df


kmeans, scaler, all_genre_cols = load_models()
df_movies = load_movies()

# --------------------------------------------------
# Selección de películas del usuario (directa)
# --------------------------------------------------
st.subheader("🎥 Películas vistas por el usuario")

MAX_MOVIES = 6
user_selections = []

# Lista completa de títulos para el buscador
all_titles = df_movies["title"].sort_values().tolist()

# Registro de películas ya seleccionadas para evitar duplicados en la entrada
selected_in_session = []

for i in range(MAX_MOVIES):
    with st.expander(f"Película {i + 1}", expanded=(i == 0)):
        
        # Filtrar la lista global para no mostrar lo que ya se eligió en otros bloques
        available_titles = [t for t in all_titles if t not in selected_in_session]

        movie = st.selectbox(
            "Busca y selecciona una película",
            options=available_titles,
            index=None, # Aparece vacío por defecto
            placeholder="Escribe el nombre de la película...",
            key=f"movie_select_{i}"
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
            # Bloquear este título para los siguientes selectores
            selected_in_session.append(movie)

# --------------------------------------------------
# Ajustes de Recomendación
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
    vector_dict = {col: 0 for col in all_genre_cols}
    vector_dict["rating"] = avg_rating

    for genres in user_df["genres"]:
        for g in genres.split("|"):
            if g in vector_dict:
                vector_dict[g] += 1

    user_vector = pd.DataFrame([vector_dict])
    cols_order = ["rating"] + all_genre_cols
    return user_vector[cols_order]

# --------------------------------------------------
# Lógica de Recomendación
# --------------------------------------------------
if recommend:
    if not user_selections:
        st.error("❌ Por favor, selecciona al menos una película.")
    else:
        with st.spinner('Analizando tus gustos...'):
            user_df = build_user_df(user_selections, df_movies)
            user_vector = build_user_vector(user_df, all_genre_cols)
            
            # Procesamiento con los modelos cargados
            user_scaled = scaler.transform(user_vector)
            cluster = kmeans.predict(user_scaled)[0]

            # Películas que el usuario ya mencionó
            seen_ids = set(user_df["movieId"])
            
            # Filtrar por cluster y excluir las ya vistas
            recommendations = (
                df_movies[df_movies["cluster_labels"] == cluster]
                .loc[~df_movies["movieId"].isin(seen_ids)]
                .head(top_n)
            )

            st.subheader("🎬 Recomendaciones basadas en tu perfil")
            if not recommendations.empty:
                st.table(recommendations[["title", "genres"]].reset_index(drop=True))
            else:
                st.warning("No se encontraron más películas similares en este grupo.")

st.divider()
st.caption("Filtros de duplicados activos. Buscador directo habilitado.")
