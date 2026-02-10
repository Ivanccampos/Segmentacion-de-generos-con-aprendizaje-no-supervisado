import streamlit as st
import pandas as pd

# -------------------------
# Carga de datos y modelos
# -------------------------

@st.cache_data
def load_movies():
    return pd.read_csv("movies_with_clusters.csv")

df_movies = load_movies()

# -------------------------
# Funciones auxiliares
# -------------------------

def get_movies_by_genre(genre, excluded_titles):
    """Devuelve una lista de películas de un género excluyendo las ya seleccionadas."""
    filtered = df_movies[df_movies[genre] == 1]
    return [title for title in filtered['title'].tolist() if title not in excluded_titles]

# -------------------------
# UI
# -------------------------

st.title("Recomendador de películas")

# Inicializamos sesión
if "user_selections" not in st.session_state:
    st.session_state.user_selections = []
if "selected_titles" not in st.session_state:
    st.session_state.selected_titles = set()

# Número máximo de selecciones
MAX_SELECTIONS = 6

# Selección de películas hasta 6 veces
for i in range(MAX_SELECTIONS):
    st.subheader(f"Selección {i+1}")

    genre = st.selectbox("Selecciona un género:", df_movies.columns[1:-1], key=f"genre_{i}")

    # Filtramos películas del género sin repetir
    available_movies = get_movies_by_genre(genre, st.session_state.selected_titles)

    if not available_movies:
        st.info("No quedan películas disponibles en este género.")
        continue

    movie = st.selectbox("Selecciona una película:", available_movies, key=f"movie_{i}")
    rating = st.slider("Indica tu nota (rating):", 1, 10, 5, key=f"rating_{i}")

    # Botón para confirmar selección
    if st.button("Añadir película", key=f"add_{i}"):
        st.session_state.user_selections.append({
            "title": movie,
            "rating": rating
        })
        st.session_state.selected_titles.add(movie)
        st.success(f"¡Añadiste {movie} con rating {rating}!")

# -------------------------
# Recomendaciones
# -------------------------

if st.session_state.user_selections:
    n_recs = st.number_input("Número de recomendaciones:", min_value=1, max_value=20, value=5)

    if st.button("Recomendar"):
        # Ejemplo simple: recomendar películas del mismo cluster de las seleccionadas
        # Aquí puedes poner tu lógica de KMeans u otro modelo
        st.subheader("Recomendaciones")
        recommended = df_movies[~df_movies['title'].isin(st.session_state.selected_titles)].sample(n=min(n_recs, len(df_movies)))
        for idx, row in recommended.iterrows():
            st.write(f"{row['title']}")

