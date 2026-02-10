import streamlit as st
import pandas as pd

# -------------------------
# Carga de datos
# -------------------------

@st.cache_data
def load_movies():
    return pd.read_csv("movies_with_clusters.csv")

df_movies = load_movies()

# -------------------------
# Inicializamos sesión
# -------------------------

if "user_selections" not in st.session_state:
    st.session_state.user_selections = []

if "selected_titles" not in st.session_state:
    st.session_state.selected_titles = set()

if "current_step" not in st.session_state:
    st.session_state.current_step = 0

MAX_SELECTIONS = 6

# -------------------------
# Funciones auxiliares
# -------------------------

def get_available_movies(genre):
    filtered = df_movies[df_movies[genre] == 1]
    return [t for t in filtered['title'].tolist() if t not in st.session_state.selected_titles]

# -------------------------
# Interfaz principal
# -------------------------

st.title("Recomendador de películas")

# Selección de película actual
if st.session_state.current_step < MAX_SELECTIONS:
    st.subheader(f"Selección {st.session_state.current_step + 1}")
    
    genre = st.selectbox(
        "Selecciona un género:",
        df_movies.columns[1:-1],  # asumimos que la primera columna es 'title' y la última quizá 'cluster'
        key=f"genre_{st.session_state.current_step}"
    )

    available_movies = get_available_movies(genre)
    
    if available_movies:
        movie = st.selectbox("Selecciona una película:", available_movies, key=f"movie_{st.session_state.current_step}")
        rating = st.slider("Indica tu nota (rating):", 1, 10, 5, key=f"rating_{st.session_state.current_step}")
        
        if st.button("Añadir película"):
            st.session_state.user_selections.append({
                "title": movie,
                "rating": rating
            })
            st.session_state.selected_titles.add(movie)
            st.session_state.current_step += 1
            st.experimental_rerun()  # recarga la app para mostrar la siguiente selección
    else:
        st.info("No quedan películas disponibles en este género.")
else:
    st.success("¡Has seleccionado todas tus películas!")

# -------------------------
# Recomendaciones
# -------------------------

if st.session_state.user_selections:
    n_recs = st.number_input("Número de recomendaciones:", min_value=1, max_value=20, value=5)
    if st.button("Recomendar"):
        st.subheader("Recomendaciones")
        recommended = df_movies[~df_movies['title'].isin(st.session_state.selected_titles)].sample(
            n=min(n_recs, len(df_movies))
        )
        for idx, row in recommended.iterrows():
            st.write(f"{row['title']}")
