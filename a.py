import pickle
import pandas as pd
import streamlit as st
import base64

# ------------------------------
# Configuración de la app
# ------------------------------
st.set_page_config(
    page_title="Recomendador de Películas",
    layout="centered"
)

# ------------------------------
# Cargar modelos (CACHEADO)
# ------------------------------
@st.cache_resource
def load_assets():
    with open("kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("all_genre_cols.pkl", "rb") as f:
        genre_cols = pickle.load(f)

    return kmeans, scaler, genre_cols


kmeans, scaler, genre_cols = load_assets()

# ------------------------------
# Fondo con imagen local
# ------------------------------
def add_local_bg(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            background-attachment: fixed;
        }}

        .main .block-container {{
            padding-top: 5rem;
            max-width: 600px;
        }}

        [data-testid="stForm"] {{
            background-color: rgba(255, 255, 255, 0.4) !important;
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
        }}

        h1 {{
            color: white !important;
            text-shadow: 2px 2px 10px #000000;
            text-align: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


add_local_bg("movies_bg.png")

st.title("🎬 Recomendador de Películas")

# ------------------------------
# Formulario
# ------------------------------
with st.form("movie_form"):
    st.header("Perfil del usuario")

    avg_rating = st.slider(
        "Rating promedio del usuario",
        min_value=0.5,
        max_value=5.0,
        value=3.5,
        step=0.5
    )

    selected_genres = st.multiselect(
        "Géneros que le gustan",
        options=genre_cols
    )

    submit = st.form_submit_button("Analizar perfil")

# ------------------------------
# Predicción
# ------------------------------
if submit:
    try:
        # Vector de géneros
        genre_df = pd.DataFrame(0, index=[0], columns=genre_cols)
        for g in selected_genres:
            genre_df[g] = 1

        user_df = pd.concat(
            [pd.DataFrame({"rating": [avg_rating]}), genre_df],
            axis=1
        )

        # Escalar
        user_scaled = scaler.transform(user_df)

        with st.spinner("Analizando preferencias..."):
            cluster = kmeans.predict(user_scaled)[0]

        st.markdown(
            f"""
            <div style="
                background-color:#6200ea;
                padding:20px;
                border-radius:15px;
                text-align:center;
                color:white;
                font-size:24px;
                font-weight:bold;
                box-shadow: 0 10px 25px rgba(0,0,0,0.5);
            ">
                🎯 Usuario asignado al clúster {cluster}
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error("❌ Error al analizar el perfil")
        st.exception(e)
