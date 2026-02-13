import streamlit as st
import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import KMeans
from skimage import color
from PIL import Image
import os
import base64

# Configuración de la página
st.set_page_config(
    page_title="Recomendador de Lanas Piccolo",
    page_icon="🧶",
    layout="wide"
)

# --- Función para convertir imagen local a base64 para CSS ---
def get_base64_image(image_path):
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- CSS personalizado para replicar el diseño ---
logo_b64 = get_base64_image("logo-piccolo.png")
logo_style = f"background-image: url(data:image/png;base64,{logo_b64});" if logo_b64 else ""

st.markdown(f"""
<style>
    /* Importar tipografía Inter */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Aplicar tipografía a todo */
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    /* Ocultar branding de Streamlit */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* Fondo principal limpio */
    .stApp {{
        background-color: #ffffff;
    }}

    /* --- Sidebar --- */
    [data-testid="stSidebar"] {{
        background-color: #f8f9fa;
        border-right: 1px solid #e9ecef;
        padding-top: 0;
    }}

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
        padding-top: 1rem;
    }}

    /* --- Títulos --- */
    h1 {{
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        color: #1a1a2e !important;
        font-size: 2.2rem !important;
        margin-bottom: 0.3rem !important;
    }}

    h2 {{
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: #1a1a2e !important;
    }}

    h3 {{
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: #2d6a4f !important;
        font-size: 1rem !important;
    }}

    /* Subtítulo descriptivo */
    .subtitle {{
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }}

    /* --- Info box cómo funciona --- */
    .info-box {{
        background-color: #f0f7f4;
        border-left: 4px solid #2d6a4f;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
    }}

    .info-box h4 {{
        color: #2d6a4f;
        font-weight: 600;
        margin: 0 0 0.6rem 0;
        font-size: 0.95rem;
    }}

    .info-box ol {{
        margin: 0;
        padding-left: 1.2rem;
        color: #2d6a4f;
        font-size: 0.88rem;
        line-height: 1.8;
    }}

    .info-box ol li {{
        margin-bottom: 0.1rem;
    }}

    /* --- Footer sidebar --- */
    .sidebar-footer {{
        position: fixed;
        bottom: 1rem;
        font-size: 0.78rem;
        color: #888;
        padding-left: 1.5rem;
    }}

    /* --- Color swatch resultado --- */
    .color-swatch {{
        width: 90px;
        height: 90px;
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }}

    /* --- Card de resultado --- */
    .result-card {{
        background: #fafbfc;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}

    .result-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    }}

    /* --- Etiqueta de match --- */
    .match-badge {{
        display: inline-block;
        background: linear-gradient(135deg, #2d6a4f, #40916c);
        color: white;
        padding: 0.25rem 0.7rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }}

    .match-badge-secondary {{
        display: inline-block;
        background: linear-gradient(135deg, #457b9d, #669bbc);
        color: white;
        padding: 0.25rem 0.7rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }}

    /* --- Nombre de lana --- */
    .wool-name {{
        font-weight: 600;
        color: #1a1a2e;
        font-size: 0.95rem;
        margin-top: 0.4rem;
    }}

    /* --- Separador --- */
    .divider {{
        border: none;
        border-top: 1px solid #e9ecef;
        margin: 1rem 0;
    }}

    /* --- Upload area --- */
    [data-testid="stFileUploader"] {{
        border: 2px dashed #2d6a4f !important;
        border-radius: 12px !important;
        padding: 0.5rem !important;
    }}

    /* --- Info message --- */
    .stAlert {{
        border-radius: 10px;
    }}

    /* --- Expander personalizado --- */
    .streamlit-expanderHeader {{
        font-size: 0.9rem !important;
        color: #555 !important;
    }}

    /* Sección de paso activo */
    .step-indicator {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0;
    }}

    .step-number {{
        background: #2d6a4f;
        color: white;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 600;
        flex-shrink: 0;
    }}

    .step-text {{
        font-size: 0.85rem;
        color: #333;
    }}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    # Logo
    if os.path.exists("logo-piccolo.png"):
        st.image("logo-piccolo.png", use_container_width=True)
    else:
        st.info("🎨 Piccolo Ind. SAS")

    # Título de la app
    st.markdown("### 🧵 Tu asistente de costura")

    # Info box - Cómo funciona
    st.markdown("""
    <div class="info-box">
        <h4>¿Cómo funciona?</h4>
        <ol>
            <li>Sube una foto de tu proyecto.</li>
            <li>Detectamos los colores clave.</li>
            <li>Te sugerimos las mejores telas<br>de nuestro inventario.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # Expander para WordPress
    with st.expander("⚙️ Ver código para insertar en WordPress"):
        embed_code = '<iframe src="https://[TU-URL-DE-STREAMLIT-CLOUD]" width="100%" height="800" frameborder="0"></iframe>'
        st.code(embed_code, language="html")

    # Footer
    st.markdown('<div class="sidebar-footer">Desarrollado para Piccolo Ind. SAS</div>', unsafe_allow_html=True)


# =============================================
# FUNCIONES
# =============================================

@st.cache_data
def load_data():
    """Carga la base de datos de lanas desde el CSV."""
    possible_paths = ["base_datos_lanas.csv", "../base_datos_lanas.csv", "base_datos_telas.csv"]
    df = None
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                # Intentar primero con coma, luego con punto y coma
                try:
                    df = pd.read_csv(path, sep=',')
                    # Si solo cargó una columna, probablemente el separador es incorrecto
                    if len(df.columns) <= 1:
                        raise ValueError("Probable separador incorrecto")
                except:
                    df = pd.read_csv(path, sep=';')
                
                # Limpiar y convertir Color_RGB
                if 'Color_RGB' in df.columns:
                    def parse_rgb(x):
                        if isinstance(x, str):
                            # Quitar paréntesis, corchetes, comillas y espacios, luego separar por comas
                            parts = x.replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace('"', '').replace("'", "").split(',')
                            return [int(p.strip()) for p in parts if p.strip().isdigit()]
                        return x if isinstance(x, list) else []
                        
                    df['Color_RGB'] = df['Color_RGB'].apply(parse_rgb)
                
                return df
            except Exception as e:
                st.warning(f"Error al procesar {path}: {e}")
                
    st.error("No se encontró la base de datos de lanas (base_datos_lanas.csv)")
    return pd.DataFrame()


def extract_colors(image, k=10):
    """Extrae los k colores dominantes de una imagen usando KMeans."""
    max_width = 800
    if image.width > max_width:
        w_percent = (max_width / float(image.width))
        h_size = int((float(image.height) * float(w_percent)))
        image = image.resize((max_width, h_size), Image.Resampling.LANCZOS)

    img_np = np.array(image)

    if len(img_np.shape) == 3 and img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

    pixels = img_np.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_.astype(int)
    return colors


def rgb_to_lab(rgb_color):
    """Convierte un color RGB a espacio CIELAB."""
    rgb_normalized = np.array([[rgb_color]], dtype=np.uint8)
    lab_color = color.rgb2lab(rgb_normalized)[0][0]
    return lab_color


def calculate_distance(color1_lab, color2_lab):
    """Calcula la distancia Delta-E entre dos colores en espacio LAB."""
    return np.linalg.norm(color1_lab - color2_lab)


# =============================================
# CONTENIDO PRINCIPAL
# =============================================

# Título principal
st.markdown("# Descubre tus Telas Ideales")
st.markdown(
    '<p class="subtitle">Sube la foto de tu proyecto y deja que nuestra IA encuentre '
    'la combinación perfecta de nuestra colección.</p>',
    unsafe_allow_html=True
)

# Cargar datos
df_lanas = load_data()

# Upload de imagen
st.markdown("**Sube la foto de tu proyecto (JPG, PNG)**")
uploaded_file = st.file_uploader(
    "Elige una imagen...",
    type=['jpg', 'jpeg', 'png'],
    label_visibility="collapsed"
)

if uploaded_file is None:
    st.info("👆 Sube una imagen para comenzar.")
elif df_lanas.empty:
    st.error("No se pudo cargar la base de datos de lanas.")
else:
    # Mostrar imagen original
    image = Image.open(uploaded_file)
    st.image(image, caption='📷 Imagen de tu proyecto', use_container_width=True)

    # Barra de progreso
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Paso 1: Análisis
    status_text.markdown("""
    <div class="step-indicator">
        <div class="step-number">1</div>
        <div class="step-text">Analizando píxeles de la imagen...</div>
    </div>
    """, unsafe_allow_html=True)
    progress_bar.progress(25)

    detected_colors = extract_colors(image, k=10)

    # Paso 2: Búsqueda
    status_text.markdown("""
    <div class="step-indicator">
        <div class="step-number">2</div>
        <div class="step-text">Buscando coincidencias en el catálogo...</div>
    </div>
    """, unsafe_allow_html=True)
    progress_bar.progress(50)

    # Pre-calcular LAB para las lanas
    if 'LAB' not in df_lanas.columns:
        df_lanas['LAB'] = df_lanas['Color_RGB'].apply(rgb_to_lab)

    results = []

    for i, dominant_color in enumerate(detected_colors):
        dom_lab = rgb_to_lab(dominant_color)

        distances = df_lanas['LAB'].apply(lambda x: calculate_distance(dom_lab, x))
        closest_indices = distances.argsort()[:2]

        recs = []
        for idx in closest_indices:
            row = df_lanas.iloc[idx]
            dist = distances[idx]
            match_score = max(0, 100 - dist)

            recs.append({
                'name': row['Nombre_Lana'],
                'image_file': row['Nombre_Archivo'],
                'image_url': row['Ruta_Completa'],
                'score': match_score,
                'rgb': row['Color_RGB']
            })

        results.append({
            'detected_rgb': dominant_color,
            'recommendations': recs
        })

        progress = 50 + int((i + 1) / len(detected_colors) * 40)
        progress_bar.progress(min(progress, 90))

    # Paso 3: Resultados
    status_text.markdown("""
    <div class="step-indicator">
        <div class="step-number">3</div>
        <div class="step-text">Generando reporte de coincidencias...</div>
    </div>
    """, unsafe_allow_html=True)
    progress_bar.progress(100)

    st.markdown("---")
    st.markdown("## 🎨 Colores Detectados y Telas Recomendadas")

    for item in results:
        r, g, b = item['detected_rgb']
        color_hex = f"#{r:02x}{g:02x}{b:02x}"

        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        cols = st.columns([1, 2, 2])

        # Columna 1: Color detectado
        with cols[0]:
            st.markdown(f"""
            <div style="text-align: center;">
                <div class="color-swatch" style="background-color: {color_hex}; margin: 0 auto;"></div>
                <p style="margin-top: 0.5rem; font-size: 0.8rem; color: #666;">
                    <strong>{color_hex.upper()}</strong><br>
                    RGB({r}, {g}, {b})
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Columna 2: Primera opción
        with cols[1]:
            rec1 = item['recommendations'][0]
            st.markdown(f'<span class="match-badge">✓ Mejor coincidencia · {rec1["score"]:.0f}%</span>', unsafe_allow_html=True)

            # Intentar cargar desde ruta local, sino desde URL
            img_file = str(rec1['image_file'])
            # Si es solo el nombre del archivo, buscar en imagenes_lanas
            if not os.path.isabs(img_file) and not img_file.startswith("http"):
                 img_path = os.path.join("imagenes_lanas", img_file)
            else:
                 img_path = img_file

            if os.path.exists(img_path):
                st.image(img_path, width=150)
            elif 'image_url' in rec1 and rec1['image_url']:
                 img_url = str(rec1['image_url'])
                 # Si 'image_url' tiene una ruta absoluta local
                 if os.path.exists(img_url):
                     st.image(img_url, width=150)
                 else:
                     st.image(img_url, width=150) # Intentar como URL
            else:
                st.warning(f"Imagen no encontrada: {rec1['image_file']}")

            st.markdown(f'<div class="wool-name">{rec1["name"]}</div>', unsafe_allow_html=True)

        # Columna 3: Segunda opción
        with cols[2]:
            rec2 = item['recommendations'][1]
            st.markdown(f'<span class="match-badge-secondary">Opción 2 · {rec2["score"]:.0f}%</span>', unsafe_allow_html=True)

            # Intentar cargar desde ruta local, sino desde URL
            img_file_2 = str(rec2['image_file'])
            # Si es solo el nombre del archivo, buscar en imagenes_lanas
            if not os.path.isabs(img_file_2) and not img_file_2.startswith("http"):
                 img_path_2 = os.path.join("imagenes_lanas", img_file_2)
            else:
                 img_path_2 = img_file_2

            if os.path.exists(img_path_2):
                st.image(img_path_2, width=150)
            elif 'image_url' in rec2 and rec2['image_url']:
                 img_url_2 = str(rec2['image_url'])
                 # Si 'image_url' tiene una ruta absoluta local
                 if os.path.exists(img_url_2):
                     st.image(img_url_2, width=150)
                 else:
                     st.image(img_url_2, width=150) # Intentar como URL
            else:
                st.warning(f"Imagen no encontrada: {rec2['image_file']}")

            st.markdown(f'<div class="wool-name">{rec2["name"]}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Completado
    status_text.markdown("""
    <div class="step-indicator">
        <div class="step-number" style="background: #2d6a4f;">✓</div>
        <div class="step-text" style="color: #2d6a4f; font-weight: 600;">¡Análisis completado!</div>
    </div>
    """, unsafe_allow_html=True)
