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

    /* --- Info box cómo funciona (Azul estilo imagen) --- */
    .info-box {{
        background-color: #e7f1f7;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
        color: #31708f;
    }}

    .info-box h4 {{
        color: #245671;
        font-weight: 600;
        margin: 0 0 0.8rem 0;
        font-size: 1rem;
    }}

    .info-box ol {{
        margin: 0;
        padding-left: 1.2rem;
        font-size: 0.9rem;
        line-height: 1.6;
    }}

    .info-box ol li {{
        margin-bottom: 0.4rem;
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

    /* --- Tarjeta de producto (estilo imagen recibida) --- */
    .product-card {{
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        margin-bottom: 10px;
        height: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: space-between;
    }}

    .product-name-new {{
        font-weight: 700;
        color: #333;
        font-size: 1.1rem;
        margin-top: 15px;
        margin-bottom: 5px;
    }}

    .product-match-new {{
        color: #777;
        font-size: 0.9rem;
        margin-bottom: 15px;
    }}

    /* --- Botón estilizado --- */
    .custom-button {{
        display: inline-block;
        width: 100%;
        padding: 10px 0;
        background-color: #ffffff;
        border: 1px solid #cccccc;
        border-radius: 8px;
        color: #333;
        text-decoration: none !important;
        font-weight: 500;
        font-size: 0.95rem;
        transition: background-color 0.2s;
        margin-top: 10px;
    }}

    .custom-button:hover {{
        background-color: #f8f9fa;
        border-color: #999999;
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
            <li>Te sugerimos las mejores lanas de nuestro inventario.</li>
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
st.markdown("# Descubre tus Lanas Ideales")
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
    
    st.markdown("### Colores que armonizan perfectamente con tu foto")
    
    for idx, item in enumerate(results):
        r, g, b = item['detected_rgb']
        color_hex = f"#{r:02x}{g:02x}{b:02x}"

        # Layout de 3 columnas como en la imagen
        cols = st.columns([1, 1.5, 1.5])

        # Columna 1: Info del Color Detectado
        with cols[0]:
            st.markdown(f"**Color Detectado #{idx + 1}**")
            st.markdown(f"""
            <div style="background-color: {color_hex}; width: 100%; height: 100px; border-radius: 8px; border: 1px solid #eee; margin: 10px 0;"></div>
            <div style="font-size: 0.8rem; color: #777; font-family: monospace;">
                RGB: ({r}, {g}, {b})
            </div>
            """, unsafe_allow_html=True)

        # Columna 2: Sugerencia 1
        with cols[1]:
            st.markdown("**Sugerencia 1**")
            rec = item['recommendations'][0]
            img_file = str(rec['image_file'])
            if not os.path.isabs(img_file) and not img_file.startswith("http"):
                img_path = os.path.join("imagenes_lanas", img_file)
            else:
                img_path = img_file
            
            img_b64 = get_base64_image(img_path) if os.path.exists(img_path) else None
            img_src = f"data:image/jpeg;base64,{img_b64}" if img_b64 else rec.get('image_url', '')

            st.markdown(f"""
            <div class="product-card">
                <img src="{img_src}" style="max-width: 100%; height: 150px; object-fit: contain; border-radius: 4px;">
                <div class="product-name-new">{rec['name']}</div>
                <div class="product-match-new">Match: {rec['score']:.1f}%</div>
                <a href="https://piccolo.com.co/producto/lana-vellon-individual/" class="custom-button">
                    🛍️ Ver Producto
                </a>
            </div>
            """, unsafe_allow_html=True)

        # Columna 3: Sugerencia 2
        with cols[2]:
            st.markdown("**Sugerencia 2**")
            rec = item['recommendations'][1]
            img_file = str(rec['image_file'])
            if not os.path.isabs(img_file) and not img_file.startswith("http"):
                img_path = os.path.join("imagenes_lanas", img_file)
            else:
                img_path = img_file
            
            img_b64 = get_base64_image(img_path) if os.path.exists(img_path) else None
            img_src = f"data:image/jpeg;base64,{img_b64}" if img_b64 else rec.get('image_url', '')

            st.markdown(f"""
            <div class="product-card">
                <img src="{img_src}" style="max-width: 100%; height: 150px; object-fit: contain; border-radius: 4px;">
                <div class="product-name-new">{rec['name']}</div>
                <div class="product-match-new">Match: {rec['score']:.1f}%</div>
                <a href="https://piccolo.com.co/producto/lana-vellon-individual/" class="custom-button">
                    🛍️ Ver Producto
                </a>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr style="border: 0.5px solid #eee; margin: 30px 0;">', unsafe_allow_html=True)

    # Completado
    status_text.markdown("""
    <div class="step-indicator">
        <div class="step-number" style="background: #2d6a4f;">✓</div>
        <div class="step-text" style="color: #2d6a4f; font-weight: 600;">¡Análisis completado!</div>
    </div>
    """, unsafe_allow_html=True)
