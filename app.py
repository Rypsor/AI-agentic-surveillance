# app.py (Versión 5.0 - Doble Pipeline: Monitoreo y Búsqueda)
import streamlit as st
import os
import tempfile
from ultralytics import YOLO
from dotenv import load_dotenv
import google.generativeai as genai

from src.agents.interpreted import interpret_and_dispatch
from src.pipelines import (
    run_monitor_pipeline,
    run_search_pipeline
)


# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Sistema de Vigilancia IA", layout="wide")

# --- 1. CONFIGURACIÓN INICIAL Y CARGA DE MODELOS ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("No se encontró la clave GEMINI_API_KEY. Asegúrate de crear un archivo .env.")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)
TEXT_MODEL = os.getenv("TEXT_MODEL", "gemini-2.0-flash")
VIDEO_MODEL = os.getenv("VIDEO_MODEL", "gemini-2.0-flash")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data/output_captures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

@st.cache_resource
def load_models():
    """Carga los modelos de IA una sola vez."""
    with st.spinner("Cargando modelos de IA (esto solo sucede la primera vez)..."):
        intention_to_model_path = {
            "accidente": os.path.join(BASE_DIR, "weights", "best_accident.pt"),
            "fuego": os.path.join(BASE_DIR, "weights", "best_fire.pt"),
            "general": os.path.join(BASE_DIR, "weights", "best_general.pt")
        }
        models = {name: YOLO(path) for name, path in intention_to_model_path.items()}
        vision_model = genai.GenerativeModel(VIDEO_MODEL)
        text_model = genai.GenerativeModel(TEXT_MODEL)

        agents = {
            "vision_model": vision_model,
            "text_model": text_model
        }
        return models, agents

detection_models, agents = load_models()

# --- LÓGICA PRINCIPAL DE LA APP ---
st.title("👁️ Sistema de Vigilancia con Agentes de IA v5.1")
st.markdown("**Modos duales:** Monitorea eventos o busca objetos con características específicas.")

col1, col2 = st.columns([2, 3])
with col1:
    st.header("1. Carga tu Video")
    uploaded_file = st.file_uploader("Selecciona un archivo", type=["mp4", "mov", "avi"])
with col2:
    st.header("2. Define la Tarea y Configuración")
    user_intention = st.text_area("Describe la intención:", "Encuentra todos los coches de color rojo.", height=100)

    sub_col1, sub_col2, sub_col3 = st.columns(3)
    with sub_col1:
        confidence = st.slider("Confianza Detección", 0.1, 1.0, 0.4, 0.05)
    with sub_col2:
        # Modifiqué el help text para que sea más claro
        frame_skip = st.slider("Analizar 1/N frames", 1, 30, 5, help="Afecta a ambos modos: Monitoreo y Búsqueda")
    with sub_col3:
        capture_duration_option = st.selectbox("Duración Captura", ["5 segundos", "10 segundos"], help="Solo para modo Monitoreo")
        capture_duration = int(capture_duration_option.split(" ")[0])

if st.button("🚀 Procesar Video", use_container_width=True):
    # --- PASO 1: Validación de entradas del usuario ---
    if not uploaded_file:
        st.warning("Por favor, sube un archivo de video.")
    elif not user_intention or not user_intention.strip():
        st.warning("Por favor, describe una intención de vigilancia.")
    else:
        # Si las validaciones son correctas, se ejecuta el resto.

        # --- PASO 2: Guardar el video y preparar la UI ---
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            video_path = tfile.name

        st.header("📊 Resultados del Análisis")
        progress_bar = st.progress(0, text="Iniciando...")
        status_text = st.empty()

        # --- PASO 3: Llamar al Agente 1 (Despachador) ---
        # Esta llamada se hace UNA SOLA VEZ, al principio.
        dispatch_result, error, raw_response = interpret_and_dispatch(agents["text_model"], user_intention)
        if error:
            st.error(f"Error del Agente Despachador: {error}")
            st.text_area(raw_response, height=150)
            status_text.text("Error.")
            progress_bar.empty()
        

        # --- PASO 4: Ejecutar el pipeline correspondiente ---
        if not dispatch_result or "workflow" not in dispatch_result:
            st.error("El Agente Despachador no pudo determinar un flujo de trabajo. Intenta reformular tu solicitud.")
            status_text.text("Error.")
            progress_bar.empty()

        elif dispatch_result["workflow"] == "monitor":
            run_monitor_pipeline(
                st=st,
                agents=agents,
                detection_models=detection_models,
                video_path=video_path,
                user_intention=user_intention,
                surveillance_config=dispatch_result["config"],
                confidence_threshold=confidence,
                frame_processing_rate=frame_skip,
                capture_duration=capture_duration,
                progress_bar=progress_bar,
                status_text=status_text
            )

        elif dispatch_result["workflow"] == "search":
            config = dispatch_result["config"]
            run_search_pipeline(
                st=st,
                agents=agents,
                detection_models=detection_models,
                video_path=video_path,
                base_class=config.get("base_class", ""),
                specific_property=config.get("specific_property", ""),
                confidence_threshold=confidence,
                frame_processing_rate=frame_skip, # Pasamos el parámetro
                progress_bar=progress_bar,
                status_text=status_text
            )

        # --- PASO 5: Limpiar el archivo de video temporal ---
        os.remove(video_path)


# --- INSTRUCCIONES PARA EL USUARIO ---
st.markdown("---")
st.subheader("Instrucciones:")
st.markdown("""
1.  **Configura tu clave API de Gemini:** Crea un archivo `.env` en la raíz de tu proyecto con el contenido `GEMINI_API_KEY='tu_clave_api_aqui'`.
2.  **Carga tu Video:** Sube un archivo de video (MP4, MOV, AVI).
3.  **Define la Tarea:** En el área de texto, describe claramente lo que el sistema debe vigilar (ej. "identificar **accidente** de transito", "detectar **fuego**", "buscar **persona** sospechosa").
    * **¡Importante!** Intenta usar palabras clave que coincidan con tus modelos YOLO si tienes modelos específicos.
4.  **Ajusta "Confianza Detección (YOLO)":** Umbral de confianza para las detecciones de YOLO. Un valor más alto significa menos falsos positivos.
5.  **Ajusta "Analizar 1/N frames":** Controla la frecuencia del análisis de **YOLO**.
6.  **Ajusta "Duración de Captura Temporal":** Define cuánto tiempo (antes y después) se analizará el video alrededor de una posible detección.
7.  **Procesar:** Haz clic en "🚀 Procesar Video y Analizar Eventos".
    * El **Agente 1** interpretará tu intención.
    * **YOLO** analizará el video fotograma a fotograma. Se intentará usar un modelo YOLO específico si tu intención lo sugiere (ej. "accidente" -> `best_accident.pt`). Si no hay uno específico, usará el modelo `general`.
    * **Solo si YOLO detecta algo relevante** (coincidencia con palabras clave), el fotograma se pasará al **Agente 2**.
    * Si el **Agente 2** clasifica una posible emergencia, el **Agente 3** realizará un análisis temporal para confirmarla o refutarla.
""")
st.warning("El análisis de video con modelos de IA es intensivo. Usa videos cortos para pruebas. Un mayor número de frames (menor `Analizar 1/N`) y una mayor `Duración de Captura Temporal` aumentarán el costo y el tiempo de procesamiento.")
