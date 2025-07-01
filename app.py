import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
import google.generativeai as genai
import json
import cv2 # Importar OpenCV
from PIL import Image # Para manejar las imágenes del video y las anotadas
from ultralytics import YOLO

# Importar los agentes
from src.agents.goal_identification_agent import identify_classification_goal
from src.agents.image_analyzer import analyze_image_for_goal
from src.agents.video_analyzer import analyze_temporal_context

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Sistema de Vigilancia IA", layout="wide")

# --- 0. Configuración de Directorios y Claves API ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TEXT_MODEL = os.getenv("TEXT_MODEL")
VIDEO_MODEL = os.getenv("VIDEO_MODEL")

# --- Mapeo de Intenciones a Modelos YOLO ---
# Define tus modelos YOLO y para qué intención son más adecuados.
# Asegúrate de que los archivos .pt existan en la carpeta 'weights'.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODELS_CONFIG = {
    "accidente": "best_accident.pt", # Modelo específico para accidentes
    "fuego": "best_fire.pt",         # Modelo específico para fuego
    "general": "best_general.pt"  # Modelo general para personas
}
YOLO_WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

if not GEMINI_API_KEY:
    st.error("No se encontró la clave GEMINI_API_KEY. Asegúrate de crear un archivo .env en la raíz del proyecto con `GEMINI_API_KEY='tu_clave_aqui'`.")
    st.stop()

# Configurar la API de Gemini globalmente
genai.configure(api_key=GEMINI_API_KEY)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_captures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Directorio para la salida de configuraciones de agentes (para el JSON del Agente 1)
AGENT_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "config_output")
os.makedirs(AGENT_OUTPUT_DIR, exist_ok=True)
GOAL_CONFIG_FILE = os.path.join(AGENT_OUTPUT_DIR, "current_goal_config.json")


# --- 1. CARGA DE MODELOS (Usando st.cache_resource para cargar una sola vez) ---
@st.cache_resource
def load_ai_models():
    """
    Carga los modelos de IA necesarios para los agentes.
    Cargamos el modelo de texto para Agente 1 y el multimodal para Agente 2.
    """
    with st.spinner("Cargando modelos de IA (esto solo sucede la primera vez)..."):
        
        # Modelo de texto para el Agente 1 (identificación de objetivo)
        text_model = genai.GenerativeModel(TEXT_MODEL)

        # Modelo multimodal para el Agente 2 (análisis de imagen y visión)
        vision_model = genai.GenerativeModel(VIDEO_MODEL)

       # Cargar todos los modelos YOLO preentrenados
        loaded_yolo_models = {}
        for intent, filename in YOLO_MODELS_CONFIG.items():
            model_path = os.path.join(YOLO_WEIGHTS_DIR, filename)
            if os.path.exists(model_path):
                try:
                    loaded_yolo_models[intent] = YOLO(model_path)
                    st.success(f"Modelo YOLO '{intent}' cargado desde: {model_path}")
                except Exception as e:
                    st.warning(f"Error al cargar el modelo YOLO '{intent}' desde '{model_path}': {e}")
                    loaded_yolo_models[intent] = None # Marcar como no cargado
            else:
                st.warning(f"No se encontró el archivo de pesos para YOLO '{intent}' en: {model_path}. Este modelo no se cargará.")
                loaded_yolo_models[intent] = None # Marcar como no cargado
        
        # Asegurarse de que el modelo general esté disponible si los específicos fallan o no se encuentran
        if "general" not in loaded_yolo_models or loaded_yolo_models["general"] is None:
            st.error("¡Advertencia Crítica! El modelo YOLO 'general' no se pudo cargar o no se encontró. El filtro YOLO será limitado.")

        return text_model, vision_model, loaded_yolo_models # Retornar todos los modelos cargados

        

# Cargar los modelos al inicio de la aplicación Streamlit
text_model_for_agent1, vision_model_for_agent2_3 = load_ai_models()


# --- 2. LÓGICA PRINCIPAL DE LA APP (Interfaz de Usuario) ---
st.title("👁️ Sistema de Vigilancia con Agentes de IA v0.1")
st.markdown("**Modos duales:** Monitorea eventos o busca objetos con características específicas.")

col1, col2 = st.columns([2, 3])

with col1:
    st.header("1. Carga tu Video")
    uploaded_file = st.file_uploader("Selecciona un archivo", type=["mp4", "mov", "avi"])

with col2:
    st.header("2. Define la Tarea y Configuración")
    user_intention = st.text_area(
        "Describe la intención de vigilancia para el sistema:",
        "Esta es una cámara de vigilancia de tránsito, tu tarea es identificar accidentes de transito.",
        height=100
    )
    
    sub_col1, sub_col2, sub_col3 = st.columns(3)
    with sub_col1:
        # La confianza podría usarse para filtrar detecciones de YOLO, pero aquí solo se muestra.
        confidence = st.slider("Confianza Detección (No funcional en esta fase)", 0.1, 1.0, 0.4, 0.05, disabled=True)
    with sub_col2:
        # Este slider es clave para el procesamiento de video
        frame_skip = st.slider("Analizar 1/N frames", 1, 30, 5, help="Define cuántos fotogramas se saltan entre análisis.")
    with sub_col3:
        capture_duration_option = st.selectbox("Duración Captura (No funcional en esta fase)", ["5 segundos", "10 segundos"], help="Solo para modo Monitoreo")
        capture_duration = int(capture_duration_option.split(" ")[0])

# Botón para procesar
if st.button("🚀 Procesar Video y Analizar Eventos", use_container_width=True):
    # --- PASO 1: Validación de entradas del usuario ---
    if not uploaded_file:
        st.warning("Por favor, sube un archivo de video.")
    elif not user_intention or not user_intention.strip():
        st.warning("Por favor, describe una intención de vigilancia.")
    else:
        st.header("📊 Resultados del Análisis")
        progress_bar = st.progress(0, text="Iniciando procesamiento...")
        status_text = st.empty()
        
        video_path = None # Inicializar para asegurar que esté definida
        temp_frames_dir = None # Directorio temporal para los frames del Agente 3

        try:
            # --- PASO 2: Guardar el video temporalmente ---
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_file.read())
                video_path = tfile.name
            
            st.success(f"Video '{uploaded_file.name}' cargado exitosamente.")
            
            # --- PASO 3: Llamar al Agente 1 (Identificación de Objetivo) ---
            status_text.text("Agente 1: Analizando intención de vigilancia...")
            progress_bar.progress(10)
            goal_identification_result = identify_classification_goal(text_model_for_agent1, user_intention)
            
            if not goal_identification_result or not goal_identification_result.get('goal'):
                st.error("El Agente 1 no pudo identificar el objetivo de clasificación. No se puede continuar.")
                status_text.text("Error en Agente 1.")
                progress_bar.empty()
                st.stop() # Detener la ejecución
            
            st.subheader("✅ Resultado del Agente 1:")
            st.write(f"**Objetivo Principal:** `{goal_identification_result.get('goal', 'Desconocido')}`")
            st.write(f"**Palabras Clave:** `{', '.join(goal_identification_result.get('keywords', []))}`")
            
            # Guardar el JSON del objetivo para referencia
            try:
                with open(GOAL_CONFIG_FILE, "w", encoding="utf-8") as f:
                    json.dump(goal_identification_result, f, indent=4)
                st.info(f"Objetivo de clasificación guardado en: `{GOAL_CONFIG_FILE}`")
            except Exception as e:
                st.warning(f"No se pudo guardar el JSON del objetivo: {e}")

            # --- PASO 4: Procesar el video con el Agente 2 (Análisis Multimodal) ---
            st.subheader("🔍 Procesando Video con Agente 2...")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Error al abrir el video. Asegúrate de que el archivo sea válido.")
                st.stop()

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            frame_count = 0
            event_detected_by_agent2 = False
            analysis_result_agent2 = None
            detected_frame_index = -1 # Para almacenar el índice del frame donde se detectó

            # Lista para almacenar las rutas de los frames procesados, para el Agente 3
            # Guardamos todos los frames procesados por si la duración es grande y necesitamos ir hacia atrás
            processed_frames_buffer = [] 
            
            while cap.isOpened() and not event_detected_by_agent2:
                ret, frame = cap.read()
                if not ret:
                    break # Fin del video

                current_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1 # Índice real del frame (0-based)

                frame_count += 1

                # Crear un archivo temporal para el frame actual
                temp_frame_path = tempfile.NamedTemporaryFile(delete=False, suffix=f'_frame_{current_frame_index}.jpg', dir=OUTPUT_DIR).name
                cv2.imwrite(temp_frame_path, frame)
                processed_frames_buffer.append(temp_frame_path)

                
                # Procesar solo 1 de cada 'frame_skip' fotogramas
                if frame_count % frame_skip == 0:
                    status_text.text(f"Agente 2: Analizando fotograma {frame_count}/{total_frames}...")
                    progress_bar.progress(int(frame_count / total_frames * 100))

                    
                    # --- Llamar al Agente 2 ---
                    analysis_result_agent2 = analyze_image_for_goal(
                        vision_model_for_agent2_3, # Usamos el modelo de visión para Agente 2
                        temp_frame_path, # Pasamos la ruta del frame actual
                        goal_identification_result # Pasar el JSON del Agente 1
                    )

                    if analysis_result_agent2.get("classification") == "Emergencia":
                        event_detected_by_agent2 = True
                        detected_frame_index = current_frame_index  # Guardar el índice del frame detectado
                        st.balloons()
                        st.subheader("🚨 ¡EMERGENCIA DETECTADA! 🚨")
                        st.write(f"**Detectado en el fotograma:** {frame_count}")
                        st.write(f"**Descripción Agente 2:** {analysis_result_agent2.get('description')}")
                        st.write(f"**Resumen Agente 2:** {analysis_result_agent2.get('summary')}")
                        st.write(f"**Servicio Sugerido Agente 2:** {analysis_result_agent2.get('service')}")
                        
                        # Mostrar la imagen anotada del Agente 2
                        if analysis_result_agent2.get("annotated_image"):
                            st.image(analysis_result_agent2["annotated_image"], caption=f"Fotograma {frame_count} con detecciones del Agente 2", use_column_width=True)
                        else:
                            st.warning("No se pudo obtener la imagen anotada del Agente 2.")
                        break # Detener el bucle de video, ya tenemos una detección inicial
            
            
            cap.release() # Liberar el objeto de captura de video

            if not event_detected_by_agent2:
                st.info("No se detectó ninguna emergencia en el video según el objetivo definido.")
                progress_bar.progress(100)
                status_text.text("Análisis de video completado.")
            
            else:
                # --- PASO 5: Preparar frames para el Agente 3 (Análisis de Contexto Temporal) ---
                st.subheader("⏱️ Agente 3: Analizando Contexto Temporal...")
                status_text.text("Agente 3: Recopilando fotogramas de contexto...")
                
                frames_per_side = int(capture_duration * fps / 2) # Número de frames antes y después
                
                # Calcular los índices de los frames que necesitamos
                start_frame_index = max(0, detected_frame_index - frames_per_side)
                end_frame_index = min(total_frames - 1, detected_frame_index + frames_per_side)
                
                # Asegurarse de que el frame detectado esté en la lista
                temporal_image_paths = []
                for idx in range(start_frame_index, end_frame_index + 1):
                    # Solo añadir si el frame actual está en nuestro buffer de frames procesados
                    # Asumimos que processed_frames_buffer contiene los frames en orden de índice
                    if 0 <= idx < len(processed_frames_buffer):
                        temporal_image_paths.append(processed_frames_buffer[idx])
                
                if not temporal_image_paths:
                    st.error("No se pudieron recopilar fotogramas de contexto temporal para el Agente 3.")
                else:
                    status_text.text(f"Agente 3: Analizando {len(temporal_image_paths)} fotogramas para confirmación...")
                    progress_bar.progress(90)

                    # --- Llamar al Agente 3 ---
                    temporal_analysis_result = analyze_temporal_context(
                        vision_model_for_agent2_3, # Usamos el mismo modelo de visión para Agente 3
                        temporal_image_paths,
                        analysis_result_agent2,       # Resultado del Agente 2
                        goal_identification_result    # Resultado del Agente 1
                    )

                    st.subheader("✨ Resultado Final del Agente 3 (Confirmación Temporal):")
                    if temporal_analysis_result.get("is_confirmed_emergency"):
                        st.success("✅ ¡EMERGENCIA CONFIRMADA POR ANÁLISIS TEMPORAL! ✅")
                        st.write(f"**Resumen Final:** {temporal_analysis_result.get('final_summary')}")
                        st.write(f"**Servicio Sugerido Final:** {temporal_analysis_result.get('final_service')}")
                        st.write(f"**Razonamiento:** {temporal_analysis_result.get('reasoning')}")
                        st.markdown("---")
                        st.warning("¡Alerta enviada a los servicios de emergencia!")
                    else:
                        st.info("❌ La emergencia **NO fue confirmada** por el análisis temporal.")
                        st.write(f"**Motivo:** {temporal_analysis_result.get('final_summary')}")
                        st.write(f"**Razonamiento:** {temporal_analysis_result.get('reasoning')}")
                        st.warning("Alerta descartada para evitar falsos positivos.")
                
                progress_bar.progress(100)
                status_text.text("Análisis temporal completado. Proceso finalizado.")


        except Exception as e:
            st.error(f"Ocurrió un error durante el procesamiento del video o el Agente 2: {e}")
            st.warning("Asegúrate de que tus modelos de IA estén cargados correctamente y de que la clave API de Gemini sea válida.")
            progress_bar.empty()
            status_text.text("Error.")
        finally:
            # --- PASO 6: Limpiar el archivo de video temporal ---
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
                st.info("Archivo de video temporal limpiado.")

            # Limpiar los frames temporales generados por el procesamiento
            for frame_path in processed_frames_buffer:
                if os.path.exists(frame_path):
                    os.remove(frame_path)
            st.info(f"{len(processed_frames_buffer)} fotogramas temporales limpiados.")


# --- INSTRUCCIONES PARA EL USUARIO ---
st.markdown("---")
st.subheader("Instrucciones:")
st.markdown("""
1.  **Configura tu clave API de Gemini:** Crea un archivo `.env` en la raíz de tu proyecto con el contenido `GEMINI_API_KEY='tu_clave_api_aqui'`.
2.  **Carga tu Video:** Sube un archivo de video (MP4, MOV, AVI).
3.  **Define la Tarea:** En el área de texto, describe claramente lo que el sistema debe vigilar (ej. "identificar accidentes de transito", "detectar armas", "buscar fuego o humo").
4.  **Ajusta "Analizar 1/N frames":** Esto controla la frecuencia con la que se analizan los fotogramas del video. Un número más alto es más rápido pero menos preciso; uno más bajo es más lento pero más exhaustivo.
5.  **Procesar:** Haz clic en "🚀 Procesar Video y Analizar Eventos".
    * El **Agente 1** interpretará tu intención.
    * El **Agente 2** analizará el video, fotograma a fotograma (según `1/N frames`).
    * Si se detecta una **emergencia**, el análisis se detiene y los resultados se muestran en pantalla, incluyendo la imagen del momento de la detección con anotaciones.
""")
st.warning("Ten en cuenta que el análisis de video con modelos de IA puede ser intensivo y consumir tokens de la API. Usa videos cortos para pruebas.")