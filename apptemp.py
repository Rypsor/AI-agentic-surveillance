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
if not GEMINI_API_KEY:
    GEMINI_API_KEY = "AIzaSyBvDu-2aYkegWsDfTS99OZqgOwV5JjC1Pk"
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
text_model_for_agent1, vision_model_for_agent2_3, loaded_yolo_models = load_ai_models()


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
        confidence = st.slider("Confianza Detección", 0.1, 1.0, 0.4, 0.05)
    with sub_col2:
        # Este slider es clave para el procesamiento de video
        frame_skip = st.slider("Analizar 1/N frames", 1, 30, 5, help="Define cuántos fotogramas se saltan entre análisis.")
    with sub_col3:
        capture_duration_option = st.selectbox("Duración Captura", ["5 segundos", "10 segundos"], help="Solo para modo Monitoreo")
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
        processed_frames_buffer = []

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

            # --- Seleccionar el modelo YOLO adecuado basado en el objetivo del Agente 1 ---
            selected_yolo_model = None
            goal_lower = goal_identification_result.get('goal', '').lower()

            if "accidente" in goal_lower and loaded_yolo_models.get("accidente"):
                selected_yolo_model = loaded_yolo_models["accidente"]
                yolo_model_name = "específico de accidentes"
            elif "fuego" in goal_lower and loaded_yolo_models.get("fuego"):
                selected_yolo_model = loaded_yolo_models["fuego"]
                yolo_model_name = "específico de fuego"
            elif loaded_yolo_models.get("general"): # Usar el modelo general como fallback
                selected_yolo_model = loaded_yolo_models["general"]
                yolo_model_name = "general"
            else:
                st.error("No se pudo cargar ningún modelo YOLO adecuado para la intención o como fallback. El filtro YOLO no funcionará.")
                # Si no hay YOLO, el sistema continuará sin filtro YOLO
                selected_yolo_model = None
                yolo_model_name = "ninguno (deshabilitado)"

            st.info(f"YOLO seleccionado para esta tarea: **{yolo_model_name}**.")


            # --- PASO 4: Procesar el video con YOLO y Agente 2/3 ---
            st.subheader("🔍 Procesando Video con YOLO y Agentes...")

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

               # Procesar solo 1 de cada 'frame_skip' fotogramas con YOLO
                if frame_count % frame_skip == 0:
                    status_text.text(f"YOLO: Analizando fotograma {frame_count}/{total_frames}...")
                    progress_bar.progress(int(frame_count / total_frames * 100))

                    yolo_detections_found = False
                    if selected_yolo_model: # Solo si un modelo YOLO se seleccionó y cargó
                        try:
                            yolo_results = selected_yolo_model.predict(source=frame, conf=confidence, verbose=False)

                            detected_class_names = []
                            if yolo_results and len(yolo_results) > 0:
                                for r in yolo_results:
                                    if r.boxes: # Asegurarse de que haya cajas de detección
                                        for c in r.boxes.cls:
                                            detected_class_names.append(selected_yolo_model.names[int(c)])

                            keywords_lower = [k.lower() for k in goal_identification_result.get('keywords', [])]
                            detected_class_names_lower = [n.lower() for n in detected_class_names]

                            # La lógica de coincidencia ahora es más robusta
                            # Busca si alguna palabra clave está contenida en el nombre de la clase detectada
                            # O si el nombre de la clase detectada está contenido en la palabra clave
                            for keyword in keywords_lower:
                                if any(keyword in class_name or class_name in keyword for class_name in detected_class_names_lower):
                                    yolo_detections_found = True
                                    st.info(f"YOLO detectó '{keyword}' en el fotograma {frame_count}. Pasando al Agente 2.")
                                    break

                            if not yolo_detections_found:
                                status_text.text(f"YOLO: No se encontraron objetos relevantes en el fotograma {frame_count}. Saltando Agente 2/3.")

                        except Exception as e:
                            st.warning(f"Error durante la inferencia YOLO en el fotograma {frame_count}: {e}. Procediendo sin filtro YOLO para este frame.")
                            # Si YOLO falla, podemos decidir si queremos pasar al Agente 2 de todos modos
                            yolo_detections_found = False
                    else: # Si no hay modelo YOLO seleccionado, el sistema procede como si YOLO hubiera detectado algo
                        yolo_detections_found = True
                        st.warning("No hay modelo YOLO activo. El sistema procederá directamente al Agente 2 para cada fotograma.")

                    if yolo_detections_found:
                        # --- Llamar al Agente 2 ---
                        status_text.text(f"Agente 2: Analizando fotograma {frame_count} para clasificación...")
                        analysis_result_agent2 = analyze_image_for_goal(
                            vision_model_for_agent2_3,
                            temp_frame_path,
                            goal_identification_result
                        )

                        if analysis_result_agent2.get("classification") == "Emergencia":
                            event_detected_by_agent2 = True
                            detected_frame_index = current_frame_index
                            st.balloons()
                            st.subheader("🚨 ¡POSIBLE EMERGENCIA DETECTADA POR AGENTE 2! 🚨")
                            st.write(f"**Detectado en el fotograma:** {frame_count} (índice real: {detected_frame_index})")
                            st.write(f"**Descripción Agente 2:** {analysis_result_agent2.get('description')}")
                            st.write(f"**Resumen Agente 2:** {analysis_result_agent2.get('summary')}")
                            st.write(f"**Servicio Sugerido Agente 2:** {analysis_result_agent2.get('service')}")

                            if analysis_result_agent2.get("annotated_image"):
                                st.image(analysis_result_agent2["annotated_image"], caption=f"Fotograma {frame_count} con detecciones del Agente 2", use_column_width=True)
                            else:
                                st.warning("No se pudo obtener la imagen anotada del Agente 2.")
                            break

            cap.release()

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
