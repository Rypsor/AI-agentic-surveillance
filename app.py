

import streamlit as st
import os
import re
import cv2
import json
import datetime
import tempfile
from PIL import Image
from ultralytics import YOLO
import numpy as np
from collections import deque

from dotenv import load_dotenv
import google.generativeai as genai



# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Sistema de Vigilancia IA", layout="wide")

# --- 1. CONFIGURACIÓN INICIAL Y CARGA DE MODELOS ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("No se encontró la clave GEMINI_API_KEY. Asegúrate de crear un archivo .env.")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output") # Renombrado para ser más genérico
CLIPS_DIR = os.path.join(OUTPUT_DIR, "clips") # Nueva carpeta para los videoclips
os.makedirs(CLIPS_DIR, exist_ok=True) # Crea ambas carpetas si no existen

# --- NUEVA FUNCIÓN AUXILIAR PARA EXTRAER SEGMENTOS DE VIDEO ---
def extract_video_segment(input_path: str, output_path: str, start_frame: int, end_frame: int, fps: float):
    """
    Extrae un segmento de un video original (con todos sus fotogramas)
    y lo guarda en un nuevo archivo.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video de entrada en {input_path}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    current_frame = start_frame
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret: break
        writer.write(frame)
        current_frame += 1

    cap.release()
    writer.release()
    
    return output_path



# ==============================================================================
# ====== REEMPLAZA TU FUNCIÓN load_models EXISTENTE CON ESTA ===================
# ==============================================================================

@st.cache_resource
def load_models():
    """
    Carga los modelos de IA una sola vez, leyendo los nombres de los modelos 
    de Gemini desde el archivo .env.
    """
    with st.spinner("Cargando modelos de IA (esto solo sucede la primera vez)..."):
        # Carga de modelos YOLO (sin cambios)
        intention_to_model_path = {
            "accidente": os.path.join(BASE_DIR, "weights", "best_accident.pt"),
            "fuego": os.path.join(BASE_DIR, "weights", "best_fire.pt"),
            "general": os.path.join(BASE_DIR, "weights", "best_general.pt")
        }
        models = {name: YOLO(path) for name, path in intention_to_model_path.items()}
        
        # --- CAMBIOS AQUÍ ---
        # 1. Leer los nombres de los modelos desde el archivo .env
        #    Se proporciona un valor por defecto ("gemini-1.5-flash") por si las variables no existen en .env
        vision_model_name = os.getenv("VISION_MODEL", "gemini-1.5-flash") # <<< NUEVO
        text_model_name = os.getenv("TEXT_MODEL", "gemini-1.5-flash")   # <<< NUEVO

        st.info(f"Cargando modelo de visión: `{vision_model_name}`") # Opcional: para depuración
        st.info(f"Cargando modelo de texto: `{text_model_name}`")   # Opcional: para depuración

        # 2. Inicializar los modelos de Gemini usando los nombres leídos
        vision_model = genai.GenerativeModel(vision_model_name) # <<< MODIFICADO
        text_model = genai.GenerativeModel(text_model_name)     # <<< MODIFICADO
        
        return models, vision_model, text_model


detection_models, vision_model, text_model = load_models()


# --- 2. DEFINICIÓN DE AGENTES DE IA ---

def interpret_and_dispatch(user_intention: str) -> dict:
    """Agente 1 Despachador: Decide qué pipeline usar y extrae los parámetros."""
    
    # --- BASE DE CONOCIMIENTO AMPLIADA ---
    # Ahora incluye TODAS las clases de tus modelos y más sinónimos.
    knowledge_base = """
    ### Modelos y Clases (MAPA FUNDAMENTAL) ###
    - Modelo "accidente": Contiene las clases ['accident', 'moderate', 'severe']. Se especializa en detectar choques y su gravedad.
    - Modelo "fuego": Contiene las clases ['fire', 'smoke']. Se especializa en detectar incendios.
    - Modelo "general": Contiene todas las demás clases: ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck', 'cat', 'dog'].

    ### Conceptos y Sinónimos (TRADUCTOR) ###
    - "mascotas", "animales": ["dog", "cat"]
    - "vehículos", "transporte": ["car", "bus", "truck", "train", "motorcycle", "bicycle"]
    - "personas", "gente", "intrusos": ["person"]
    - "incendio", "llama", "fuego": ["fire"]
    - "humo": ["smoke"]
    - "accidente", "choque", "colisión": ["accident", "severe"]
    - "carros", "coches", "autos": ["car"]
    - "camioneta", "camión": ["truck"]
    - "moto", "motocicleta": ["motorcycle"]
    - "bici", "bicicleta": ["bicycle"]
    - "transporte público": ["bus", "train"]

    ### Reglas de Decisión Clave ###
    1.  **Prioridad de Modelos:** Si la solicitud menciona "fuego", "humo", "incendio" o "accidente", SIEMPRE debes usar los modelos especializados ("fuego" o "accidente") para esas clases.
    2.  **Default a 'search':** Si el usuario pide buscar algo con una propiedad visual (color, tamaño, acción, estado como "estacionado", "roto"), el workflow es "search".
    3.  **Default a 'monitor':** Si el usuario usa palabras como "vigila", "alerta si", "avísame cuando", "monitorea", el workflow es "monitor".
    4. **Generalización:** nunca uses "general" para buscar "car" cuando de accidentes se trata, usa "accidente" directamente.
    """
    
    prompt = f"""
    Eres un despachador de IA experto que dirige las solicitudes a dos pipelines: "monitor" (vigilancia en tiempo real) y "search" (encontrar objetos específicos).
    Tu única tarea es analizar la solicitud del usuario basándote en tu base de conocimiento y responder ÚNICAMENTE con un objeto JSON válido que contenga el pipeline y su configuración. No añadas explicaciones ni texto adicional.

    ### Formato de Salida (Estricto) ###
    - Para "monitor": {{"workflow": "monitor", "config": {{"nombre_del_modelo": ["clase_1", "clase_2"]}}}}
    - Para "search": {{"workflow": "search", "config": {{"base_class": "nombre_de_la_clase", "specific_property": "descripción de la propiedad"}}}}

    ### Ejemplos Detallados ###
    - Solicitud: "vigila si hay mascotas o un accidente"
      JSON: {{"workflow": "monitor", "config": {{"general": ["dog", "cat"], "accidente": ["accident"]}}}}
    - Solicitud: "encuentra todos los camiones de color azul"
      JSON: {{"workflow": "search", "config": {{"base_class": "truck", "specific_property": "de color azul"}}}}
    - Solicitud: "quiero que busques cualquier indicio de fuego"
      JSON: {{"workflow": "search", "config": {{"base_class": "fire", "specific_property": "cualquier indicio"}}}}
    - Solicitud: "encuentra una bici que esté estacionada"
      JSON: {{"workflow": "search", "config": {{"base_class": "bicycle", "specific_property": "que esté estacionada"}}}}
    - Solicitud: "monitorea todos los vehículos y si hay humo"
      JSON: {{"workflow": "monitor", "config": {{"general": ["car", "bus", "truck", "train", "motorcycle", "bicycle"], "fuego": ["smoke"]}}}}

    ### Base de Conocimiento ###
    {knowledge_base}
    ---
    Solicitud del Usuario: "{user_intention}"
    ---
    JSON de Respuesta:
    """
    try:
        # Usar un modelo de texto potente es clave aquí
        # gemini-1.5-flash o superior es ideal
        response = text_model.generate_content(prompt)
        # Limpieza robusta de la respuesta para extraer solo el JSON
        cleaned_response = response.text.strip()
        match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            st.error("El Agente Despachador no devolvió un JSON válido.")
            st.text_area("Respuesta completa del modelo:", cleaned_response, height=150)
            return {}

    except Exception as e:
        st.error(f"Error crítico en el Agente Despachador: {e}")
        st.text_area("Respuesta del modelo que causó el error:", response.text if 'response' in locals() else "No se recibió respuesta.", height=150)
        return {}

def describe_scene_with_gemini(frames: list, user_intention: str, target_classes: list) -> dict:
    """Agente 2 (Guardia de Seguridad): Analiza una secuencia de fotogramas y genera un reporte JSON."""
    if not frames: 
        return {"error": "No se capturaron fotogramas para el análisis."}
        
    pil_images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]

    # --- PROMPT MEJORADO CON ROL ESPECÍFICO ---
    prompt = f"""
    Eres un guardia de seguridad experimentado y vigilante. Tu misión es analizar la siguiente secuencia de fotogramas de una cámara de seguridad para redactar un parte de situación.

    **Contexto de la Alerta:**
    - **Intención del Operador:** "{user_intention}"
    - **Detección Automática Inicial:** Se han identificado los siguientes objetos de interés: {target_classes}.

    **Tu Tarea:**
    Observa la secuencia de imágenes y describe la escena de manera objetiva, clara y concisa. Céntrate en los hechos observables.

    **Formato de Salida Obligatorio:**
    Responde ÚNICAMENTE con un objeto JSON válido que siga esta estructura:
    {{
        "resumen_escena": "Una descripción breve y general de la situación. Describe el entorno (ej. 'calle urbana de día', 'aparcamiento nocturno') y los actores principales.",
        "eventos_clave": [
            "Una lista de 2 a 4 eventos secuenciales que narran lo que ocurre. Sé específico. Por ejemplo: 'Un coche rojo se aproxima a otro vehículo detenido.', 'El coche rojo no frena e impacta por detrás.', 'Comienza a salir humo del capó de ambos vehículos.'"
        ],
        "evaluacion_inicial": "Tu evaluación profesional como guardia. Clasifica la situación (ej. 'Falsa alarma', 'Incidente menor', 'Posible emergencia', 'Emergencia confirmada') y justifica brevemente por qué."
    }}
    """
    
    try:
        # El prompt va primero, seguido de la lista de imágenes
        response = vision_model.generate_content([prompt] + pil_images)
        # Limpieza robusta del texto de respuesta para asegurar que sea un JSON válido
        cleaned_response = response.text.strip()
        
        # Encuentra el inicio y el fin del JSON para evitar texto extra
        json_start = cleaned_response.find('{')
        json_end = cleaned_response.rfind('}') + 1
        if json_start != -1 and json_end != 0:
            json_str = cleaned_response[json_start:json_end]
            return json.loads(json_str)
        else:
            return {"error": "El modelo no devolvió un JSON válido."}

    except Exception as e:
        # Devuelve el error específico para facilitar la depuración
        return {"error": f"Fallo en el reporte del Agente 2: {str(e)}"}

def verify_alarm_with_gemini(scene_report: dict, user_intention: str) -> dict:
    """
    Agente 3 (Jefe de Seguridad/Despachador): Analiza el reporte del guardia,
    decide si es una emergencia real, determina a qué autoridad notificar
    y redacta el mensaje de alerta.
    """
    if "error" in scene_report:
        # Si el reporte del Agente 2 ya tiene un error, lo propagamos.
        return {"veredicto": "ERROR_EN_REPORTE", "justificacion": scene_report["error"]}
        
    report_str = json.dumps(scene_report, indent=2, ensure_ascii=False)

    # --- PROMPT MEJORADO CON EL NUEVO ROL DE DESPACHADOR ---
    prompt = f"""
    Eres el Jefe de Seguridad de un centro de control. Has recibido el siguiente parte de situación de uno de tus guardias.
    Tu misión es tomar una decisión final: determinar si la situación requiere una intervención externa, decidir qué autoridad contactar (si aplica) y redactar un mensaje de alerta claro y conciso.

    **Contexto de la Tarea:**
    - **Intención Original del Operador:** "{user_intention}"
    - **Parte de Situación del Guardia (Agente 2):**
    {report_str}

    **Tus Responsabilidades:**
    1.  **Analiza el Veredicto Inicial:** Evalúa la conclusión de tu guardia a la luz de la intención original.
    2.  **Toma la Decisión Final:** Emite un veredicto. Los veredictos posibles son: 'ALARMA_CONFIRMADA', 'INCIDENTE_MENOR' o 'FALSA_ALARMA'.
    3.  **Determina la Autoridad Competente:** Si es 'ALARMA_CONFIRMADA', decide a quién notificar. Las opciones son: 'Policía', 'Bomberos', 'Servicios Médicos', 'Control de Animales' o 'Ninguna'.
    4.  **Redacta el Mensaje de Alerta:** Si se notifica a una autoridad, escribe un mensaje breve y directo para ellos.

    **Formato de Salida Obligatorio:**
    Responde ÚNICAMENTE con un objeto JSON válido que siga esta estructura. No añadas texto fuera del JSON.

    {{
      "veredicto_final": "Tu decisión final ('ALARMA_CONFIRMADA', 'INCIDENTE_MENOR', 'FALSA_ALARMA').",
      "justificacion": "Una explicación breve y profesional de por qué tomaste esa decisión, basándote en el reporte.",
      "autoridad_a_notificar": "La autoridad competente (por ejemplo 'Policía', o 'Bomberos', o 'Paramédicos', etc.) o 'Ninguna' si no es una emergencia.",
      "mensaje_de_alerta": "El mensaje para la autoridad, describe la escena muy brevemente y recomendaciones de seguridad brevemente. Si la autoridad es 'Ninguna', este campo debe ser 'No se requiere acción.'."
    }}

    **Ejemplo para un incendio:**
    {{
      "veredicto_final": "ALARMA_CONFIRMADA",
      "justificacion": "El reporte confirma la presencia de humo y llamas, lo que constituye una emergencia de incendio.",
      "autoridad_a_notificar": "Bomberos",
      "mensaje_de_alerta": "Alerta de Incendio: Se detecta fuego activo. Se observa humo denso y llamas visibles. Se requiere intervención inmediata."
    }}
    """
    
    try:
        response = text_model.generate_content(prompt)
        # Usamos la misma lógica de limpieza robusta para el JSON
        cleaned_response = response.text.strip()
        json_start = cleaned_response.find('{')
        json_end = cleaned_response.rfind('}') + 1
        if json_start != -1 and json_end != 0:
            json_str = cleaned_response[json_start:json_end]
            return json.loads(json_str)
        else:
            return {"veredicto_final": "ERROR_DE_MODELO", "justificacion": "El modelo no devolvió un JSON válido.", "autoridad_a_notificar": "Ninguna", "mensaje_de_alerta": ""}

    except Exception as e:
        return {"veredicto_final": "ERROR_DE_CODIGO", "justificacion": str(e), "autoridad_a_notificar": "Ninguna", "mensaje_de_alerta": ""}

def verify_property_with_gemini(image: Image, base_class: str, specific_property: str) -> bool:
    """Agente de Búsqueda: Verifica si una imagen cumple una propiedad específica."""
    
    # --- PROMPT MEJORADO ---
    # Este nuevo prompt es más directo, le da un rol al modelo y le exige una respuesta única.
    prompt = f"""
    Tu única tarea es analizar la imagen y responder a una pregunta con una sola palabra.

    Pregunta: ¿El objeto en esta imagen es un/a '{base_class}' que es/tiene la característica '{specific_property}'?

    Instrucción estricta: Responde ÚNICAMENTE con la palabra "SÍ" o la palabra "NO".
    No añadas ninguna explicación, frase, o puntuación. Tu respuesta debe ser una de esas dos palabras y nada más.
    """
    
    try:
        response = vision_model.generate_content([prompt, image])
        
        # --- LÓGICA DE COMPROBACIÓN MEJORADA ---
        # Usamos .strip() para eliminar espacios en blanco y comparamos la respuesta exacta.
        # Esto es más seguro que usar 'in' por si el modelo respondiera algo como "NO SE VE CLARO".
        return response.text.strip().upper() == "SÍ"
        
    except Exception as e:
        # Es buena práctica registrar el error si algo sale mal durante la llamada a la API
        st.warning(f"Error en la API de Gemini durante la verificación: {e}")
        return False


# --- 3. DEFINICIÓN DE PIPELINES ---

# No olvides importar 'deque' al principio de tu archivo app.py
from collections import deque

def run_monitor_pipeline(video_path: str, user_intention: str, surveillance_config: dict, confidence_threshold: float, frame_processing_rate: int, capture_duration: int, progress_bar, status_text):
    """
    Pipeline de monitoreo que continúa el análisis tras un incidente menor o falsa alarma,
    y se detiene solo ante una alarma confirmada o un error.
    """
    st.info(f"🧠 **Pipeline de Monitoreo Activado:** Configuración: `{json.dumps(surveillance_config)}`")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error al abrir el archivo de video en la ruta: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames_to_skip_for_analysis = max(1, int(fps * 0.5))
    num_frames_for_analysis = int(capture_duration * fps / frames_to_skip_for_analysis)
    num_before_for_analysis = num_frames_for_analysis // 2
    analysis_frame_buffer = deque(maxlen=num_before_for_analysis)

    frame_count = 0
    stop_processing = False # Controla la detención del bucle principal
    events_found_count = 0

    while cap.isOpened() and not stop_processing:
        ret, frame = cap.read()
        if not ret: break

        # El buffer se llena continuamente con frames de alta calidad para el análisis
        if frame_count % frames_to_skip_for_analysis == 0:
            analysis_frame_buffer.append(frame.copy())
        
        frame_count += 1
        
        # Actualizar la barra de progreso
        if frame_count % 10 == 0: 
            progress_bar.progress(frame_count / total_frames, text=f"Monitoreando... (Frame {frame_count}/{total_frames})")

        # Saltar frames para el procesamiento con YOLO
        if frame_count % frame_processing_rate != 0:
            continue
        
        for model_name, target_classes in surveillance_config.items():
            if model_name not in detection_models: continue
            
            model = detection_models[model_name]
            results = model(frame, conf=confidence_threshold, verbose=False)
            
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                detected_class_name = model.names[class_id]

                if detected_class_name.lower() in [cls.lower() for cls in target_classes]:
                    events_found_count += 1
                    
                    with st.spinner(f"🚨 ¡Posible evento '{detected_class_name}'! Analizando con IA (Evento #{events_found_count})..."):
                        
                        # Recopilar fotogramas para el análisis de IA (contexto antes y después)
                        collected_analysis_frames = list(analysis_frame_buffer)
                        num_after_to_collect = num_frames_for_analysis - len(collected_analysis_frames)
                        
                        temp_cap = cv2.VideoCapture(video_path)
                        temp_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                        for _ in range(num_after_to_collect):
                            for _ in range(frames_to_skip_for_analysis):
                                temp_cap.read()
                            ret_capture, next_frame = temp_cap.read()
                            if not ret_capture: break
                            collected_analysis_frames.append(next_frame.copy())
                        temp_cap.release()

                        # Análisis completo de los agentes
                        scene_report = describe_scene_with_gemini(collected_analysis_frames, user_intention, target_classes)
                        decision_result = verify_alarm_with_gemini(scene_report, user_intention)
                        veredicto = decision_result.get("veredicto_final", "ERROR_DESCONOCIDO")

                    # --- DECISIÓN GRANULAR: CONTINUAR, REGISTRAR O DETENER ---
                    justificacion = decision_result.get("justificacion", "No se proporcionó justificación.")

                    if veredicto == "FALSA_ALARMA":
                        with st.container():
                            st.success(f"✔️ **Falsa Alarma Descartada** (en frame {frame_count}). Reanudando monitoreo.")
                            st.info(f"**Justificación:** {justificacion}")
                            st.divider()
                        break 

                    elif veredicto == "INCIDENTE_MENOR":
                        with st.container():
                            st.warning(f"⚠️ **Incidente Menor Registrado** (en frame {frame_count}). El monitoreo continúa.")
                            st.info(f"**Justificación:** {justificacion}")
                            annotated_frame = results[0].plot()
                            st.image(annotated_frame, caption=f"Frame del incidente menor: '{detected_class_name}'", channels="BGR")
                            st.divider()
                        break 

                    else: # ALARMA_CONFIRMADA o ERROR
                        stop_processing = True # ¡CLAVE! Esto detendrá el bucle principal 'while'.
                        
                        st.error(f"🚨 ¡ALARMA CONFIRMADA! Evento: '{detected_class_name}' en frame {frame_count}. Deteniendo análisis.")
                        annotated_frame = results[0].plot()
                        st.image(annotated_frame, caption=f"Fotograma clave de la alarma: '{detected_class_name}'", channels="BGR", use_column_width=True)

                        # Extraer el videoclip del evento real
                        clip_path = None
                        with st.spinner("💾 Extrayendo videoclip del evento..."):
                            clip_total_frames = int(capture_duration * fps)
                            start_clip_frame = max(0, frame_count - (clip_total_frames // 2))
                            end_clip_frame = min(total_frames, frame_count + (clip_total_frames // 2))
                            
                            clip_filename = f"detection_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                            clip_path = os.path.join(CLIPS_DIR, clip_filename)
                            extract_video_segment(video_path, clip_path, start_clip_frame, end_clip_frame, fps)

                        # Mostrar resultados finales
                        st.subheader("📝 Reporte del Agente 2 (Guardia)")
                        if "error" not in scene_report: st.json(scene_report)
                        else: st.error(f"Fallo en el reporte del Agente 2: {scene_report['error']}")

                        st.subheader("⚖️ Decisión Final del Jefe de Seguridad (Agente 3)")
                        autoridad = decision_result.get("autoridad_a_notificar", "Ninguna")
                        mensaje = decision_result.get("mensaje_de_alerta", "No hay mensaje.")

                        if "ERROR" in veredicto:
                            st.error(f"**Veredicto: {veredicto}**")
                        else: # ALARMA_CONFIRMADA
                            st.error(f"**Veredicto: {veredicto}**")
                            st.info(f"**Justificación:** {justificacion}")
                            st.warning(f"**Acción Inmediata: Notificar a `{autoridad}`**")
                            st.code(mensaje, language="text")
                        
                        if clip_path and os.path.exists(clip_path):
                            st.divider()
                            with open(clip_path, "rb") as file:
                                st.download_button(
                                    label="📥 Descargar Videoclip de la Alarma (.mp4)",
                                    data=file,
                                    file_name=clip_filename,
                                    mime="video/mp4",
                                    use_container_width=True
                                )
                        
                        break # Rompemos el bucle de 'boxes'
            
            if stop_processing:
                break
    
    cap.release()
    if not stop_processing:
        st.success(f"✅ Monitoreo completo del video. Se analizaron {events_found_count} eventos potenciales.")
    
    progress_bar.empty()
    status_text.text("Análisis finalizado.")


def run_search_pipeline(video_path: str, base_class: str, specific_property: str, confidence_threshold: float, frame_processing_rate: int, progress_bar, status_text):
    st.info(f" **Pipeline de Búsqueda Activado:**.")
    st.info(f" **Configuración:** Analizando 1 de cada **{frame_processing_rate}** fotogramas.") # Informar al usuario
    
    class_to_model_map = {
        "person": "general", "car": "general", "bus": "general", "truck": "general",
        "train": "general", "dog": "general", "cat": "general",
        "accident": "accidente", "fire": "fuego"
    }
    model_name = class_to_model_map.get(base_class)
    if not model_name:
        st.error(f"No hay modelo para '{base_class}'."); return
    
    model = detection_models[model_name]
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detected_objects = []

    status_text.text(f"Fase 1/3: Detectando '{base_class}'...")
    class_index = list(model.names.values()).index(base_class)
    
    # Bucle principal para recorrer el video
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        
        progress_bar.progress((frame_idx + 1) / total_frames)

        # --- TODO ESTE BLOQUE AHORA ESTÁ DENTRO DEL 'FOR' ---
        
        # Condición de salto
        if frame_idx % frame_processing_rate != 0:
            continue # Saltar al siguiente fotograma si no toca procesar este

        # El resto del código solo se ejecuta si el frame no se salta
        results = model(frame, classes=[class_index], conf=confidence_threshold, verbose=False)
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Se añade el objeto recortado y el índice del frame
            detected_objects.append({"image": frame[y1:y2, x1:x2], "frame_idx": frame_idx})
    
    # --- FIN DEL BUCLE 'FOR' ---

    cap.release()
    if not detected_objects:
        st.success(f"No se detectó ningún '{base_class}' con la configuración actual."); 
        progress_bar.empty(); status_text.text("Análisis finalizado."); 
        return

    # El resto de la función (Fase 2 y 3) no necesita cambios
    status_text.text(f"Fase 2/3: Seleccionando muestras...")
    num_to_sample = min(len(detected_objects), 2)
    indices_to_sample = np.linspace(0, len(detected_objects) - 1, num_to_sample, dtype=int)
    sampled_objects = [detected_objects[i] for i in indices_to_sample]
    
    status_text.text(f"Fase 3/3: Verificando {len(sampled_objects)} muestras con Gemini...")
    final_matches = []
    cols = st.columns(5)
    col_idx = 0
    for i, item in enumerate(sampled_objects):
        progress_bar.progress((i + 1) / len(sampled_objects))
        pil_image = Image.fromarray(cv2.cvtColor(item["image"], cv2.COLOR_BGR2RGB))
        with st.spinner(f"Analizando muestra {i+1}..."):
            if verify_property_with_gemini(pil_image, base_class, specific_property):
                final_matches.append(item)
                with cols[col_idx % 5]:
                    st.image(item["image"], channels="BGR", caption=f"Frame {item['frame_idx']}")
                col_idx += 1

    progress_bar.empty(); status_text.text("Análisis finalizado.")
    if not final_matches: st.success("Búsqueda completa. No se encontraron coincidencias.")
    else: st.success(f"Búsqueda completa. Se encontraron {len(final_matches)} coincidencias.")





# --- 4. LÓGICA PRINCIPAL DE LA APP ---
st.title("👁️ Sistema de Vigilancia con Agentes de IA")

# --- GENERACIÓN DE LA DESCRIPCIÓN ESTÁTICA ---

# 1. Este diccionario contiene el texto exacto que quieres mostrar.
# Será la única fuente para la lista de capacidades.
categories_to_display = {
    "Eventos Críticos": ['accidentes automovilísticos', 'choques', 'etc'],
    "Personas y Animales": ['personas', 'perros', 'gatos'],
    "Vehículos": ['coches', 'autobuses', 'camiones', 'trenes', 'motos', 'bicicletas']
}

# 2. Construimos el texto del markdown directamente desde el diccionario.
# NO hay ninguna conexión con las clases reales de los modelos.
capabilities_markdown = ""
for category_name, keyword_list in categories_to_display.items():
    # Simplemente unimos la lista de palabras que definiste arriba.
    capabilities_markdown += f"- **{category_name}:** {', '.join(keyword_list)}.\n"

# 3. Mostrar el título y la descripción completa usando el texto que acabamos de generar.
# Este es el texto exacto que proporcionaste en tu mensaje anterior.
st.markdown(f"""
Este es un sistema avanzado que utiliza un modelo de visión artificial (YOLOv5m) con una cadena de tres agentes de Inteligencia Artificial para analizar videos. 
Puedes usarlo de dos formas principales:

- **Modo Monitoreo:** Vigila un video en tiempo real para detectar eventos específicos (ej. informa de la presencia de choques vehiculares).
- **Modo Búsqueda:** Busca objetos que cumplan con una característica particular. (ej. encuentra algun camión de color azul).

**Capacidades del Sistema:**
Nuestros modelos pueden detectar una amplia gama de elementos, incluyendo:
{capabilities_markdown}
**Para comenzar:** Carga un video, describe tu tarea y haz clic en "Procesar Video".
""")

st.divider()

# El resto del código de la interfaz va aquí...


# ...
col1, col2 = st.columns([2, 3])
# ...

col1, col2 = st.columns([2, 3])
with col1:
    st.header("1. Carga tu Video")
    uploaded_file = st.file_uploader("Selecciona un archivo", type=["mp4", "mov", "avi"])

with col2:
    st.header("2. Define la Tarea y Configuración")
    user_intention = st.text_area(
        label="Describe la tarea:\nEj: 'vigila si hay un accidente' o 'revisa si hay algun camioneta roja'",
        value="cámara de seguridad, informa de la presencia de cualquier persona.", # Un valor por defecto más útil
        height=100
    )
    sub_col1, sub_col2, sub_col3 = st.columns(3)
    with sub_col1:
        confidence = st.slider("Confianza Detección", 0.1, 1.0, 0.4, 0.05)
    with sub_col2:
        # Modifiqué el help text para que sea más claro
        frame_skip = st.slider("Analizar 1/N frames", 15, 300, 5, help="Afecta a ambos modos: Monitoreo y Búsqueda")
    with sub_col3:
        capture_duration_option = st.selectbox("Duración Captura", ["5 segundos", "10 segundos"], help="Solo para modo Monitoreo")
        capture_duration = int(capture_duration_option.split(" ")[0])

if st.button(" Procesar Video", use_container_width=True):
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
        
        st.header(" Resultados del Análisis")
        progress_bar = st.progress(0, text="Iniciando...")
        status_text = st.empty()
        
        # --- PASO 3: Llamar al Agente 1 (Despachador) ---
        # Esta llamada se hace UNA SOLA VEZ, al principio.
        dispatch_result = interpret_and_dispatch(user_intention)

        # --- PASO 4: Ejecutar el pipeline correspondiente ---
        if not dispatch_result or "workflow" not in dispatch_result:
            st.error("El Agente Despachador no pudo determinar un flujo de trabajo. Intenta reformular tu solicitud.")
            status_text.text("Error.")
            progress_bar.empty()
        
        elif dispatch_result["workflow"] == "monitor":
            run_monitor_pipeline(
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