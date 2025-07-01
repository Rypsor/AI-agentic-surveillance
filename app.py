# app.py (Versión 5.0 - Doble Pipeline: Monitoreo y Búsqueda)

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
OUTPUT_DIR = os.path.join(BASE_DIR, "output_captures")
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
        vision_model = genai.GenerativeModel('gemini-2.0-flash')
        text_model = genai.GenerativeModel('gemini-2.0-flash')
        return models, vision_model, text_model

detection_models, vision_model, text_model = load_models()


# --- 2. DEFINICIÓN DE AGENTES DE IA ---

def interpret_and_dispatch(user_intention: str) -> dict:
    """Agente 1 Despachador: Decide qué pipeline usar y extrae los parámetros."""
    knowledge_base = """
    Módulos de Vigilancia: Modelo "accidente" (clase: accident), Modelo "fuego" (clase: fire), Modelo "general" (clases: person, car, bus, truck, train, dog, cat).
    Conceptos: "mascotas" -> ["dog", "cat"], "vehículos" -> ["car", "bus", "truck", "train"], "personas" -> "person", "incendio" -> "fire", "carros" -> "car".
    """
    prompt = f"""
    Eres un despachador de IA con dos pipelines: "monitor" (vigilancia en tiempo real) y "search" (encontrar objetos con características).
    Analiza la solicitud y responde ÚNICAMENTE con un JSON indicando el pipeline y su config.

    ### Formato de Salida ###
    - Para "monitor": {{"workflow": "monitor", "config": {{"modelo": ["clase"]}}}}
    - Para "search": {{"workflow": "search", "config": {{"base_class": "...", "specific_property": "..."}}}}

    ### Ejemplos ###
    - Solicitud: "vigila si hay mascotas o un accidente" -> {{"workflow": "monitor", "config": {{"general": ["dog", "cat"], "accidente": ["accident"]}}}}
    - Solicitud: "encuentra todos los camiones de color azul" -> {{"workflow": "search", "config": {{"base_class": "truck", "specific_property": "de color azul"}}}}
    
    ### Conocimiento ###
    {knowledge_base}
    ---
    Solicitud del Usuario: "{user_intention}"
    ---
    JSON de Respuesta:
    """
    try:
        response = text_model.generate_content(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response)
    except Exception as e:
        st.error(f"Error del Agente Despachador: {e}")
        st.text_area("Respuesta del modelo:", response.text if 'response' in locals() else "No response", height=150)
        return {}

def describe_scene_with_gemini(frames: list, user_intention: str, target_classes: list) -> dict:
    """Agente 2: Analiza una secuencia de fotogramas y genera un reporte JSON."""
    if not frames: return {"error": "No se capturaron fotogramas."}
    pil_images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    prompt = f"Intención: '{user_intention}', Objetivos: {target_classes}. Analiza los fotogramas y responde con JSON: {{\"resumen_escena\": ..., \"eventos_clave\": [...], \"evaluacion_inicial\": ...}}"
    try:
        response = vision_model.generate_content([prompt] + pil_images)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response)
    except Exception as e: return {"error": f"Fallo en reporte: {e}"}

def verify_alarm_with_gemini(scene_report: dict, user_intention: str) -> dict:
    """Agente 3: Recibe el reporte y emite un veredicto final."""
    if "error" in scene_report: return {"veredicto": "ERROR", "justificacion": scene_report["error"]}
    report_str = json.dumps(scene_report, indent=2, ensure_ascii=False)
    prompt = f"Intención: '{user_intention}'. Reporte: {report_str}. ¿Alarma real? Responde con JSON: {{\"veredicto\": ..., \"justificacion\": ...}}"
    try:
        response = text_model.generate_content(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response)
    except Exception as e: return {"veredicto": "ERROR", "justificacion": str(e)}

def verify_property_with_gemini(image: Image, base_class: str, specific_property: str) -> bool:
    """Agente de Búsqueda: Verifica si una imagen cumple una propiedad."""
    prompt = f"¿Es un/a '{base_class}' que es/tiene '{specific_property}'? Responde SÍ o NO."
    try:
        response = vision_model.generate_content([prompt, image])
        return "SI" in response.text.upper()
    except Exception: return False


# --- 3. DEFINICIÓN DE PIPELINES ---

def run_monitor_pipeline(video_path: str, user_intention: str, surveillance_config: dict, confidence_threshold: float, frame_processing_rate: int, capture_duration: int, progress_bar, status_text):
    st.info(f"🧠 **Pipeline de Monitoreo Activado:** Configuración: `{json.dumps(surveillance_config)}`")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_to_skip_for_capture = int(fps * 0.5) if fps > 2 else 1
    num_frames_to_capture = int(capture_duration / 0.5)
    
    frame_count = 0
    event_triggered = False

    while cap.isOpened() and not event_triggered:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        if frame_count % frame_processing_rate != 0:
            if frame_count % 10 == 0: progress_bar.progress(frame_count / total_frames, text=f"Monitoreando... ({frame_count}/{total_frames})")
            continue
        
        progress_bar.progress(frame_count / total_frames, text=f"Monitoreando... ({frame_count}/{total_frames})")

        for model_name, target_classes in surveillance_config.items():
            if model_name not in detection_models: continue
            model = detection_models[model_name]
            results = model(frame, conf=confidence_threshold, verbose=False)
            
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                detected_class_name = model.names[class_id]
                if detected_class_name.lower() in target_classes:
                    event_triggered = True
                    st.warning(f"🚨 ¡DETECCIÓN! '{detected_class_name}' en frame {frame_count}.")
                    annotated_frame = results[0].plot()
                    st.image(annotated_frame, caption=f"Detección de '{detected_class_name}'", channels="BGR", use_column_width=True)
                    
                    status_text.text(f"Capturando secuencia de {num_frames_to_capture} frames...")
                    collected_frames = [frame.copy()]
                    for _ in range(num_frames_to_capture - 1):
                        for _ in range(frames_to_skip_for_capture): cap.read()
                        ret_capture, next_frame = cap.read()
                        if not ret_capture: break
                        collected_frames.append(next_frame.copy())
                    
                    with st.spinner('🤖 Agente 2 (Analista) preparando reporte...'):
                        scene_report = describe_scene_with_gemini(collected_frames, user_intention, target_classes)
                    st.subheader("📝 Reporte del Agente 2")
                    if "error" not in scene_report: st.json(scene_report)
                    else: st.error(f"Fallo: {scene_report['error']}")

                    with st.spinner('🧐 Agente 3 (Supervisor) emitiendo veredicto...'):
                        verification_result = verify_alarm_with_gemini(scene_report, user_intention)
                    st.subheader("⚖️ Veredicto Final del Agente 3")
                    if "error" not in verification_result:
                        if verification_result.get("veredicto") == "ALARMA_REAL": st.error(f"**Veredicto: {verification_result.get('veredicto')}**")
                        else: st.success(f"**Veredicto: {verification_result.get('veredicto')}**")
                        st.info(f"**Justificación:** {verification_result.get('justificacion')}")
                    else: st.error(f"Fallo: {verification_result.get('justificacion')}")
                    break
            if event_triggered: break
    
    cap.release()
    if not event_triggered: st.success("✅ Monitoreo completo. No se detectaron eventos de interés.")
    progress_bar.empty(); status_text.text("Análisis finalizado.")

# --- Función run_search_pipeline CORREGIDA ---

def run_search_pipeline(video_path: str, base_class: str, specific_property: str, confidence_threshold: float, frame_processing_rate: int, progress_bar, status_text):
    st.info(f"🧠 **Pipeline de Búsqueda Activado:** Buscando '{base_class}' que sean/tengan '{specific_property}'.")
    st.info(f"⚙️ **Configuración:** Analizando 1 de cada **{frame_processing_rate}** fotogramas.") # Informar al usuario
    
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