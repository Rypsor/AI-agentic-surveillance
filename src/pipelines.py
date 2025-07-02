import cv2
import json
import numpy as np
from PIL import Image

from src.agents.video_analyzer import analyze_scene
from src.agents.search import verify_property
from src.agents.verify import verify_alarm



def run_monitor_pipeline(st, agents: dict , detection_models: dict, video_path: str, user_intention: str, surveillance_config: dict, confidence_threshold: float, frame_processing_rate: int, capture_duration: int, progress_bar, status_text):
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
                if detected_class_name in target_classes:
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
                        scene_report = analyze_scene(agents["vision_model"], collected_frames, user_intention, target_classes)
                    st.subheader("📝 Reporte del Agente 2")
                    if "error" not in scene_report: st.json(scene_report)
                    else: st.error(f"Fallo: {scene_report['error']}")

                    with st.spinner('🧐 Agente 3 (Supervisor) emitiendo veredicto...'):
                        verification_result = verify_alarm(agents["text_model"] ,scene_report, user_intention)
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

def run_search_pipeline(st, agents: dict, detection_models: dict, video_path: str, base_class: str, specific_property: str, confidence_threshold: float, frame_processing_rate: int, progress_bar, status_text):
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
            if verify_property(agents["vision_model"], pil_image, base_class, specific_property):
                final_matches.append(item)
                with cols[col_idx % 5]:
                    st.image(item["image"], channels="BGR", caption=f"Frame {item['frame_idx']}")
                col_idx += 1

    progress_bar.empty(); status_text.text("Análisis finalizado.")
    if not final_matches: st.success("Búsqueda completa. No se encontraron coincidencias.")
    else: st.success(f"Búsqueda completa. Se encontraron {len(final_matches)} coincidencias.")
