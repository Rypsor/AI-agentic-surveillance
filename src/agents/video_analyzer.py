import os
import sys
import google.generativeai as genai
from PIL import Image
import json
import cv2

# Add the project root directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Inicialización del Modelo Gemini (multimodal) ---

def initialize_video_model():
    """
    Inicializa y retorna el modelo de visión de Google Gemini ('gemini-pro-vision').
    """
    if not GEMINI_API_KEY:
        raise ValueError(
            "La clave de API de Gemini no está configurada. "
            "Por favor, establece la variable de entorno GEMINI_API_KEY "
            "o actualiza config.py."
        )
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-pro-vision')
        print("Modelo Gemini (gemini-pro-vision) inicializado exitosamente para análisis temporal.")
        return model
    except Exception as e:
        print(f"Error al inicializar el modelo Gemini (gemini-pro-vision) para análisis temporal: {e}")
        raise

# --- Agente de Análisis de Contexto Temporal ---

def analyze_scene(vision_model, frames: list, user_intention: str, target_classes: list) -> dict:
    """Agente 2: Analiza una secuencia de fotogramas y genera un reporte JSON."""
    if not frames: return {"error": "No se capturaron fotogramas."}
    pil_images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    prompt = f"Intención: '{user_intention}', Objetivos: {target_classes}. Analiza los fotogramas y responde con JSON: {{\"resumen_escena\": ..., \"eventos_clave\": [...], \"evaluacion_inicial\": ...}}"
    try:
        response = vision_model.generate_content([prompt] + pil_images)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response)
    except Exception as e: return {"error": f"Fallo en reporte: {e}"}