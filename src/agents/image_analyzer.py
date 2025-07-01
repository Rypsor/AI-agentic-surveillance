import os
import sys
import google.generativeai as genai
from PIL import Image
import io # Usado si procesas imágenes desde bytes, aunque Image.open(path) es más simple aquí.
import json # Para parsear la respuesta JSON de Gemini.

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
# Importa las configuraciones del proyecto
from config import GEMINI_API_KEY, DEBUG_MODE, TEST_IMAGES_DIR

# --- Inicialización del Modelo Gemini (para visión) ---

def initialize_vision_model():
    """
    Inicializa y retorna el modelo de visión de Google Gemini ('gemini-1.5-flash').
    
    Raises:
        ValueError: Si la clave de API de Gemini no está configurada.
        Exception: Para otros errores durante la inicialización del modelo.
    """
    if not GEMINI_API_KEY or GEMINI_API_KEY == "TU_CLAVE_DE_API_AQUI_SI_NO_USAS_ENV":
        raise ValueError(
            "La clave de API de Gemini no está configurada. "
            "Por favor, establece la variable de entorno GEMINI_API_KEY "
            "o actualiza config.py."
        )
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        if DEBUG_MODE:
            print("Modelo Gemini (gemini-1.5-flash) inicializado exitosamente para análisis de imágenes.")
        return model
    except Exception as e:
        print(f"Error al inicializar el modelo Gemini (gemini-1.5-flash): {e}")
        raise

# --- Agente de Análisis Multimodal ---

def analyze_image_for_goal(vision_model, image_path: str, classification_goal: dict) -> dict:
    """
    Analiza una imagen utilizando Gemini Vision Pro, enfocándose en el objetivo
    de clasificación proporcionado por el agente anterior.

    Args:
        vision_model: El modelo GenerativeModel de Gemini ya inicializado ('gemini-pro-vision').
        image_path (str): Ruta a la imagen a analizar.
        classification_goal (dict): El JSON de resultado del agente de identificación de objetivo,
                                    ej: {"goal": "accidentes de transito", "keywords": ["accidente", "colisión", "choque", "vehículo"]}

    Returns:
        dict: Un diccionario con la descripción de la escena, la clasificación (Normal/Emergencia),
              el resumen del evento (si aplica) y el servicio de emergencia sugerido.
              Retorna un diccionario con valores por defecto si hay un error o no se detecta nada.
              Formato esperado:
              {
                "description": "...",
                "classification": "Normal" | "Emergencia" | "Error",
                "summary": "...",
                "service": "Policía" | "Bomberos" | "Ambulancia" | "Policía y Ambulancia" | "N/A"
              }
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"La imagen no se encontró en: {image_path}")

        img = Image.open(image_path)

        # Extraer el objetivo y las palabras clave del diccionario de configuración
        goal = classification_goal.get("goal", "evento general")
        keywords_list = classification_goal.get("keywords", [])
        keywords_str = ", ".join(keywords_list) if keywords_list else "ninguna palabra clave específica"
        
        # Construir el prompt dinámico para Gemini
        prompt_instruction = f"""
        Analiza cuidadosamente esta imagen de vigilancia.
        Tu tarea principal es identificar cualquier indicio de: **{goal}**.
        Presta especial atención a la presencia de: personas, vehículos, objetos peligrosos (como armas blancas o de fuego), fuego, humo, o signos de conflicto/accidente.

        Si detectas el objetivo principal de '{goal}', o cualquier otra situación que parezca una emergencia, sigue estos pasos:
        1.  **Describe la escena** de forma concisa y objetiva.
        2.  **Clasifica la situación** como 'Normal' o 'Emergencia'.
        3.  Si la clasificación es 'Emergencia', **resume el evento** que observas.
        4.  Si la clasificación es 'Emergencia', **sugiere el servicio de emergencia más adecuado** (Policía, Bomberos, Ambulancia, o una combinación como 'Policía y Ambulancia').

        Las palabras clave específicas que indican '{goal}' son: {keywords_str}.

        Responde usando **únicamente** el siguiente formato JSON. Si una sección no aplica o no se puede determinar, deja el valor en blanco o usa 'N/A' según corresponda para ese campo. NO INCLUYAS TEXTO ADICIONAL FUERA DEL JSON.
        ```json
        {{
          "description": "",
          "classification": "",
          "summary": "",
          "service": ""
        }}
        ```
        """
        
        # Enviar la imagen y el prompt a Gemini
        response = vision_model.generate_content(
            [prompt_instruction, img],
            generation_config=genai.GenerationConfig(
                temperature=0.01, # Muy baja temperatura para respuestas deterministas y estructuradas
                max_output_tokens=300 # Limitar la longitud de la respuesta JSON
            )
        )
        
        response_text = response.text.strip()
        
        # Extraer el bloque JSON de la respuesta
        # Gemini a menudo envuelve JSON en bloques de Markdown.
        if response_text.startswith("```json") and response_text.endswith("```"):
            json_str = response_text[7:-3].strip()
        else:
            json_str = response_text
        
        # Intentar parsear el JSON
        parsed_response = json.loads(json_str)

        # Extraer los datos del JSON, proporcionando valores por defecto y limpieza
        result = {
            "description": parsed_response.get("description", "No se pudo describir la escena.").strip(),
            "classification": parsed_response.get("classification", "Normal").strip(),
            "summary": parsed_response.get("summary", "N/A").strip(),
            "service": parsed_response.get("service", "N/A").strip()
        }
        
        if DEBUG_MODE:
            print(f"\n--- Análisis de Imagen: {os.path.basename(image_path)} ---")
            print(f"Objetivo: '{goal}' con palabras clave: '{keywords_str}'")
            print(f"Descripción: {result['description']}")
            print(f"Clasificación: {result['classification']}")
            print(f"Resumen: {result['summary']}")
            print(f"Servicio: {result['service']}")
            print("-----------------------------------")

        return result

    except FileNotFoundError as fnfe:
        print(f"Error: {fnfe}")
        return {
            "description": "Imagen no encontrada.",
            "classification": "Error",
            "summary": "Error de archivo.",
            "service": "N/A"
        }
    except json.JSONDecodeError as jde:
        print(f"Error al parsear la respuesta JSON de Gemini: {jde}")
        print(f"Respuesta cruda que causó el error: \n{response_text[:500]}...") # Imprime parte de la respuesta problemática
        return {
            "description": "Error al interpretar la respuesta del AI.",
            "classification": "Error",
            "summary": "Formato de respuesta incorrecto.",
            "service": "N/A"
        }
    except Exception as e:
        print(f"Error inesperado durante el análisis de imagen con Gemini Multimodal: {e}")
        return {
            "description": "Error interno del sistema.",
            "classification": "Error",
            "summary": "Error desconocido.",
            "service": "N/A"
        }

# --- Ejemplo de Uso (Solo para pruebas locales rápidas del módulo) ---
if __name__ == "__main__":
    print("--- Probando gemini_multimodal_analyzer.py ---")
    
    # Importar el agente de identificación de objetivo para simular su salida
    from src.agents.goal_identification_agent import initialize_text_model, identify_classification_goal
    
    try:
        # 1. Inicializar modelos
        # Modelo de visión para este agente
        vision_model = initialize_vision_model()
        # Modelo de texto para el agente de identificación de objetivo (solo para simular su salida aquí)
        text_model_for_goal = initialize_text_model()

        # 2. Simular la salida del Agente 1 (Identificación de Objetivo)
        print("\n--- SIMULANDO AGENTE 1: Identificación de Objetivo ---")
        config_prompt_accident = "Esta es una cámara de vigilancia de tráfico. Tu tarea es identificar accidentes de tráfico graves, colisiones o vehículos volcados."
        goal_output_accident = identify_classification_goal(text_model_for_goal, config_prompt_accident)
        print(f"Objetivo Identificado para accidentes: {goal_output_accident}")

        config_prompt_weapons = "Cámara de seguridad en una tienda. Detectar la presencia de armas de fuego o cuchillos en manos de personas."
        goal_output_weapons = identify_classification_goal(text_model_for_goal, config_prompt_weapons)
        print(f"Objetivo Identificado para armas: {goal_output_weapons}")
        
        config_prompt_fire = "Cámara en un almacén. Alerta por la presencia de fuego o humo denso."
        goal_output_fire = identify_classification_goal(text_model_for_goal, config_prompt_fire)
        print(f"Objetivo Identificado para fuego: {goal_output_fire}")

        # 3. Preparar imágenes de prueba (Asegúrate de que existan en data/test_images/)
        # Nombres de ejemplo, ajusta según tus archivos:
        image_accident = os.path.join(TEST_IMAGES_DIR, "accidente1.jpg")
        image_weapon = os.path.join(TEST_IMAGES_DIR, "cuchillo1.jpg")
        image_fire = os.path.join(TEST_IMAGES_DIR, "fuego2.jpg")
        image_normal = os.path.join(TEST_IMAGES_DIR, "telefonos1.jpg")

        test_images_and_goals = [
            (image_accident, goal_output_accident, "Escenario de Accidente de Tráfico"),
            (image_weapon, goal_output_weapons, "Escenario de Persona con Arma"),
            (image_fire, goal_output_fire, "Escenario de Fuego/Humo"),
            (image_normal, {"goal": "actividad normal", "keywords": ["persona", "vehículo"]}, "Escenario Normal"),
        ]

        # 4. Probar Agente 2 con diferentes escenarios
        for img_path, goal_data, scenario_name in test_images_and_goals:
            if os.path.exists(img_path):
                print(f"\n--- Probando Agente 2 con: {scenario_name} ({os.path.basename(img_path)}) ---")
                analysis_result = analyze_image_for_goal(vision_model, img_path, goal_data)
                print(f"  Resultado Final (Clasificación): {analysis_result['classification']}")
                print(f"  Resultado Final (Servicio Sugerido): {analysis_result['service']}")
            else:
                print(f"\nSaltando prueba para '{scenario_name}': Imagen no encontrada en {img_path}")

        print("\n--- Pruebas de gemini_multimodal_analyzer.py finalizadas. ---")

    except ValueError as ve:
        print(f"Error de configuración: {ve}. Asegúrate de que GEMINI_API_KEY esté establecida.")
    except Exception as e:
        print(f"Ocurrió un error general durante la prueba: {e}")