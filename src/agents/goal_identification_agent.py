import os
import sys
import google.generativeai as genai

# Add the project root directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Importa las configuraciones desde el archivo config.py
from config import GEMINI_API_KEY, DEBUG_MODE

# --- Inicialización del Modelo Gemini (para texto) ---

def initialize_text_model():
    """
    Inicializa y retorna el modelo de texto de Google Gemini ('gemini-1.5-flash').
    """
    if not GEMINI_API_KEY:
        raise ValueError(
            "La clave de API de Gemini no está configurada. "
            "Por favor, establece la variable de entorno GEMINI_API_KEY "
            "o actualiza config.py."
        )
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        if DEBUG_MODE:
            print("Modelo Gemini (gemini-1.5-flash) inicializado exitosamente para identificación de objetivo.")
        return model
    except Exception as e:
        print(f"Error al inicializar el modelo Gemini (gemini-1.5-flash): {e}")
        raise

# --- Agente de Identificación de Objetivo ---

def identify_classification_goal(text_model, configuration_prompt: str) -> dict:
    """
    Identifica el objetivo principal de clasificación (ej., 'accidentes de tránsito',
    'armas y personas', 'fuego/humo') a partir de un prompt de configuración.

    Args:
        text_model: El modelo GenerativeModel de Gemini ya inicializado.
        configuration_prompt (str): El prompt de texto que describe la tarea de vigilancia.
                                    Ej: "Esta es una cámara de vigilancia de tránsito, tu tarea es identificar accidentes de transito."

    Returns:
        dict: Un diccionario con el objetivo de clasificación identificado y las palabras clave.
              Ej: {"goal": "accidentes de transito", "keywords": ["accidente", "colisión", "choque", "vehículo"]}
              Si no se puede identificar, retorna {"goal": "desconocido", "keywords": []}.
    """
    # Prompt para instruir a Gemini a identificar el objetivo
    prompt_instruction = f"""
    Dado el siguiente texto que describe una tarea de vigilancia para un sistema de alerta:
    "{configuration_prompt}"

    Tu tarea es identificar el **objetivo principal de clasificación** que el sistema debe buscar.
    Luego, lista **palabras clave relevantes** que podrían indicar la presencia de este objetivo.
    Prioriza objetivos como 'accidentes de transito', 'presencia de armas', 'fuego o humo', 'peleas o altercados'.
    
    Responde en formato JSON, con las claves "goal" (el objetivo identificado) y "keywords" (una lista de palabras clave).
    Ejemplo de respuesta:
    ```json
    {{
      "goal": "accidentes de transito",
      "keywords": ["accidente", "colisión", "choque", "vehículo", "atropello"]
    }}
    ```
    Si no puedes identificar un objetivo claro, usa "desconocido" para "goal" y una lista vacía para "keywords".
    """

    try:
        if DEBUG_MODE:
            print(f"Enviando al Agente de Identificación de Objetivo:\n'{prompt_instruction[:200]}...'")

        response = text_model.generate_content(
            prompt_instruction,
            generation_config=genai.GenerationConfig(
                temperature=0.1,  # Baja temperatura para respuestas más directas
                max_output_tokens=150 # Limitar la longitud de la respuesta
            )
        )
        
        # Intentar parsear la respuesta como JSON
        response_text = response.text.strip()
        if DEBUG_MODE:
            print(f"Respuesta cruda de Gemini:\n{response_text}")

        # Extraer el bloque JSON de la respuesta (Gemini a veces envuelve en markdown)
        if response_text.startswith("```json") and response_text.endswith("```"):
            json_str = response_text[7:-3].strip()
        else:
            json_str = response_text
        
        import json
        parsed_response = json.loads(json_str)

        goal = parsed_response.get("goal", "desconocido").lower()
        keywords = parsed_response.get("keywords", [])
        
        if DEBUG_MODE:
            print(f"Objetivo de clasificación identificado: {goal}, Palabras clave: {keywords}")
        
        return {"goal": goal, "keywords": keywords}

    except Exception as e:
        print(f"Error al identificar el objetivo de clasificación con Gemini: {e}")
        return {"goal": "desconocido", "keywords": []}

# --- Ejemplo de Uso (Solo para pruebas locales rápidas del módulo) ---
if __name__ == "__main__":
    print("--- Probando goal_identification_agent.py ---")
    try:
        # Inicializar el modelo de texto de Gemini
        text_model = initialize_text_model()

        # Escenario 1: Tráfico y accidentes
        config_prompt_1 = "Esta es una cámara de vigilancia de tránsito, tu tarea es identificar accidentes de transito."
        goal_1 = identify_classification_goal(text_model, config_prompt_1)
        print(f"\nPrompt: '{config_prompt_1}'")
        print(f"Resultado: {goal_1}")

        # Escenario 2: Seguridad en tienda (armas)
        config_prompt_2 = "Cámara de seguridad de una tienda. Necesitamos detectar cualquier arma de fuego o cuchillo en manos de personas."
        goal_2 = identify_classification_goal(text_model, config_prompt_2)
        print(f"\nPrompt: '{config_prompt_2}'")
        print(f"Resultado: {goal_2}")
        
        # Escenario 3: Parque (peleas)
        config_prompt_3 = "Vigilancia de un parque. Alerta ante cualquier señal de pelea o agresión entre personas."
        goal_3 = identify_classification_goal(text_model, config_prompt_3)
        print(f"\nPrompt: '{config_prompt_3}'")
        print(f"Resultado: {goal_3}")

        # Escenario 4: Detección de fuego
        config_prompt_4 = "Cámara en un almacén. Alerta por la presencia de fuego o humo."
        goal_4 = identify_classification_goal(text_model, config_prompt_4)
        print(f"\nPrompt: '{config_prompt_4}'")
        print(f"Resultado: {goal_4}")

        # Escenario 5: Prompt ambiguo
        config_prompt_5 = "Cámara general. Busca cosas raras."
        goal_5 = identify_classification_goal(text_model, config_prompt_5)
        print(f"\nPrompt: '{config_prompt_5}'")
        print(f"Resultado: {goal_5}")

        print("\n--- Pruebas de goal_identification_agent.py finalizadas. ---")

    except ValueError as ve:
        print(f"Error de configuración: {ve}")
    except Exception as e:
        print(f"Ocurrió un error general durante la prueba: {e}")