import os
import sys
import google.generativeai as genai
import json

# Add the project root directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Inicialización del Modelo Gemini (para texto) ---

def initialize_text_model():
    """
    Inicializa y retorna el modelo de texto de Google Gemini ('gemma-3n-e4b-it').
    """
    if not GEMINI_API_KEY:
        raise ValueError(
            "La clave de API de Gemini no está configurada. "
            "Por favor, establece la variable de entorno GEMINI_API_KEY "
            "o actualiza config.py."
        )
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemma-3n-e4b-it')
        print("Modelo Gemini (gemma-3n-e4b-it) inicializado exitosamente para identificación de objetivo.")
        return model
    except Exception as e:
        print(f"Error al inicializar el modelo Gemini (gemma-3n-e4b-it): {e}")
        raise

# --- Agente de Identificación de Objetivo ---

def interpret_and_dispatch(text_model, user_intention: str) -> tuple[dict, str, str]:
    """
    Agente 1 Despachador: Decide qué pipeline usar y extrae los parámetros.

    Returns:
        tuple: (result_dict, error_message, raw_response)
        - result_dict: Configuración del pipeline o {} si hay error
        - error_message: Mensaje de error o "" si no hay error
        - raw_response: Respuesta cruda del modelo para debugging
    """
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
        raw_response = response.text if response else "No response"
        cleaned_response = raw_response.strip().replace("```json", "").replace("```", "")
        result = json.loads(cleaned_response)
        return result, "", raw_response
    except Exception as e:
        error_message = f"Error del Agente Despachador: {e}"
        raw_response = response.text if 'response' in locals() and response else "No response"
        return {}, error_message, raw_response



# --- Ejemplo de Uso (Solo para pruebas locales rápidas del módulo) ---
if __name__ == "__main__":
    print("--- Probando goal_identification_agent.py ---")
    try:
        # Inicializar el modelo de texto de Gemini
        text_model = initialize_text_model()

        # Escenario 1: Tráfico y accidentes
        config_prompt_1 = "Esta es una cámara de vigilancia de tránsito, tu tarea es identificar accidentes de transito."
        goal_1, error_1, raw_1 = interpret_and_dispatch(text_model, config_prompt_1)
        print(f"\nPrompt: '{config_prompt_1}'")
        print(f"Resultado: {goal_1}")
        if error_1:
            print(f"Error: {error_1}")

        # Escenario 2: Seguridad en tienda (armas)
        config_prompt_2 = "Cámara de seguridad de una tienda. Necesitamos detectar cualquier arma de fuego o cuchillo en manos de personas."
        goal_2, error_2, raw_2 = interpret_and_dispatch(text_model, config_prompt_2)
        print(f"\nPrompt: '{config_prompt_2}'")
        print(f"Resultado: {goal_2}")
        if error_2:
            print(f"Error: {error_2}")

        # Escenario 3: Parque (peleas)
        config_prompt_3 = "Vigilancia de un parque. Alerta ante cualquier señal de pelea o agresión entre personas."
        goal_3, error_3, raw_3 = interpret_and_dispatch(text_model, config_prompt_3)
        print(f"\nPrompt: '{config_prompt_3}'")
        print(f"Resultado: {goal_3}")
        if error_3:
            print(f"Error: {error_3}")

        # Escenario 4: Detección de fuego
        config_prompt_4 = "Cámara en un almacén. Alerta por la presencia de fuego o humo."
        goal_4, error_4, raw_4 = interpret_and_dispatch(text_model, config_prompt_4)
        print(f"\nPrompt: '{config_prompt_4}'")
        print(f"Resultado: {goal_4}")
        if error_4:
            print(f"Error: {error_4}")

        # Escenario 5: Prompt ambiguo
        config_prompt_5 = "Cámara general. Busca cosas raras."
        goal_5, error_5, raw_5 = interpret_and_dispatch(text_model, config_prompt_5)
        print(f"\nPrompt: '{config_prompt_5}'")
        print(f"Resultado: {goal_5}")
        if error_5:
            print(f"Error: {error_5}")

        print("\n--- Pruebas de goal_identification_agent.py finalizadas. ---")

    except ValueError as ve:
        print(f"Error de configuración: {ve}")
    except Exception as e:
        print(f"Ocurrió un error general durante la prueba: {e}")
