import os
import google.generativeai as genai
from PIL import Image # Necesario para abrir y pasar imágenes
import json
import base64 # Para codificar imágenes a base64 si es necesario pasarlas como texto en el prompt

# Importa las configuraciones del proyecto
from config import GEMINI_API_KEY, DEBUG_MODE

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
        if DEBUG_MODE:
            print("Modelo Gemini (gemini-pro-vision) inicializado exitosamente para análisis temporal.")
        return model
    except Exception as e:
        print(f"Error al inicializar el modelo Gemini (gemini-pro-vision) para análisis temporal: {e}")
        raise

# --- Agente de Análisis de Contexto Temporal ---

def analyze_temporal_context(
    temporal_model, 
    image_paths: list[str], 
    initial_detection_context: dict, 
    classification_goal_from_agent1: dict
) -> dict:
    """
    Analiza una secuencia de imágenes para verificar la persistencia y evolución
    de un evento de emergencia previamente identificado.

    Args:
        temporal_model: El modelo GenerativeModel de Gemini ('gemini-pro-vision').
        image_paths (list[str]): Lista de rutas a los fotogramas (imagen central, 5s antes, 5s después).
        initial_detection_context (dict): El JSON de resultado del Agente 2 (gemini_multimodal_analyzer.py).
                                          Incluye description, classification, summary, service, detections.
        classification_goal_from_agent1 (dict): El JSON de resultado del Agente 1 (goal_identification_agent.py).
                                                Incluye goal y keywords.

    Returns:
        dict: Un diccionario indicando si la emergencia se confirma, un mensaje actualizado,
              y los detalles para la alerta final.
              Ej: {
                    "is_confirmed_emergency": True,
                    "final_summary": "Accidente de tráfico confirmado con evolución de colisión a vehículo volcado.",
                    "final_service": "Policía y Ambulancia",
                    "reasoning": "El análisis de la secuencia de video muestra el desarrollo de una colisión, con vehículos impactando y uno volcándose, confirmando la emergencia."
                  }
              Retorna valores por defecto si no se confirma o hay un error.
    """
    if not image_paths:
        print("Advertencia: No se proporcionaron rutas de imagen para el análisis temporal.")
        return {
            "is_confirmed_emergency": False,
            "final_summary": "No se pudo verificar la emergencia por falta de fotogramas.",
            "final_service": "N/A",
            "reasoning": "No hay imágenes para analizar."
        }
    
    if not initial_detection_context or initial_detection_context.get("classification") != "Emergencia":
        print("Advertencia: El contexto inicial no indica una emergencia. No se necesita verificación temporal.")
        return {
            "is_confirmed_emergency": False,
            "final_summary": "El evento inicial no fue clasificado como emergencia.",
            "final_service": "N/A",
            "reasoning": "El Agente 2 no clasificó esto como emergencia."
        }

    try:
        # Prepara el contenido para Gemini: lista de partes de texto y objetos de imagen
        contents = []
        
        # Añade el contexto de los agentes 1 y 2 al inicio del prompt
        initial_description = initial_detection_context.get("description", "Sin descripción inicial.")
        initial_summary = initial_detection_context.get("summary", "Sin resumen inicial.")
        initial_service = initial_detection_context.get("service", "N/A.")
        initial_detections = initial_detection_context.get("detections", [])
        
        goal = classification_goal_from_agent1.get("goal", "evento desconocido")
        keywords_str = ", ".join(classification_goal_from_agent1.get("keywords", []))

        prompt_text = f"""
        **Análisis de Confirmación de Emergencia (Secuencia Temporal)**

        Has recibido una secuencia de imágenes de video. Una detección inicial (Agente 2) clasificó un evento como **"{initial_detection_context['classification']}"** con el objetivo principal de **"{goal}"**.

        **Contexto de la Detección Inicial (Agente 2):**
        - Descripción de la imagen central: "{initial_description}"
        - Resumen inicial: "{initial_summary}"
        - Servicio sugerido inicial: "{initial_service}"
        - Palabras clave relevantes para el objetivo: "{keywords_str}"
        
        **Tu tarea es analizar la secuencia de imágenes para:**
        1.  **Verificar la persistencia y la evolución** del evento inicialmente detectado. ¿El evento se mantiene, escala, o se resuelve?
        2.  **Confirmar si es una emergencia real** basándote en la evidencia temporal.
        3.  Si se confirma, **proporcionar un resumen final del evento** considerando toda la secuencia.
        4.  Si se confirma, **sugerir el servicio de emergencia definitivo**, si hay algún cambio.

        **Instrucciones para el Análisis:**
        - Cada imagen está numerada (Imagen 1, Imagen 2, etc.). La imagen central es la que disparó la detección original.
        - Describe brevemente los cambios o la evolución observada en cada imagen relevante.
        - Presta atención a si los objetos/situaciones clave del "{goal}" persisten o empeoran.

        Responde en formato JSON. Si la emergencia se confirma, "is_confirmed_emergency" debe ser `true`. De lo contrario, `false`.

        ```json
        {{
          "is_confirmed_emergency": true | false,
          "final_summary": "[Resumen final del evento si confirmado, sino motivo de no confirmación]",
          "final_service": "[Servicio de emergencia definitivo si confirmado, sino N/A]",
          "reasoning": "[Explicación concisa de por qué se confirmó o no la emergencia, basándose en la secuencia]"
        }}
        ```
        """
        contents.append(prompt_text)

        # Añadir las imágenes a la lista de contenidos
        for i, path in enumerate(image_paths):
            if not os.path.exists(path):
                print(f"Advertencia: Imagen {i+1} no encontrada en la secuencia: {path}")
                continue # Saltar esta imagen si no existe
            
            img = Image.open(path).convert("RGB") # Asegurarse de que sea RGB
            contents.append(img)
            # Opcional: añadir una pequeña etiqueta de texto para cada imagen en el prompt
            # contents.append(f"\nImagen {i+1}:\n")

        if DEBUG_MODE:
            print(f"Enviando {len(image_paths)} imágenes para análisis temporal con el prompt que comienza con:\n'{prompt_text[:300]}...'")

        # Realizar la inferencia con el modelo multimodal
        response = temporal_model.generate_content(
            contents,
            generation_config=genai.GenerationConfig(
                temperature=0.1,  # Baja temperatura para respuestas lógicas y de confirmación
                max_output_tokens=400 # Suficientes tokens para el razonamiento
            )
        )
        
        response_text = response.text.strip()
        if DEBUG_MODE:
            print(f"Respuesta cruda de Gemini (Agente Temporal):\n{response_text}")

        # Extraer y parsear el bloque JSON
        if response_text.startswith("```json") and response_text.endswith("```"):
            json_str = response_text[7:-3].strip()
        else:
            json_str = response_text
        
        parsed_response = json.loads(json_str)

        result = {
            "is_confirmed_emergency": parsed_response.get("is_confirmed_emergency", False),
            "final_summary": parsed_response.get("final_summary", "No se pudo confirmar la emergencia.").strip(),
            "final_service": parsed_response.get("final_service", "N/A").strip(),
            "reasoning": parsed_response.get("reasoning", "Sin razonamiento proporcionado.").strip()
        }
        
        if DEBUG_MODE:
            print(f"\n--- Análisis Temporal Completo ---")
            print(f"Emergencia Confirmada: {result['is_confirmed_emergency']}")
            print(f"Resumen Final: {result['final_summary']}")
            print(f"Servicio Final: {result['final_service']}")
            print(f"Razonamiento: {result['reasoning']}")
            print("----------------------------------")

        return result

    except json.JSONDecodeError as jde:
        print(f"Error al parsear la respuesta JSON de Gemini para análisis temporal: {jde}")
        print(f"Respuesta cruda que causó el error: \n{response_text[:500]}...")
        return {
            "is_confirmed_emergency": False,
            "final_summary": "Error al interpretar la respuesta del AI.",
            "final_service": "N/A",
            "reasoning": "Error de formato de respuesta."
        }
    except Exception as e:
        print(f"Error inesperado durante el análisis temporal: {e}")
        return {
            "is_confirmed_emergency": False,
            "final_summary": "Error interno del sistema durante el análisis temporal.",
            "final_service": "N/A",
            "reasoning": str(e)
        }

# --- Ejemplo de Uso (Solo para pruebas locales rápidas del módulo) ---
if __name__ == "__main__":
    print("--- Probando temporal_context_analyzer.py ---")
    
    # Importar agentes anteriores para simular su salida
    from src.agents.goal_identification_agent import initialize_text_model, identify_classification_goal
    from src.agents.image_analyzer import initialize_vision_model, analyze_image_for_goal
    from config import TEST_IMAGES_DIR # Para las imágenes de prueba

    try:
        # 1. Inicializar modelos para los agentes
        gemini_text_model_for_goal = initialize_text_model()
        gemini_vision_model_for_analyzer = initialize_vision_model()
        gemini_temporal_model = initialize_video_model()

        # 2. Simular salida del Agente 1 (Identificación de Objetivo)
        print("\n--- SIMULANDO AGENTE 1 ---")
        config_prompt = "Cámara de vigilancia de tráfico. Identificar accidentes automovilísticos y colisiones."
        goal_output = identify_classification_goal(gemini_text_model_for_goal, config_prompt)
        print(f"Objetivo identificado: {goal_output}")

        # 3. Simular salida del Agente 2 (Detección Inicial en una imagen)
        print("\n--- SIMULANDO AGENTE 2 (Detección Inicial) ---")
        # Asegúrate de tener imágenes que simulen una secuencia (ej. frame_001.jpg, frame_002.jpg, ...)
        # Una imagen donde Agente 2 detectaría una *posible* emergencia.
        
        # Para la prueba, simularemos una secuencia de frames que muestren un accidente
        # Asegúrate de tener estas imágenes en tu data/test_images/
        # Por ejemplo: traffic_before_accident.jpg, traffic_accident_impact.jpg, traffic_accident_aftermath.jpg
        # Y una imagen central para simular la detección inicial
        central_image_path = os.path.join(TEST_IMAGES_DIR, "traffic_accident_impact.jpg") 
        if not os.path.exists(central_image_path):
            print(f"ERROR: Imagen central '{central_image_path}' no encontrada. Crea una para la prueba.")
            exit() # Salir si no hay imagen central

        # Simular el resultado del Agente 2 para la imagen central
        initial_detection_result = analyze_image_for_goal(gemini_vision_model_for_analyzer, central_image_path, goal_output)
        print(f"Detección Inicial del Agente 2 (clasificación): {initial_detection_result['classification']}")
        
        if initial_detection_result.get("classification") != "Emergencia":
            print("El Agente 2 no detectó una emergencia inicial. No se procede con el análisis temporal.")
            exit()

        # 4. Preparar secuencia de imágenes para el Agente 3
        # Aquí, imagina que tienes frames numerados de tu "video"
        # 5 segundos antes y 5 segundos después.
        # Para el hackathon, puedes usar un conjunto pequeño de imágenes estáticas.
        # EJEMPLO: asume que tienes frames:
        # traffic_frame_01.jpg (5s antes)
        # traffic_frame_02.jpg (2.5s antes)
        # traffic_accident_impact.jpg (IMAGEN CENTRAL, donde se detectó)
        # traffic_frame_04.jpg (2.5s después)
        # traffic_frame_05.jpg (5s después)
        
        # Ajusta estas rutas a tus imágenes reales de prueba
        sequence_image_paths = [
            os.path.join(TEST_IMAGES_DIR, "traffic_before_accident.jpg"),
            os.path.join(TEST_IMAGES_DIR, "traffic_accident_impact.jpg"), # Imagen central
            os.path.join(TEST_IMAGES_DIR, "traffic_accident_aftermath.jpg"),
        ]
        
        # Asegurarse de que todas las imágenes de la secuencia existan
        for p in sequence_image_paths:
            if not os.path.exists(p):
                print(f"ERROR: Imagen de secuencia '{p}' no encontrada. Asegúrate de tener los frames de prueba.")
                exit()
        
        print(f"\n--- Agente 3: Análisis Temporal con {len(sequence_image_paths)} fotogramas ---")
        temporal_analysis_result = analyze_temporal_context(
            gemini_temporal_model,
            sequence_image_paths,
            initial_detection_result,
            goal_output
        )

        print("\n--- Resultado Final del Agente Temporal ---")
        print(f"Emergencia Confirmada: {temporal_analysis_result['is_confirmed_emergency']}")
        print(f"Resumen Final: {temporal_analysis_result['final_summary']}")
        print(f"Servicio Sugerido: {temporal_analysis_result['final_service']}")
        print(f"Razonamiento: {temporal_analysis_result['reasoning']}")

        print("\n--- Pruebas de temporal_context_analyzer.py finalizadas. ---")

    except ValueError as ve:
        print(f"Error de configuración: {ve}. Asegúrate de que GEMINI_API_KEY esté establecida.")
    except Exception as e:
        print(f"Ocurrió un error general durante la prueba: {e}")