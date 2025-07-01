# config.py
import os
from dotenv import load_dotenv

# Carga las variables de entorno desde un archivo .env
# Esto es más seguro que poner la clave directamente en el código
# Especifica el path explícito del archivo .env
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

# Tu clave de API de Google Gemini
# Crea un archivo llamado .env en la misma carpeta y añade:
# GEMINI_API_KEY="TU_CLAVE_AQUI"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "AIzaSyBSFPlb-oKIz9_ohauZ5gy5iM2QRpXQto0"

# Configuración de debug
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

TEST_IMAGES_DIR = "data/images"

# Mapeo de los modelos de detección a sus pesos y clases de interés
MODEL_CONFIG = {
    'accident': {
        'path': 'weights/best_accident.pt',
        'classes': ['accident'] # Asegúrate que esta es la clase que tu modelo detecta
    },
    'fire': {
        'path': 'weights/best_fire.pt',
        'classes': ['fire', 'smoke'] # Clases de interés para este modelo
    },
    'general': {
        'path': 'weights/best_general.pt',
        'classes': ['person'] # Clase de interés
    }
}

# Palabras clave para mapear la intención del usuario a un modelo
INTENTION_KEYWORDS = {
    'accident': ['accidente', 'choque', 'colision', 'autopista', 'carretera'],
    'fire': ['fuego', 'humo', 'incendio', 'quemando', 'deposito'],
    'general': ['persona', 'gente', 'intruso', 'ladrón', 'casa', 'afuera']
}