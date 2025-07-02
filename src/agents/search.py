
from PIL import Image

def verify_property(vision_model, image: Image, base_class: str, specific_property: str) -> bool:
    """Agente de Búsqueda: Verifica si una imagen cumple una propiedad."""
    prompt = f"¿Es un/a '{base_class}' que es/tiene '{specific_property}'? Responde SÍ o NO."
    try:
        response = vision_model.generate_content([prompt, image])
        return "SI" in response.text.upper()
    except Exception: return False
