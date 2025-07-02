import google.generativeai as genai
import json



def verify_alarm(text_model, scene_report: dict, user_intention: str) -> dict:
    """Agente 3: Recibe el reporte y emite un veredicto final."""
    if "error" in scene_report: return {"veredicto": "ERROR", "justificacion": scene_report["error"]}
    report_str = json.dumps(scene_report, indent=2, ensure_ascii=False)
    prompt = f"Intención: '{user_intention}'. Reporte: {report_str}. ¿Alarma real? Responde con JSON: {{\"veredicto\": ..., \"justificacion\": ...}}"
    try:
        response = text_model.generate_content(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response)
    except Exception as e: return {"veredicto": "ERROR", "justificacion": str(e)}
