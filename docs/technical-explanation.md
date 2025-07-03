# 🔬 Explicación Técnica - Sistema de Vigilancia IA

Esta documentación explica en detalle el funcionamiento interno, arquitectura y algoritmos del Sistema de Vigilancia con Agentes de IA.

## 🏗️ Arquitectura General

### Visión de Alto Nivel
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Core Engine   │    │   AI Models     │
│   (Streamlit)   │◄──►│   (Python)      │◄──►│   (YOLO+Gemini) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │    │   Video Proc.   │    │   Decision      │
│   • Videos      │    │   • OpenCV      │    │   Engine        │
│   • Commands    │    │   • Frame Ext.  │    │   • Multi-Agent │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🤖 Sistema Multi-Agente

### Arquitectura de Agentes
El sistema implementa una arquitectura de **tres agentes especializados** que trabajan en cadena:

```python
def agent_pipeline():
    """
    Pipeline de procesamiento multi-agente
    """
    # Agente 1: Análisis de Intención
    dispatch_result = interpret_and_dispatch(user_intention)
    
    # Agente 2: Análisis de Escena  
    scene_report = describe_scene_with_gemini(frames, intention)
    
    # Agente 3: Toma de Decisiones
    final_decision = verify_alarm_with_gemini(report, intention)
    
    return final_decision
```

### Agente 1: Despachador Inteligente

**Función:** `interpret_and_dispatch(user_intention: str)`

**Responsabilidades:**
- Análisis semántico de la intención del usuario
- Mapeo de conceptos a clases de objetos
- Selección del pipeline apropiado (Monitor vs Búsqueda)
- Configuración de parámetros de detección

**Algoritmo:**
```python
def interpret_and_dispatch(user_intention: str) -> dict:
    # Base de conocimiento con mapeo semántico
    knowledge_base = {
        "accidente": ["accident", "severe"],
        "fuego": ["fire", "smoke"], 
        "general": ["person", "car", "dog", ...]
    }
    
    # Procesamiento con LLM
    prompt = f"""
    Analiza: "{user_intention}"
    Base de conocimiento: {knowledge_base}
    
    Decide:
    - Workflow: "monitor" o "search"
    - Config: parámetros específicos
    """
    
    return gemini_text_model.generate(prompt)
```

**Ejemplos de Mapeo:**
- "vigila accidentes" → `{"workflow": "monitor", "config": {"accidente": ["accident", "severe"]}}`
- "busca coches rojos" → `{"workflow": "search", "config": {"base_class": "car", "specific_property": "rojo"}}`

### Agente 2: Guardia de Seguridad

**Función:** `describe_scene_with_gemini(frames, user_intention, target_classes)`

**Responsabilidades:**
- Análisis visual detallado de múltiples frames
- Descripción contextual de la escena
- Evaluación inicial de riesgo
- Generación de reporte estructurado

**Procesamiento de Frames:**
```python
def collect_frames_for_analysis(video, frame_idx, duration, fps):
    """
    Recolecta frames antes y después del evento
    para análisis contextual completo
    """
    frames_before = duration * fps // 2
    frames_after = duration * fps // 2
    
    # Buffer circular para frames anteriores
    frame_buffer = deque(maxlen=frames_before)
    
    # Extracción de frames posteriores
    post_frames = extract_subsequent_frames(video, frame_idx, frames_after)
    
    return list(frame_buffer) + post_frames
```

**Análisis Visual:**
```python
def describe_scene_with_gemini(frames, intention, target_classes):
    # Convierte frames a secuencia de imágenes
    image_sequence = [frame_to_pil(frame) for frame in frames]
    
    prompt = f"""
    Eres un guardia de seguridad experto.
    Analiza esta secuencia de {len(frames)} frames.
    
    Contexto: {intention}
    Objetos detectados: {target_classes}
    
    Evalúa:
    1. ¿Qué está ocurriendo exactamente?
    2. ¿Hay riesgo real para personas o propiedad?
    3. ¿Se requiere intervención?
    
    Formato: JSON con evaluación detallada
    """
    
    return vision_model.generate_content([prompt] + image_sequence)
```

### Agente 3: Jefe de Seguridad

**Función:** `verify_alarm_with_gemini(scene_report, user_intention)`

**Responsabilidades:**
- Revisión crítica del reporte del guardia
- Toma de decisión final: ALARMA_CONFIRMADA | INCIDENTE_MENOR | FALSA_ALARMA
- Selección de autoridad competente
- Redacción de mensaje de alerta

**Lógica de Decisión:**
```python
def verify_alarm_with_gemini(scene_report, user_intention):
    prompt = f"""
    Eres el Jefe de Seguridad. Revisa este reporte:
    
    Reporte del Guardia: {scene_report}
    Intención Original: {user_intention}
    
    Decisión requerida:
    - ALARMA_CONFIRMADA: Emergencia real, notificar autoridades
    - INCIDENTE_MENOR: Situación controlable, registro solamente  
    - FALSA_ALARMA: Sin riesgo real, continuar monitoreo
    
    Si ALARMA_CONFIRMADA, determinar:
    - Autoridad: Policía | Bomberos | Servicios Médicos | Control de Animales
    - Mensaje: Descripción concisa para respuesta rápida
    """
    
    return text_model.generate_content(prompt)
```

## 🔍 Detección de Objetos con YOLO

### Modelos Especializados

El sistema usa **tres modelos YOLO especializados**:

```python
models = {
    "accidente": YOLO("weights/best_accident.pt"),  # Classes: ['accident', 'severe']
    "fuego": YOLO("weights/best_fire.pt"),         # Classes: ['fire', 'smoke'] 
    "general": YOLO("weights/best_general.pt")      # Classes: ['person', 'car', ...]
}
```

### Pipeline de Detección

**1. Preprocesamiento de Video:**
```python
def process_video_frames(video_path, frame_skip_rate):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    for frame_idx in range(0, total_frames, frame_skip_rate):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            yield frame_idx, frame
```

**2. Inferencia YOLO:**
```python
def detect_objects(frame, model, confidence_threshold, target_classes):
    # Inferencia del modelo
    results = model(frame, conf=confidence_threshold, verbose=False)
    
    detections = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        
        if class_name.lower() in [cls.lower() for cls in target_classes]:
            bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            
            detections.append({
                "class": class_name,
                "confidence": confidence,
                "bbox": bbox,
                "frame_idx": frame_idx
            })
    
    return detections
```

**3. Post-procesamiento:**
```python
def filter_detections(detections, min_confidence=0.5):
    """
    Filtrado adicional de detecciones
    """
    filtered = []
    for det in detections:
        # Filtro por confianza
        if det["confidence"] < min_confidence:
            continue
            
        # Filtro por tamaño de bounding box
        x1, y1, x2, y2 = det["bbox"]
        area = (x2 - x1) * (y2 - y1)
        if area < 100:  # Muy pequeño
            continue
            
        filtered.append(det)
    
    return filtered
```

## 🔄 Pipelines de Procesamiento

### Pipeline de Monitoreo

**Objetivo:** Vigilancia continua con alertas automáticas

```python
def run_monitor_pipeline(video_path, surveillance_config, params):
    """
    Pipeline de monitoreo en tiempo real
    """
    cap = cv2.VideoCapture(video_path)
    frame_buffer = deque(maxlen=buffer_size)
    events_detected = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Mantener buffer para análisis contextual
        frame_buffer.append(frame.copy())
        
        # Detección con modelos YOLO
        for model_name, target_classes in surveillance_config.items():
            detections = detect_objects(frame, models[model_name], 
                                      confidence_threshold, target_classes)
            
            if detections:
                # Activar análisis multi-agente
                analysis_frames = list(frame_buffer)
                scene_report = describe_scene_with_gemini(analysis_frames, 
                                                        user_intention, target_classes)
                
                decision = verify_alarm_with_gemini(scene_report, user_intention)
                
                # Manejar resultado
                if decision["veredicto_final"] == "ALARMA_CONFIRMADA":
                    handle_confirmed_alarm(decision, frame, video_path)
                    break  # Detener monitoreo
                    
                elif decision["veredicto_final"] == "INCIDENTE_MENOR":
                    log_minor_incident(decision, frame)
                    # Continuar monitoreo
                    
                # FALSA_ALARMA: continuar sin acción
    
    cap.release()
```

### Pipeline de Búsqueda

**Objetivo:** Localización de objetos con características específicas

```python
def run_search_pipeline(video_path, base_class, specific_property, params):
    """
    Pipeline de búsqueda específica
    """
    # Fase 1: Detección masiva con YOLO
    model = get_model_for_class(base_class)
    class_index = get_class_index(model, base_class)
    
    detected_objects = []
    cap = cv2.VideoCapture(video_path)
    
    for frame_idx in range(0, total_frames, frame_skip):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            results = model(frame, classes=[class_index], 
                          conf=confidence_threshold, verbose=False)
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped_object = frame[y1:y2, x1:x2]
                
                detected_objects.append({
                    "image": cropped_object,
                    "frame_idx": frame_idx,
                    "bbox": [x1, y1, x2, y2]
                })
    
    # Fase 2: Muestreo inteligente
    sampled_objects = intelligent_sampling(detected_objects, max_samples=10)
    
    # Fase 3: Verificación con Gemini
    final_matches = []
    for obj in sampled_objects:
        pil_image = Image.fromarray(cv2.cvtColor(obj["image"], cv2.COLOR_BGR2RGB))
        
        if verify_property_with_gemini(pil_image, base_class, specific_property):
            final_matches.append(obj)
    
    return final_matches
```

## 🧠 Integración con Google Gemini

### Configuración y Optimización

```python
def setup_gemini_models():
    """
    Configuración optimizada de modelos Gemini
    """
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Configuración para análisis visual
    vision_config = genai.types.GenerationConfig(
        temperature=0.1,  # Respuestas más determinísticas
        top_p=0.8,
        top_k=40,
        max_output_tokens=1024
    )
    
    # Configuración para análisis de texto
    text_config = genai.types.GenerationConfig(
        temperature=0.2,
        top_p=0.9,
        max_output_tokens=512
    )
    
    vision_model = genai.GenerativeModel('gemini-1.5-flash', 
                                       generation_config=vision_config)
    text_model = genai.GenerativeModel('gemini-1.5-flash',
                                     generation_config=text_config)
    
    return vision_model, text_model
```

### Procesamiento de Imágenes

```python
def prepare_image_for_gemini(cv2_frame):
    """
    Optimización de imágenes para Gemini
    """
    # Conversión de color
    rgb_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
    
    # Redimensionado para optimizar API calls
    height, width = rgb_frame.shape[:2]
    if width > 1024:
        scale = 1024 / width
        new_width = 1024
        new_height = int(height * scale)
        rgb_frame = cv2.resize(rgb_frame, (new_width, new_height))
    
    # Conversión a PIL
    pil_image = Image.fromarray(rgb_frame)
    
    # Compresión para reducir tamaño
    if pil_image.size[0] * pil_image.size[1] > 1024*1024:
        pil_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    
    return pil_image
```

### Manejo de Errores y Reintentos

```python
def robust_gemini_call(model, prompt, images=None, max_retries=3):
    """
    Llamada robusta a Gemini con manejo de errores
    """
    for attempt in range(max_retries):
        try:
            if images:
                response = model.generate_content([prompt] + images)
            else:
                response = model.generate_content(prompt)
            
            # Validación de respuesta
            if response.text and len(response.text.strip()) > 0:
                return parse_response(response.text)
            
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Falló llamada a Gemini después de {max_retries} intentos: {e}")
                return default_response()
            
            # Backoff exponencial
            time.sleep(2 ** attempt)
    
    return default_response()
```

## 🎯 Optimizaciones de Rendimiento

### Gestión de Memoria

```python
class FrameBuffer:
    """
    Buffer circular optimizado para frames de video
    """
    def __init__(self, max_size=50):
        self.buffer = deque(maxlen=max_size)
        self.max_memory_mb = 500  # Límite de memoria
    
    def add_frame(self, frame):
        # Verificar uso de memoria
        current_memory = self.estimate_memory_usage()
        if current_memory > self.max_memory_mb:
            self.reduce_buffer_size()
        
        # Comprimir frame si es necesario
        if frame.nbytes > 1024*1024:  # > 1MB
            frame = self.compress_frame(frame)
        
        self.buffer.append(frame)
    
    def compress_frame(self, frame):
        """Compresión lossy para ahorrar memoria"""
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = 640
            new_height = int(height * scale)
            return cv2.resize(frame, (new_width, new_height))
        return frame
```

### Cache de Modelos

```python
@lru_cache(maxsize=3)
def load_yolo_model(model_path):
    """
    Cache de modelos YOLO para evitar recargas
    """
    return YOLO(model_path)

class ModelManager:
    """
    Gestor inteligente de modelos
    """
    def __init__(self):
        self.models = {}
        self.last_used = {}
        self.max_models_in_memory = 2
    
    def get_model(self, model_name):
        if model_name not in self.models:
            # Liberar memoria si es necesario
            if len(self.models) >= self.max_models_in_memory:
                self.evict_least_used_model()
            
            # Cargar modelo
            model_path = f"weights/best_{model_name}.pt"
            self.models[model_name] = YOLO(model_path)
        
        self.last_used[model_name] = time.time()
        return self.models[model_name]
```

### Procesamiento Paralelo

```python
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

def parallel_frame_processing(frames, model, confidence_threshold):
    """
    Procesamiento paralelo de frames
    """
    num_workers = min(mp.cpu_count(), 4)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Dividir frames en chunks
        chunk_size = len(frames) // num_workers
        frame_chunks = [frames[i:i+chunk_size] 
                       for i in range(0, len(frames), chunk_size)]
        
        # Procesar chunks en paralelo
        futures = [executor.submit(process_frame_chunk, chunk, model, confidence_threshold)
                  for chunk in frame_chunks]
        
        # Recopilar resultados
        all_detections = []
        for future in futures:
            chunk_detections = future.result()
            all_detections.extend(chunk_detections)
    
    return all_detections
```

## 📊 Métricas y Monitoreo

### Sistema de Logging

```python
import logging
from datetime import datetime

class PerformanceLogger:
    """
    Logger especializado para métricas del sistema
    """
    def __init__(self):
        self.setup_logging()
        self.metrics = {
            "frames_processed": 0,
            "detections_found": 0,
            "api_calls": 0,
            "processing_time": [],
            "memory_usage": []
        }
    
    def log_detection_event(self, frame_idx, class_name, confidence, processing_time):
        self.metrics["detections_found"] += 1
        self.metrics["processing_time"].append(processing_time)
        
        logging.info(f"Detection at frame {frame_idx}: {class_name} "
                    f"(conf: {confidence:.2f}, time: {processing_time:.2f}s)")
    
    def log_agent_decision(self, agent_name, decision, reasoning):
        logging.info(f"{agent_name} Decision: {decision} | Reasoning: {reasoning}")
    
    def generate_performance_report(self):
        """
        Genera reporte de rendimiento del sistema
        """
        avg_processing_time = np.mean(self.metrics["processing_time"])
        total_frames = self.metrics["frames_processed"]
        detection_rate = self.metrics["detections_found"] / total_frames if total_frames > 0 else 0
        
        return {
            "total_frames_processed": total_frames,
            "total_detections": self.metrics["detections_found"],
            "detection_rate": detection_rate,
            "avg_processing_time": avg_processing_time,
            "total_api_calls": self.metrics["api_calls"]
        }
```

### Monitoreo en Tiempo Real

```python
def monitor_system_resources():
    """
    Monitoreo de recursos del sistema
    """
    import psutil
    
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "gpu_memory": get_gpu_memory_usage() if torch.cuda.is_available() else None
    }

def adaptive_frame_skip(current_performance):
    """
    Ajuste dinámico del frame skip basado en rendimiento
    """
    cpu_usage = current_performance["cpu_percent"]
    memory_usage = current_performance["memory_percent"]
    
    if cpu_usage > 80 or memory_usage > 85:
        return min(frame_skip * 2, 30)  # Aumentar skip
    elif cpu_usage < 50 and memory_usage < 60:
        return max(frame_skip // 2, 1)  # Reducir skip
    
    return frame_skip  # Mantener actual
```

## 🔒 Consideraciones de Seguridad

### Validación de Entrada

```python
def validate_video_file(file_path, max_size_mb=100):
    """
    Validación exhaustiva de archivos de video
    """
    # Verificar existencia
    if not os.path.exists(file_path):
        raise ValueError("Archivo no encontrado")
    
    # Verificar tamaño
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        raise ValueError(f"Archivo muy grande: {file_size_mb:.1f}MB > {max_size_mb}MB")
    
    # Verificar formato
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError("Formato de video no válido")
        
        # Verificar propiedades básicas
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if frame_count <= 0 or fps <= 0:
            raise ValueError("Video corrupto o sin contenido")
        
        cap.release()
        
    except Exception as e:
        raise ValueError(f"Error al validar video: {str(e)}")

def sanitize_user_input(user_text):
    """
    Sanitización de entrada del usuario
    """
    # Remover caracteres peligrosos
    import re
    sanitized = re.sub(r'[<>"\']', '', user_text)
    
    # Limitar longitud
    if len(sanitized) > 500:
        sanitized = sanitized[:500]
    
    # Verificar contenido apropiado
    forbidden_patterns = ['script', 'eval', 'exec', 'import']
    for pattern in forbidden_patterns:
        if pattern.lower() in sanitized.lower():
            raise ValueError("Entrada contiene contenido no permitido")
    
    return sanitized.strip()
```

### Gestión Segura de API Keys

```python
class SecureConfig:
    """
    Gestión segura de configuración
    """
    def __init__(self):
        self.load_config()
    
    def load_config(self):
        # Cargar desde archivo .env
        load_dotenv()
        
        # Validar API key
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key or len(self.api_key) < 32:
            raise ValueError("API Key de Gemini no válida")
        
        # No almacenar en logs
        self.masked_key = self.api_key[:8] + "..." + self.api_key[-4:]
    
    def configure_gemini(self):
        genai.configure(api_key=self.api_key)
        logging.info(f"Gemini configurado con key: {self.masked_key}")
```

---

## 📚 Referencias Técnicas

### Documentación de APIs
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Google Gemini AI](https://ai.google.dev/docs)
- [OpenCV Python](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Streamlit](https://docs.streamlit.io/)

### Papers de Referencia
- "You Only Look Once: Unified, Real-Time Object Detection" (Redmon et al.)
- "YOLOv5: A State-of-the-Art Real-Time Object Detection" (Ultralytics)
- "Gemini: A Family of Highly Capable Multimodal Models" (Google DeepMind)

### Optimizaciones Adicionales
- Quantización de modelos para reducir tamaño
- TensorRT para aceleración GPU
- ONNX para optimización multiplataforma
- Pruning de modelos para eficiencia

---

**🔬 Esta documentación técnica proporciona las bases para entender, modificar y extender el sistema de vigilancia IA.**
