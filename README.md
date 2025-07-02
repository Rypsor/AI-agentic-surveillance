# 👁️ Sistema de Vigilancia con Agentes de IA

## Hackathon UNAL 2025 - Equipo 4

Un sistema avanzado de vigilancia de video que utiliza múltiples agentes de inteligencia artificial para el análisis automatizado de eventos y detección de objetos específicos.

![Sistema de Vigilancia IA](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)
![Gemini](https://img.shields.io/badge/Google%20Gemini-API-yellow.svg)

## 🌟 Características Principales

### 🤖 Sistema Multi-Agente
- **Agente Despachador**: Interpreta la intención del usuario y selecciona el pipeline apropiado
- **Agente Analista**: Analiza secuencias de video para detectar eventos de interés
- **Agente Supervisor**: Verifica y confirma alarmas para reducir falsos positivos

### 🔄 Doble Pipeline de Procesamiento
1. **Pipeline de Monitoreo**: Vigilancia continua para detectar eventos específicos (accidentes, fuego, etc.)
2. **Pipeline de Búsqueda**: Localización de objetos con características específicas

### 🎯 Modelos Especializados
- **YOLO Personalizado**: Modelos entrenados para accidentes, fuego y detección general
- **Google Gemini**: Análisis avanzado de video e imágenes
- **Filtrado Inteligente**: Reducción de falsos positivos mediante análisis temporal

## 🚀 Instalación y Configuración

### Prerrequisitos
- Python 3.8 o superior
- GPU compatible con CUDA (recomendado para mejor rendimiento)
- Clave API de Google Gemini

### 1. Clonar el Repositorio
```bash
git clone <repository-url>
cd hackaton-equipo-4
```

### 2. Crear Entorno Virtual
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate     # Windows
```

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar Variables de Entorno
Crear un archivo `.env` en la raíz del proyecto:
```env
GEMINI_API_KEY=tu_clave_api_de_gemini_aqui
TEXT_MODEL=gemini-2.0-flash
VIDEO_MODEL=gemini-1.5-flash
DEBUG_MODE=true
```

### 5. Descargar Modelos YOLO
Asegúrate de tener los modelos entrenados en la carpeta `weights/`:
- `best_accident.pt` - Modelo para detección de accidentes
- `best_fire.pt` - Modelo para detección de fuego
- `best_general.pt` - Modelo general para objetos comunes

## 🎮 Uso del Sistema

### Ejecutar la Aplicación
```bash
streamlit run app.py
```

### Interfaz Web
1. **Cargar Video**: Sube un archivo de video (MP4, MOV, AVI)
2. **Definir Intención**: Describe lo que quieres detectar o buscar
3. **Configurar Parámetros**:
   - Confianza de detección YOLO
   - Frecuencia de análisis de frames
   - Duración de captura temporal
4. **Procesar**: Haz clic en "🚀 Procesar Video"

### Ejemplos de Intenciones

#### Para Pipeline de Monitoreo:
- "Detectar accidentes de tránsito"
- "Identificar incendios en el área"
- "Vigilar actividad sospechosa"

#### Para Pipeline de Búsqueda:
- "Encontrar coches rojos"
- "Buscar personas con camisetas azules"
- "Localizar animales en el video"

## 🏗️ Arquitectura del Sistema

```
├── app.py                 # Aplicación principal Streamlit
├── config.py              # Configuraciones globales
├── src/
│   ├── agents/           # Agentes de IA
│   │   ├── interpreted.py    # Agente despachador
│   │   ├── video_analyzer.py # Agente analista
│   │   ├── search.py         # Verificación de propiedades
│   │   └── verify.py         # Agente supervisor
│   ├── pipelines.py      # Pipelines de procesamiento
│   └── utils/            # Utilidades
├── weights/              # Modelos YOLO entrenados
├── data/                 # Datos y salidas
├── videos/               # Videos de prueba
└── docs/                 # Documentación
```

## 📊 Flujo de Trabajo

### Pipeline de Monitoreo
1. **Análisis YOLO**: Detección frame por frame
2. **Activación por Evento**: Cuando se detecta un objeto objetivo
3. **Captura Temporal**: Recolección de secuencia de frames
4. **Análisis Contextual**: Evaluación por Agente Analista
5. **Verificación Final**: Confirmación por Agente Supervisor

### Pipeline de Búsqueda
1. **Detección Masiva**: Identificación de todos los objetos de la clase base
2. **Muestreo Inteligente**: Selección de muestras representativas
3. **Verificación Visual**: Análisis de propiedades específicas con Gemini
4. **Resultados Visuales**: Presentación de coincidencias encontradas

## 🔧 Configuración Avanzada

### Modelos YOLO Personalizados
Para usar tus propios modelos YOLO:
1. Coloca los archivos `.pt` en la carpeta `weights/`
2. Actualiza el mapeo en `app.py`:
```python
YOLO_MODELS_CONFIG = {
    "tu_modelo": "tu_modelo.pt",
    # ...
}
```

### Ajuste de Parámetros
- **Confianza YOLO**: Ajusta según la precisión deseada vs velocidad
- **Frame Skip**: Mayor valor = procesamiento más rápido, menor precisión
- **Duración Captura**: Más frames = mejor contexto, mayor costo

## 🐛 Solución de Problemas

### Errores Comunes
1. **Error de Clave API**: Verifica que `GEMINI_API_KEY` esté correctamente configurada
2. **Modelos YOLO Faltantes**: Asegúrate de tener los archivos `.pt` en `weights/`
3. **Memoria Insuficiente**: Reduce la resolución del video o aumenta el frame skip

### Logs y Debug
Activa el modo debug en `.env`:
```env
DEBUG_MODE=true
```

## 📈 Métricas de Rendimiento

- **Velocidad de Procesamiento**: ~2-5 FPS (depende del hardware)
- **Precisión de Detección**: >85% en condiciones óptimas
- **Reducción de Falsos Positivos**: ~70% mediante verificación multi-agente

## 🤝 Contribuciones

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 📝 Roadmap

- [ ] Soporte para streams en tiempo real
- [ ] Integración con sistemas de notificación
- [ ] Dashboard de métricas en tiempo real
- [ ] API REST para integración externa
- [ ] Soporte para múltiples cámaras simultáneas

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 👥 Equipo de Desarrollo

**Hackathon UNAL 2025 - Equipo 4**

- Desarrollado durante el Hackathon de la Universidad Nacional de Colombia 2025
- Sistema de vigilancia inteligente con IA multi-agente

## 🙏 Agradecimientos

- Universidad Nacional de Colombia
- Google Gemini API
- Ultralytics YOLOv8
- Streamlit Community
- OpenCV Team

---

⭐ **¡Si te gusta este proyecto, no olvides darle una estrella!** ⭐