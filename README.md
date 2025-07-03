#  Sistema de Vigilancia con Agentes de IA
Un sistema avanzado de vigilancia inteligente que utiliza múltiples modelos de IA (YOLOv5 + Google Gemini) para el análisis automático de videos de seguridad.

## Link de YouTube:
https://youtu.be/KTCd0Cu397o



##  Características Principales

- **🔍 Doble Modo de Operación:**
  - **Monitoreo en Tiempo Real:** Vigilancia continua con alertas automáticas
  - **Búsqueda Específica:** Localización de objetos con características particulares

- ** Arquitectura Multi-Agente:**
  - **Agente Despachador:** Analiza la intención del usuario y selecciona el pipeline apropiado
  - **Agente Guardia:** Describe escenas y evalúa situaciones detectadas
  - **Agente Jefe de Seguridad:** Toma decisiones finales y determina acciones a seguir

- ** Detección Especializada:**
  - **Accidentes:** Choques vehiculares y evaluación de gravedad
  - **Incendios:** Detección de fuego y humo
  - **Objetos Generales:** Personas, vehículos, animales domésticos

##  Tecnologías Utilizadas

- **Frontend:** Streamlit
- **Detección de Objetos:** YOLOv5 (Ultralytics)
- **Análisis Visual:** Google Gemini AI
- **Procesamiento de Video:** OpenCV
- **Procesamiento de Imágenes:** PIL/Pillow
- **Framework ML:** PyTorch

##  Requisitos del Sistema

- Python 3.8+
- GPU recomendada para mejor rendimiento
- Clave API de Google Gemini
- Al menos 4GB de RAM
- Espacio en disco suficiente para videos y clips generados

##  Instalación

1. **Clonar el repositorio:**
```bash
git clone <url-del-repositorio>
cd hackaton-equipo-4
```

2. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

3. **Configurar variables de entorno:**
Crear un archivo `.env` en la raíz del proyecto:
```env
GEMINI_API_KEY=tu_clave_api_aqui
VISION_MODEL=gemini-1.5-flash
TEXT_MODEL=gemini-1.5-flash
```

4. **Verificar estructura de archivos:**
```
hackaton-equipo-4/
├── app.py
├── requirements.txt
├── .env
├── weights/
│   ├── best_accident.pt
│   ├── best_fire.pt
│   └── best_general.pt
├── videos/
├── output/
└── output_captures/
```

##  Uso

1. **Ejecutar la aplicación:**
```bash
streamlit run app.py
```

2. **Acceder a la interfaz web:**
   - Abrir navegador en `http://localhost:8501`

3. **Cargar video y configurar:**
   - Subir archivo de video (MP4, AVI, MOV)
   - Describir la tarea en lenguaje natural
   - Ajustar parámetros de detección
   - Hacer clic en "Procesar Video"

##  Ejemplos de Uso

### Modo Monitoreo
- *"Vigila si hay algún accidente automovilístico"*
- *"Alerta si detectas fuego o humo"*
- *"Monitorea la presencia de intrusos"*
- *"Avísame cuando aparezcan animales"*

### Modo Búsqueda
- *"Encuentra coches de color rojo"*
- *"Busca personas con ropa oscura"*
- *"Localiza camiones estacionados"*
- *"Identifica perros grandes"*

##  Arquitectura del Sistema

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Agente 1:      │    │  Agente 2:      │    │  Agente 3:      │
│  Despachador    │───▶│  Guardia        │───▶│  Jefe Seguridad │
│                 │    │                 │    │                 │
│ • Analiza       │    │ • Describe      │    │ • Decide        │
│   intención     │    │   escenas       │    │   acciones      │
│ • Selecciona    │    │ • Evalúa        │    │ • Notifica      │
│   pipeline      │    │   situaciones   │    │   autoridades   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

##  Estructura del Proyecto

```
├── app.py                 # Aplicación principal
├── requirements.txt       # Dependencias Python
├── README.md             # Documentación
├── LICENSE               # Licencia MIT
├── .env                  # Variables de entorno (no incluido)
├── weights/              # Modelos YOLO entrenados
│   ├── best_accident.pt  # Modelo para accidentes
│   ├── best_fire.pt      # Modelo para incendios
│   └── best_general.pt   # Modelo general
├── videos/               # Videos de entrada
├── output/               # Archivos de salida
│   └── clips/           # Clips de eventos detectados
└── output_captures/      # Capturas de pantalla
```

##  Configuración Avanzada

### Parámetros de Detección
- **Confianza:** Umbral mínimo para considerar una detección válida (0.1-1.0)
- **Salto de Frames:** Frecuencia de análisis para optimizar rendimiento
- **Duración de Captura:** Segundos de video a extraer alrededor de eventos

### Modelos Disponibles
- **Accidentes:** `['accident', 'severe']`
- **Incendios:** `['fire', 'smoke']`
- **General:** `['person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck', 'cat', 'dog']`

##  Consideraciones de Seguridad

- Mantener la clave API de Gemini segura
- No subir archivos `.env` al control de versiones
- Revisar permisos de archivos generados
- Validar entrada de videos antes del procesamiento

##  Resolución de Problemas

### Errores Comunes

1. **Error de API Key:**
   - Verificar que `GEMINI_API_KEY` esté correctamente configurada
   - Comprobar límites de uso de la API

2. **Modelos no encontrados:**
   - Asegurar que los archivos `.pt` estén en la carpeta `weights/`
   - Verificar permisos de lectura

3. **Error de video:**
   - Comprobar formato de video compatible
   - Verificar que el archivo no esté corrupto

##  Contribuciones

Las contribuciones son bienvenidas. Para contribuir:

1. Fork el proyecto
2. Crear una rama para la feature (`git checkout -b feature/AmazingFeature`)
3. Commit los cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

##  Casos de Uso

- **Seguridad Industrial:** Monitoreo de accidentes en plantas
- **Seguridad Vial:** Análisis de tráfico y accidentes
- **Seguridad Comercial:** Vigilancia de establecimientos
- **Investigación:** Análisis forense de videos
- **Smart Cities:** Monitoreo urbano inteligente

##  Créditos

Desarrollado como parte del Hackathon UNAL 2025 por el Equipo 4.

##  Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

##  Soporte

Para soporte y preguntas:
- Crear un Issue en el repositorio
- Contactar al equipo de desarrollo

---

**  Powered by AI | Built with  for Smart Security**
