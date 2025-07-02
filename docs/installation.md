# 📦 Guía de Instalación

## Sistema de Vigilancia con Agentes de IA

Esta guía te llevará paso a paso por el proceso de instalación del sistema de vigilancia inteligente.

## 📋 Prerrequisitos

### Requisitos de Sistema
- **Sistema Operativo**: Windows 10+, macOS 10.14+, o Linux (Ubuntu 18.04+)
- **Python**: Versión 3.8 o superior
- **RAM**: Mínimo 8GB (recomendado 16GB)

### Requisitos de Software
- Git (para clonar el repositorio)
- Python 3.8+ con pip
- Navegador web moderno

## 🚀 Instalación Paso a Paso

### 1. Verificar Instalación de Python

Primero, verifica que tienes Python instalado:

```bash
python --version
# o en algunos sistemas:
python3 --version
```

Si no tienes Python instalado, descárgalo desde [python.org](https://www.python.org/downloads/).

### 2. Clonar el Repositorio

```bash
git clone <repository-url>
cd hackaton-equipo-4
```

### 3. Crear y Activar Entorno Virtual

#### En Windows:
```cmd
python -m venv .venv
.venv\Scripts\activate
```

#### En macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Tu terminal debería mostrar `(.venv)` al inicio de la línea, indicando que el entorno virtual está activo.

### 4. Actualizar pip y Herramientas Básicas

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 5. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 6. Configurar Variables de Entorno

#### 6.1 Crear archivo .env
Modifica el archivo `.env` en la raíz del proyecto.

#### 6.2 Configurar Clave API de Gemini

1. Ve a [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Crea una nueva clave API
3. Edita el archivo `.env` y reemplaza `<API_KEY>` con tu clave:

```env
# .env
GEMINI_API_KEY="tu_clave_api_aqui"
DEBUG_MODE=true
TEXT_MODEL="gemini-2.0-flash"
VIDEO_MODEL="gemini-1.5-flash"
```

### 7. Descargar y Configurar Modelos YOLO

#### 7.1 Crear Directorio de Pesos
```bash
mkdir -p weights
```

#### 7.2 Descargar Modelos Preentrenados

Los modelos YOLO deben colocarse en la carpeta `weights/`. Asegúrate de tener:

- `best_accident.pt` - Modelo para detección de accidentes
- `best_fire.pt` - Modelo para detección de fuego  
- `best_general.pt` - Modelo general para objetos comunes

```bash
# Ejemplo de descarga (reemplaza con las URLs reales)
wget -O weights/best_general.pt <URL_DEL_MODELO_GENERAL>
wget -O weights/best_accident.pt <URL_DEL_MODELO_ACCIDENTES>
wget -O weights/best_fire.pt <URL_DEL_MODELO_FUEGO>
```

### 8. Verificar Instalación

Ejecuta el script de verificación para asegurar que todo está configurado correctamente:

```bash
python scripts/check_quality.py
```

O verifica manualmente cargando las dependencias principales:

```bash
python -c "
import streamlit as st
import cv2
import google.generativeai as genai
from ultralytics import YOLO
from dotenv import load_dotenv
import os

load_dotenv()
print('✅ Todas las dependencias cargadas correctamente')
print('✅ GEMINI_API_KEY:', 'Configurada' if os.getenv('GEMINI_API_KEY') else 'NO CONFIGURADA')
"
```

## 🧪 Verificación Final

### 1. Ejecutar Tests
```bash
python -m pytest tests/ -v
```

### 2. Ejecutar la Aplicación
```bash
streamlit run app.py
```

Si ves el mensaje:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

¡Felicidades! 🎉 La instalación fue exitosa.

## 🆘 Solución de Problemas

### Error: "ModuleNotFoundError"
```bash
# Asegúrate de que el entorno virtual esté activo
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate     # Windows

# Reinstala las dependencias
pip install -r requirements.txt
```

### Error: "GEMINI_API_KEY not found"
- Verifica que el archivo `.env` existe en la raíz del proyecto
- Asegúrate de que la clave API no tenga espacios extra
- Reinicia la aplicación después de modificar el `.env`

### Error: "No such file or directory: weights/"
```bash
# Crea el directorio de pesos
mkdir -p weights

# Verifica que los archivos .pt estén presentes
ls -la weights/
```

### Problemas de Memoria
- Reduce la resolución de los videos de prueba
- Aumenta el parámetro de "frame skip" en la interfaz
- Cierra otras aplicaciones que consuman RAM

### Problemas de Rendimiento
- Asegúrate de usar un entorno virtual
- Verifica si tienes GPU compatible
- Usa videos cortos para las primeras pruebas

## 📞 Soporte

Si encuentras problemas durante la instalación:

1. Revisa la sección de solución de problemas arriba
2. Consulta los logs de error para más detalles
3. Verifica que cumples todos los prerrequisitos
4. Considera usar la instalación con Docker como alternativa

## 🔄 Actualizaciones

Para actualizar el sistema:

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

---

📝 **Nota**: Esta guía asume conocimiento básico de terminal/línea de comandos. Si eres nuevo en esto, considera buscar tutoriales de terminal para tu sistema operativo.