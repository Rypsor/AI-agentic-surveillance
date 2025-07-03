# 🛠️ Guía de Instalación - Sistema de Vigilancia IA

Esta guía detallada te ayudará a instalar y configurar el Sistema de Vigilancia con Agentes de IA paso a paso.

## 📋 Requisitos Previos

### Requisitos del Sistema

- **Sistema Operativo:** Windows 10/11, macOS 10.15+, o Linux Ubuntu 18.04+
- **Python:** Versión 3.8 o superior
- **RAM:** Mínimo 4GB, recomendado 8GB o más
- **Almacenamiento:** Al menos 2GB libres para modelos y datos

### Cuentas Necesarias

- **Google Cloud Account:** Para acceso a Gemini AI API
- **Git:** Para clonar el repositorio (opcional)

## 🎯 Paso 1: Preparación del Entorno

### 1.1 Verificar Python

```bash
python --version
# Debe mostrar Python 3.8 o superior
```

Si no tienes Python instalado:

- **Windows:** Descarga desde [python.org](https://python.org)
- **macOS:** `brew install python3` (con Homebrew)
- **Linux:** `sudo apt update && sudo apt install python3 python3-pip`

### 1.2 Crear Entorno Virtual (Recomendado)

```bash
# Crear entorno virtual
python -m venv venv_vigilancia

# Activar entorno virtual
# Windows:
venv_vigilancia\Scripts\activate
# macOS/Linux:
source venv_vigilancia/bin/activate
```

## 📥 Paso 2: Obtener el Código

### Opción A: Clonar Repositorio (Recomendado)

```bash
git clone <url-del-repositorio>
cd hackaton-equipo-4
```

### Opción B: Descarga Manual

1. Descargar ZIP del repositorio
2. Extraer en carpeta deseada
3. Navegar a la carpeta extraída

## 🔧 Paso 3: Instalación de Dependencias

### 3.1 Instalar Dependencias Python

```bash
# Actualizar pip
pip install --upgrade pip

# Instalar dependencias del proyecto
pip install -r requirements.txt
```

### 3.2 Verificar Instalación de PyTorch

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

Si hay problemas con PyTorch, instalar manualmente:

```bash
# CPU only
pip install torch torchvision torchaudio

# Con CUDA (GPU NVIDIA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🔑 Paso 4: Configuración de API Keys

### 4.1 Obtener Clave de Google Gemini

1. Ir a [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Iniciar sesión con cuenta de Google
3. Crear nueva API Key
4. Copiar la clave generada

### 4.2 Crear Archivo de Configuración

Crear archivo `.env` en la raíz del proyecto:

```bash
# Crear archivo .env
touch .env  # Linux/macOS
# En Windows: crear archivo .env manualmente
```

Contenido del archivo `.env`:

```env
# API Key de Google Gemini (OBLIGATORIO)
GEMINI_API_KEY=tu_clave_api_aqui

# Modelos de Gemini (OPCIONAL - usa valores por defecto si no se especifica)
VISION_MODEL=gemini-1.5-flash
TEXT_MODEL=gemini-1.5-flash

# Configuraciones adicionales (OPCIONAL)
DEBUG=False
MAX_FILE_SIZE=100MB
```

⚠️ **IMPORTANTE:** Nunca subas el archivo `.env` a control de versiones.

## 📁 Paso 5: Preparar Estructura de Archivos

### 5.1 Verificar Estructura de Carpetas

```bash
# Verificar que existen las carpetas necesarias
ls -la
```

Estructura esperada:

```
hackaton-equipo-4/
├── app.py                    # ✅ Debe existir
├── requirements.txt          # ✅ Debe existir
├── .env                      # ✅ Que acabas de crear
├── weights/                  # ✅ Debe existir con modelos
│   ├── best_accident.pt      # ✅ Modelo de accidentes
│   ├── best_fire.pt          # ✅ Modelo de incendios
│   └── best_general.pt       # ✅ Modelo general
├── videos/                   # ✅ Para videos de entrada
├── output/                   # ✅ Se crea automáticamente
│   └── clips/                # ✅ Se crea automáticamente
└── output_captures/          # ✅ Para capturas
```

### 5.2 Crear Carpetas Faltantes

```bash
# Crear carpetas si no existen
mkdir -p videos output output_captures output/clips
```

## 📦 Paso 6: Descargar Modelos YOLO

### 6.1 Verificar Modelos Existentes

```bash
ls -la weights/
```

### 6.2 Si Faltan Modelos

Los modelos personalizados (`best_*.pt`) deben estar incluidos en el repositorio. Si faltan:

1. **Contactar al equipo de desarrollo** para obtener los modelos entrenados
2. **O usar modelos base** (menos precisos):

```python
# Código temporal para usar modelos base
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Descarga automáticamente
```

## 🚀 Paso 7: Primera Ejecución

### 7.1 Prueba de Configuración

```bash
# Verificar que Python encuentra todas las dependencias
python -c "
import streamlit
import cv2
import torch
from ultralytics import YOLO
import google.generativeai as genai
print('✅ Todas las dependencias están instaladas correctamente')
"
```

### 7.2 Ejecutar la Aplicación

```bash
streamlit run app.py
```

### 7.3 Verificar Funcionamiento

1. Abrir navegador en `http://localhost:8501`
2. Verificar que la interfaz carga correctamente
3. Probar subir un video pequeño como prueba

## 🔍 Paso 8: Solución de Problemas de Instalación

### Error: ModuleNotFoundError

```bash
# Reinstalar dependencia específica
pip install nombre_del_modulo

# O reinstalar todo
pip install -r requirements.txt --force-reinstall
```

### Error: API Key inválida

1. Verificar que la clave está correctamente copiada en `.env`
2. Verificar que no hay espacios extra
3. Regenerar la clave en Google AI Studio

### Error: Puerto ocupado

```bash
# Usar puerto diferente
streamlit run app.py --server.port 8502
```

### Error: Archivo de video no compatible

- Usar formatos: MP4, AVI, MOV
- Verificar que el archivo no esté corrupto
- Reducir tamaño si es muy grande (>100MB)

## 🔄 Actualización

### Actualizar Dependencias

```bash
pip install -r requirements.txt --upgrade
```

### Actualizar Código

```bash
git pull origin main  # Si usas Git
```

## 🧪 Verificación Final

### Test Completo

1. ✅ La aplicación inicia sin errores
2. ✅ Interfaz web es accesible
3. ✅ Se puede subir un video
4. ✅ El procesamiento funciona (aunque sea lento)
5. ✅ Se generan resultados visuales

### Comandos de Diagnóstico

```bash
# Verificar versiones
python --version
pip list | grep -E "(streamlit|torch|ultralytics|opencv|google)"

# Test de conectividad API
python -c "
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
if api_key:
    genai.configure(api_key=api_key)
    print('✅ API Key configurada correctamente')
else:
    print('❌ API Key no encontrada')
"
```

---

**🎉 ¡Felicitaciones! Si llegaste hasta aquí, tu sistema está listo para usar.**

Continúa con la [Guía Técnica](technical-explanation.md) para entender cómo funciona el sistema internamente.
