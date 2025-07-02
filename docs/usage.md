# 📚 Guía de Uso

## Sistema de Vigilancia con Agentes de IA

Esta guía te enseñará a usar el sistema de vigilancia inteligente para detectar eventos y buscar objetos específicos en videos.

## 🚀 Inicio Rápido

### 1. Ejecutar la Aplicación
```bash
# Activa el entorno virtual
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate     # Windows

# Ejecuta la aplicación
streamlit run app.py
```

### 2. Abrir en el Navegador
- La aplicación se abrirá automáticamente en tu navegador
- Si no, ve manualmente a: `http://localhost:8501`

## 🎯 Interfaz de Usuario

### Panel Principal
La aplicación se divide en dos secciones principales:

#### 📁 **1. Carga tu Video**
- Formatos soportados: MP4, MOV, AVI
- Tamaño recomendado: Máximo 100MB para pruebas
- Resolución óptima: 720p-1080p

#### ⚙️ **2. Define la Tarea y Configuración**
- **Descripción de Intención**: Campo de texto para describir qué quieres detectar
- **Confianza de Detección**: Control deslizante (0.1-1.0)
- **Analizar 1/N frames**: Frecuencia de análisis
- **Duración de Captura**: Tiempo de contexto temporal

## 🔄 Modos de Operación

### 🎯 **Modo Monitoreo**
Para vigilancia de eventos específicos como accidentes, incendios, etc.

#### Ejemplos de Intenciones:
```
"Detectar accidentes de tránsito en esta intersección"
"Identificar incendios en el edificio"
"Vigilar actividad sospechosa en el área"
"Monitorear caídas de personas"
```

#### Flujo de Trabajo:
1. **Detección YOLO**: Análisis frame por frame
2. **Activación**: Cuando se detecta objeto objetivo
3. **Captura Temporal**: Recolección de secuencia
4. **Análisis Contextual**: Evaluación por IA
5. **Verificación**: Confirmación final

### 🔍 **Modo Búsqueda**
Para encontrar objetos con características específicas.

#### Ejemplos de Intenciones:
```
"Encontrar todos los coches rojos"
"Buscar personas con camisetas azules"
"Localizar perros en el parque"
"Identificar vehículos de emergencia"
```

#### Flujo de Trabajo:
1. **Detección Masiva**: Encuentra todos los objetos de la clase
2. **Muestreo**: Selecciona muestras representativas
3. **Verificación Visual**: Confirma características específicas
4. **Resultados**: Muestra coincidencias encontradas

## 🎛️ Configuración de Parámetros

### 📊 **Confianza de Detección (0.1-1.0)**
- **0.1-0.3**: Máxima sensibilidad, más falsos positivos
- **0.4-0.6**: Balance recomendado
- **0.7-1.0**: Alta precisión, puede perder algunas detecciones

**Recomendación**: Comienza con 0.4 y ajusta según resultados.

### ⚡ **Analizar 1/N frames**
- **1-5**: Análisis denso, mayor precisión, más lento
- **6-15**: Balance entre velocidad y precisión
- **16-30**: Análisis rápido, menos preciso

**Recomendación**: Usa 5 para pruebas, 10-15 para producción.

### ⏱️ **Duración de Captura (Solo Monitoreo)**
- **5 segundos**: Contexto rápido
- **10 segundos**: Contexto completo

## 💡 Ejemplos Prácticos

### Ejemplo 1: Detectar Accidentes de Tráfico

**Configuración**:
- **Intención**: `"Detectar accidentes de tránsito en esta carretera"`
- **Confianza**: `0.5`
- **Frame Skip**: `5`
- **Duración**: `10 segundos`

**Resultado Esperado**:
- Detección automática de colisiones
- Análisis temporal del evento
- Verificación de la gravedad

### Ejemplo 2: Buscar Vehículos Específicos

**Configuración**:
- **Intención**: `"Encontrar todos los coches de color azul"`
- **Confianza**: `0.4`
- **Frame Skip**: `10`

**Resultado Esperado**:
- Lista de todos los vehículos azules
- Imágenes recortadas de cada detección
- Timestamp de aparición

### Ejemplo 3: Monitorear Seguridad

**Configuración**:
- **Intención**: `"Detectar personas que corren o se comportan de manera sospechosa"`
- **Confianza**: `0.6`
- **Frame Skip**: `3`
- **Duración**: `10 segundos`

**Resultado Esperado**:
- Detección de comportamientos anómalos
- Análisis contextual de la situación
- Alerta con nivel de confianza

## 📊 Interpretación de Resultados

### 🚨 **Pipeline de Monitoreo**

#### **Agente 1 (Despachador)**
```json
{
  "workflow": "monitor",
  "config": {
    "accidente": ["car", "person", "motorcycle"]
  }
}
```

#### **Agente 2 (Analista)**
```json
{
  "scene_description": "Se observa un vehículo detenido en posición anormal...",
  "threat_level": "HIGH",
  "objects_detected": ["car", "person"],
  "recommended_action": "dispatch_emergency"
}
```

#### **Agente 3 (Supervisor)**
```json
{
  "veredicto": "ALARMA_REAL",
  "justificacion": "Confirmo presencia de accidente vehicular...",
  "confidence": 0.89
}
```

### 🔍 **Pipeline de Búsqueda**

#### **Resultados de Búsqueda**
- **Detecciones Totales**: Número de objetos encontrados
- **Coincidencias Verificadas**: Objetos que cumplen los criterios
- **Galería Visual**: Imágenes de las coincidencias
- **Timestamps**: Momento de aparición en el video

## ⚙️ Consejos de Optimización

### 🎥 **Preparación de Videos**

#### Calidad Óptima:
- **Resolución**: 720p-1080p
- **FPS**: 24-30 fps
- **Duración**: 30 segundos - 5 minutos para pruebas
- **Iluminación**: Buena visibilidad
- **Estabilidad**: Videos sin mucho movimiento de cámara

#### Evitar:
- Videos muy oscuros o con contraluz
- Resoluciones muy altas (>1080p) para pruebas
- Videos muy largos (>10 minutos) sin necesidad

### 🧠 **Formulación de Intenciones**

#### ✅ **Buenas Prácticas**:
```
"Detectar accidentes de tránsito" ✓
"Encontrar coches rojos" ✓
"Identificar personas corriendo" ✓
"Buscar animales en el video" ✓
```

#### ❌ **Evitar**:
```
"Encontrar cosas raras" ❌ (demasiado vago)
"Detectar todo" ❌ (demasiado amplio)
"Buscar objetos pequeños" ❌ (sin especificar qué objetos)
```

### ⚡ **Rendimiento**

#### Para Videos Largos:
1. Aumenta el frame skip (15-30)
2. Reduce la duración de captura (5 segundos)
3. Usa resolución más baja
4. Procesa en segmentos más pequeños

#### Para Máxima Precisión:
1. Frame skip bajo (3-5)
2. Confianza moderada (0.4-0.6)
3. Duración de captura completa (10 segundos)
4. Videos con buena iluminación

## 🔧 Solución de Problemas Comunes

### **"No se detecta nada"**
- Reduce la confianza de detección
- Verifica que el objeto esté en el video
- Reformula la intención más específicamente
- Revisa la calidad del video

### **"Demasiados falsos positivos"**
- Aumenta la confianza de detección
- Usa intenciones más específicas
- Verifica la iluminación del video

### **"Procesamiento muy lento"**
- Aumenta el frame skip
- Reduce la resolución del video
- Usa videos más cortos para pruebas
- Cierra otras aplicaciones

### **"Error de memoria"**
- Reduce la duración de captura
- Aumenta el frame skip
- Usa videos más pequeños
- Reinicia la aplicación

## 📈 Monitoreo de Rendimiento

### **Métricas en Tiempo Real**
- **FPS de Procesamiento**: Visible en la barra de progreso
- **Memoria Utilizada**: Monitorea el uso de RAM
- **Tiempo de Respuesta**: De cada agente

### **Logs de Debug**
Si tienes `DEBUG_MODE=true` en `.env`:
```bash
# Ver logs en terminal
tail -f logs/app.log
```

## 🎯 Casos de Uso Avanzados

### **Vigilancia de Tráfico**
1. **Detección de Infracciones**:
   - "Detectar vehículos que no respetan el semáforo"
   - "Identificar automóviles en carril de bicicletas"

2. **Análisis de Flujo**:
   - "Contar vehículos que pasan por hora"
   - "Detectar congestión de tráfico"

### **Seguridad Industrial**
1. **Equipos de Protección**:
   - "Verificar que los trabajadores usen casco"
   - "Detectar personas sin chaleco reflectivo"

2. **Incidentes de Seguridad**:
   - "Identificar caídas de trabajadores"
   - "Detectar fugas o derrames"

### **Monitoreo Ambiental**
1. **Vida Silvestre**:
   - "Contar animales en reserva natural"
   - "Detectar especies en peligro"

2. **Eventos Naturales**:
   - "Identificar incendios forestales"
   - "Detectar inundaciones"

## 🔄 Flujo de Trabajo Recomendado

### **Para Nuevos Usuarios**:
1. Comienza con videos cortos (30 segundos)
2. Usa configuración por defecto
3. Prueba intenciones simples
4. Familiarízate con los resultados
5. Ajusta parámetros gradualmente

### **Para Usuarios Avanzados**:
1. Personaliza modelos YOLO
2. Ajusta parámetros según caso de uso
3. Implementa pipelines personalizados
4. Integra con sistemas externos

## 📞 Soporte y Recursos

### **Documentación Adicional**:
- `README.md`: Descripción general del proyecto
- `docs/installation.md`: Guía de instalación detallada
- Código fuente comentado en `src/`

### **Ejemplos y Tutoriales**:
- Videos de ejemplo en `videos/`
- Scripts de ejemplo en `scripts/`
- Configuraciones de prueba

---

💡 **Tip**: Experimenta con diferentes configuraciones para encontrar el balance perfecto entre velocidad y precisión para tu caso de uso específico.