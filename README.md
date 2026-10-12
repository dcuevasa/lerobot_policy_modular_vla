# LeRobot Modular Hierarchical VLA Policy (`lerobot_policy_modular_vla`)

> **Arquitectura de "Cerebro Dividido" (*Divided Brain*):** Reemplazo modular y *plug-and-play* para VLAs monolíticos gigantes dentro del ecosistema [🤗 LeRobot](https://github.com/huggingface/lerobot).

---

## 1. Motivación y Concepto Central

Los modelos **VLA (*Vision-Language-Action*) monolíticos** (como RT-2, OpenVLA, Octo, $\pi_0$, SmolVLA) agrupan la percepción visual completa, el razonamiento de alto nivel y el control motor continuo de bajo nivel en una sola red neuronal masiva. Esto presenta tres barreras críticas:

1. **Latencia y Cómputo Excesivo:** Dificultad para mantener bucles cerrados a 30–50 Hz en hardware accesible.
2. **Olvido Catastrófico (*Catastrophic Forgetting*):** Adaptar el modelo a una nueva habilidad puede degradar severamente las habilidades previas.
3. **Ambigüedad Visual:** Instrucciones en texto libre sufren distracciones ante fondos desordenados o múltiples objetos.

### La Solución: Un "AI MODEL" Modular en LeRobot
Este paquete implementa la arquitectura descrita en la [Wiki de Tesis](https://github.com/dcuevasa/docs_tesis/wiki/Arquitectura-del-Sistema):
* **Orquestador VLM (Nivel Alto):** Planifica la secuencia de subtareas atómicas y decide qué objeto manipular.
* **Aislamiento Perceptual (Bounding Boxes y Máscaras):** El detector acota espacialmente el objeto objetivo. **Las políticas motoras no reciben el nombre semántico del objeto, sino que operan directamente sobre el recorte (*crop*) o máscara visual.**
* **Micro-Expertos (Nivel Bajo):** Políticas ligeras y especializadas (ACT, Diffusion Policy, etc.) que ejecutan micro-tareas motoras precisas.
* **Model Swapping Dinámico:** Se carga en memoria GPU únicamente el micro-modelo activo, garantizando bajo consumo de VRAM y protección estructural contra olvido catastrófico.
* **Opción CORAL Switching:** Preparado para conmutación instantánea de cabezales LoRA manteniendo el *backbone* congelado.
* **Superposición Visual en Grabaciones (HUD & Bboxes):** Quema en vivo la telemetría, el plan activo y las cajas envolventes directamente sobre el video grabado.

---

## 2. Compatibilidad Plug-and-Play con LeRobot

Este paquete implementa `PreTrainedPolicy` y `PreTrainedConfig`, registrándose automáticamente como plugin de tercera parte en LeRobot mediante el prefijo `lerobot_policy_`:

```python
from lerobot.policies.factory import get_policy_class, make_policy
from lerobot_policy_modular_vla import ModularVLAConfig, ModularVLAPolicy

# Descubrimiento automático en el registro de LeRobot
policy_cls = get_policy_class("modular_vla")  # o "hierarchical_vla"
```

Cualquier comando CLI de LeRobot puede invocar este modelo:
```bash
lerobot-eval --policy.type=modular_vla ...
lerobot-vision-eval --policy.type=modular_vla ...
```

---

## 3. Características Principales

### A. Proveedores VLM Totalmente Intercambiables
Cambiar el motor de razonamiento es cuestión de modificar dos parámetros (`provider` y `model_name`):

| Proveedor (`provider`) | Modelos Soportados | Endpoint / Configuración |
| :--- | :--- | :--- |
| **`ollama`** | `qwen2.5-vl:7b`, `llava`, `minicpm-v`, `llama3.2-vision` | `api_base: "http://localhost:11434"` (Local / Edge) |
| **`openai`** | `gpt-4o`, `gpt-4o-mini`, vLLM, LiteLLM | `api_key: "sk-..."`, `api_base: "..."` |
| **`gemini`** | `gemini-2.0-flash`, `gemini-1.5-flash` | `api_key: "..."` |
| **`anthropic`** | `claude-3-5-sonnet-20241022`, `claude-3-5-haiku` | `api_key: "..."` |
| **`transformers`** | `HuggingFaceTB/SmolVLM-Instruct`, `Qwen/Qwen2-VL-7B` | Local Hugging Face pipeline |
| **`mock`** | Heurístico determinista offline | Sin API keys ni descarga de modelos (Tests / CI) |

### B. Motores de Detección Visual y Acondicionamiento
* **`yolo` / `yolov8`:** Bounding boxes con modelos estándar (`yolov8n.pt`, etc.).
* **`yolo_world`:** Detección en vocabulario abierto (*Open-Vocabulary*) basada en texto de la tarea.
* **`yolo_seg`:** Segmentación por instancias y máscaras píxel a píxel.
* **Modos de acondicionamiento de observación:**
  * `crop`: Recorta la imagen alrededor de la bounding box con margen configurable y la escala a la entrada de la micro-política.
  * `mask_overlay`: Aplica la máscara atenuando el fondo para concentrar la atención.
  * `bbox_overlay`: Traza la bounding box y cruz de fijación sobre los canales de entrada.

### C. Catálogo y Mezcla de Micro-Expertos
Permite mezclar diferentes tipos de arquitectura por subtarea:
* **Subtarea 1 (`pick_mug`):** Modelo **ACT** (*Action Chunking with Transformers*).
* **Subtarea 2 (`place_mug`):** Modelo **Diffusion Policy**.
* **Subtarea 3 (`press_button`):** Micro-modelo ACT ligero.

### D. Estrategias de Conmutación en Memoria
* **`swap_on_demand` (Recomendado para bajo VRAM):** Solo 1 modelo activo en VRAM. Al terminar una subtarea, se libera la anterior (`torch.cuda.empty_cache()`) y se carga el siguiente micro-experto.
* **`coral_switching` / `lora_switching` (Opción CORAL):** Mantiene un *backbone* base congelado en GPU y conmuta dinámicamente únicamente el adaptador LoRA o cabezal de acción específico de la subtarea.
* **`cache_all` / `lazy_load`:** Mantiene los micro-modelos en memoria RAM/VRAM para transiciones ultra-rápidas si la memoria lo permite.

### E. Telemetría y Overlays en la Grabación de Video
Al activar `burn_overlays_into_recording: true`, el frame grabado contiene en tiempo real:
* Recuadro envolvente dinámico del objeto objetivo activo.
* Cruz de centrado sobre la affordance.
* Banner HUD superior:
  * Meta global (`GOAL`).
  * Subtarea activa (`PLAN [1/3]: Skill='pick_mug'`).
  * Contador de horizonte y progreso (`HORIZON: 45/85 steps (52.9%)`).
  * Estado del coordinador (`Status: EXECUTING`).

---

## 4. Instalación (usando `venv`)

> [!NOTE]
> Recuerda activar siempre tu entorno virtual (`source .venv/bin/activate`).

```bash
# 1. Crear e inicializar el entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# 2. Instalar dependencias base de LeRobot y el plugin
pip install -e /workspaces/docs_tesis/lerobot
pip install -e /workspaces/docs_tesis/submodules/lerobot-vision-detector
pip install -e /workspaces/docs_tesis/lerobot_policy_modular_vla
```

---

## 5. Ejemplo de Uso en Python

```python
import torch
from lerobot_policy_modular_vla import (
    ModularVLAConfig,
    ModularVLAPolicy,
    VLMConfig,
    DetectorConfig,
)

# 1. Configuración
config = ModularVLAConfig(
    vlm=VLMConfig(
        provider="ollama",          # o "openai", "gemini", "mock"
        model_name="qwen2.5-vl:7b",
    ),
    detector=DetectorConfig(
        engine="yolo_world",
        model_name="yolov8s-worldv2.pt",
    ),
    swapping_strategy="swap_on_demand",  # o "coral_switching"
    default_task="Prepara un café: agarra la taza y colócala en la cafetera",
    burn_overlays_into_recording=True,   # Quema overlays en la grabación
)

# 2. Instanciación del Modelo
policy = ModularVLAPolicy(config)
policy.reset()

# 3. Inferencia en Bucle Cerrado (compatible con robot.get_observation())
batch = {
    "observation.images.cam_high": torch.zeros((1, 3, 224, 224)),
    "observation.state": torch.zeros((1, 6)),
}

action = policy.select_action(batch)
print("Acción generada:", action.shape)
print("Subtarea activa:", policy.latest_info["skill_name"])
print("Frame con telemetría HUD:", policy.latest_info["visualized_frame"].shape)
```

---

## 6. Ejecución del Demo y Tests

### Ejecutar Demo del Pipeline de Café
Genera una simulación completa con 3 subtareas y guarda el video con HUD en `outputs/coffee_rollout_with_hud.mp4`:
```bash
python lerobot_policy_modular_vla/examples/coffee_pipeline_demo.py
```

### Ejecutar Suite de Tests
```bash
pytest lerobot_policy_modular_vla/tests/test_modular_vla.py
```
*(Todos los 9 tests unitarios validan cajas envolventes, detector, visualizador HUD, planificador VLM, router de habilidades, conmutación CORAL y ejecución de política de extremo a extremo).*
