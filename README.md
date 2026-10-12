# LeRobot Modular Hierarchical VLA Policy (`lerobot_policy_modular_vla`)

> **"Divided Brain" Architecture:** A modular, plug-and-play alternative to monolithic Vision-Language-Action (VLA) models within the [🤗 LeRobot](https://github.com/huggingface/lerobot) robotics ecosystem.

---

## 1. Motivation & Core Concept

Current state-of-the-art **monolithic Vision-Language-Action (VLA)** models (such as RT-2, OpenVLA, Octo, $\pi_0$, and SmolVLA) attempt to handle end-to-end perception, high-level semantic reasoning, and low-level continuous motor control inside a single dense neural network. In real-world robotic environments with accessible hardware, this presents three major bottlenecks:

1. **Severe Latency & Compute Footprint:** Executing multi-billion parameter models at 30–50 Hz requires industrial GPU clusters. Embedded systems suffer high latency and trajectory jitter.
2. **Catastrophic Forgetting:** Fine-tuning or adapting a monolithic model to a new motor skill often degrades performance on previously mastered tasks due to shared weight updates.
3. **Visual Ambiguity:** Natural language commands like *"pick the mug"* struggle in cluttered environments with multiple candidate objects and changing distractors.

### The Solution: A Modular "AI Model" in LeRobot
This repository implements the hierarchical **Divided Brain** architecture:
* **High-Level VLM Coordinator:** Interprets human goals, reasons over scene affordances, formulates a subtask execution plan, and supervises progress.
* **Perceptual Affordance Grounding (Bounding Boxes & Masks):** Fast vision models (YOLOv8, YOLO-World open-vocabulary, or instance segmentation) localize objects of interest. **Crucially, the underlying motor skills are NOT instructed with semantic object labels; they receive visual crops or masks and are trained to interact with spatial affordances.**
* **Specialized Micro-Experts:** Lightweight, focused motor policies (Action Chunking with Transformers - ACT, Diffusion Policy, etc.) executing discrete atomic motions.
* **Dynamic Model Swapping:** The system loads only the active micro-expert into GPU memory, unloading previous checkpoints to guarantee zero catastrophic forgetting and minimal VRAM usage.
* **CORAL / LoRA Head Switching Option:** Built-in extension point to keep a frozen shared backbone resident on GPU while switching only task-specific LoRA heads.
* **Live Video Recording Overlays (HUD & Telemetry):** Bounding boxes, active subtask status, step count progress, and VLM reasoning are burned directly into recorded rollout streams.

---

## 2. Plug-and-Play Compatibility with LeRobot

This package implements LeRobot's `PreTrainedPolicy` and `PreTrainedConfig` standards, and registers itself as a third-party policy plugin via LeRobot's `lerobot_policy_` convention:

```python
from lerobot.policies.factory import get_policy_class, make_policy
from lerobot_policy_modular_vla import ModularVLAConfig, ModularVLAPolicy

# Automatic discovery in LeRobot's policy registry
policy_cls = get_policy_class("modular_vla")  # or "hierarchical_vla"
```

It seamlessly integrates with standard LeRobot evaluation and rollout loops:
```bash
lerobot-eval --policy.type=modular_vla ...
lerobot-vision-eval --policy.type=modular_vla ...
```

---

## 3. Key Capabilities & Extensibility

### A. Easily Swappable VLM Providers
Switching VLM reasoning engines is as simple as updating two parameters (`provider` and `model_name`):

| Provider (`provider`) | Supported Models | Configuration / Endpoint |
| :--- | :--- | :--- |
| **`ollama`** | `qwen2.5-vl:7b`, `llava`, `minicpm-v`, `llama3.2-vision` | `api_base: "http://localhost:11434"` (Local / Edge) |
| **`openai`** | `gpt-4o`, `gpt-4o-mini`, vLLM, LiteLLM, Groq | `api_key: "sk-..."`, `api_base: "..."` |
| **`gemini`** | `gemini-2.0-flash`, `gemini-1.5-flash` | `api_key: "..."` (Google Gemini API) |
| **`anthropic`** | `claude-3-5-sonnet-20241022`, `claude-3-5-haiku` | `api_key: "..."` (Anthropic Claude API) |
| **`transformers`** | `HuggingFaceTB/SmolVLM-Instruct`, `Qwen/Qwen2-VL-7B` | Local Hugging Face pipeline |
| **`mock`** | Deterministic rule-based planner | Fully offline, zero API keys (Tests / CI) |

### B. Pluggable Vision Affordance Detectors
* **`yolo` / `yolov8`:** Standard bounding box detection (`yolov8n.pt`, etc.).
* **`yolo_world`:** Open-vocabulary object detection conditioned on natural language queries (`yolov8s-worldv2.pt`).
* **`yolo_seg`:** Instance segmentation providing pixel-accurate masks (`yolov8n-seg.pt`).
* **Observation Conditioning Modes:**
  * `crop`: Crops the camera frame around the target bounding box with configurable margin ratio and resizes to the micro-policy's input resolution.
  * `mask_overlay`: Emphasizes object pixels while dimming or filtering out background context.
  * `bbox_overlay`: Overlays bounding boxes and targeting crosshairs onto observation channels.
  * `passthrough`: Passes the full observation frame alongside bounding box coordinate tensors.

### C. Mixing Different Micro-Expert Architectures
The skills catalog allows mixing different policy types within the same task:
* **Skill 1 (`pick_mug`):** ACT (*Action Chunking with Transformers*) model trained on object crops.
* **Skill 2 (`place_mug`):** Diffusion Policy model specialized in spatial positioning.
* **Skill 3 (`press_button`):** Lightweight ACT policy for switch activation.

### D. Memory Management & Swapping Strategies
* **`swap_on_demand` (Default for Low VRAM):** Maintains only ONE policy in GPU memory. When a subtask finishes, the model is unloaded (`torch.cuda.empty_cache()`) and the next skill checkpoint is mounted from disk.
* **`coral_switching` / `lora_switching` (CORAL Option):** Keeps a shared base backbone frozen on GPU and dynamically swaps only the lightweight task-specific LoRA adapter or action head.
* **`cache_all` / `lazy_load`:** Keeps policies resident in memory for zero-latency transitions when RAM/VRAM permits.

### E. Live HUD & Bounding Box Video Recording Overlays
When `burn_overlays_into_recording: true`, the rollout stream burns telemetry directly onto output video frames:
* Highlighted green bounding box and targeting reticle for the active object affordance.
* Top HUD banner:
  * Overall goal instruction (`GOAL`).
  * Current plan step (`PLAN [1/3]: Skill='pick_mug'`).
  * Step countdown and execution horizon (`HORIZON: 45/85 steps (52.9%)`).
  * Coordinator status (`Status: EXECUTING` / `Status: VERIFYING`).

---

## 4. Installation (Using `venv`)

```bash
# 1. Clone this repository
git clone https://github.com/dcuevasa/lerobot_policy_modular_vla.git
cd lerobot_policy_modular_vla

# 2. Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install LeRobot and this package in editable mode
pip install --upgrade pip
pip install git+https://github.com/huggingface/lerobot.git
pip install -e .

# (Optional) Install vision detector companion tool for camera streaming overlays:
pip install git+https://github.com/dcuevasa/lerobot-vision-detector.git
```

---

## 5. Quickstart Example

```python
import torch
from lerobot_policy_modular_vla import (
    ModularVLAConfig,
    ModularVLAPolicy,
    VLMConfig,
    DetectorConfig,
)

# 1. Configure the modular architecture
config = ModularVLAConfig(
    vlm=VLMConfig(
        provider="ollama",          # or "openai", "gemini", "mock"
        model_name="qwen2.5-vl:7b",
    ),
    detector=DetectorConfig(
        engine="yolo_world",
        model_name="yolov8s-worldv2.pt",
    ),
    swapping_strategy="swap_on_demand",  # or "coral_switching"
    default_task="Prepare coffee: grasp the mug and place it in the coffee machine",
    burn_overlays_into_recording=True,   # Burn live HUD & bboxes onto video
)

# 2. Instantiate the policy
policy = ModularVLAPolicy(config)
policy.reset()

# 3. Step the policy with robot camera observations
batch = {
    "observation.images.cam_high": torch.zeros((1, 3, 224, 224)),
    "observation.state": torch.zeros((1, 6)),
}

action = policy.select_action(batch)
print("Action tensor shape:", action.shape)
print("Active skill:", policy.latest_info["skill_name"])
print("Target affordance:", policy.latest_info["target_object"])
```

---

## 6. LeRobot CLI Integration & Execution

The architecture is strictly integrated with LeRobot's official command-line interface. You do not need standalone evaluation or simulation scripts.

### A. Policy Evaluation via LeRobot CLI
Evaluate the modular policy using `lerobot-eval` or `lerobot-vision-eval`:
```bash
# Evaluate in simulation or on robot using a config file
lerobot-eval \
  --policy.type=modular_vla \
  --policy.path=examples/configs/coffee_task.yaml \
  --env.type=aloha \
  --eval.n_episodes=5

# Or with vision-detector overlay evaluation:
lerobot-vision-eval \
  --policy.type=modular_vla \
  --policy.path=examples/configs/coffee_task.yaml
```

### B. Live Recording & Bounding Box Overlays
When recording robot demonstrations or evaluation rollouts:
1. **Camera Stream Overlays:** With `burn_overlays_into_recording: true` (or when using `lerobot-vision-detector`), bounding boxes and the HUD are written directly into the MP4 video frames. Micro-experts trained on spatial affordances consume these frames.
2. **Recorded Demonstrations:**
```bash
lerobot-record \
  --robot.type=so100 \
  --dataset.repo_id="username/so100-modular-eval" \
  --dataset.save_video=true
```

---

## 7. Dataset Annotations & Dataset Visualizer Integration

During rollout execution, `ModularVLAPolicy` automatically records high-level plans, subtask execution intervals, and affordance bounding boxes in the official LeRobot v3.1 language atom format (`meta/lerobot_annotations.json`).

### Saving Annotations
Configure `annotations_dir` in your configuration or save programmatically:
```python
# Saves meta/lerobot_annotations.json inside the dataset directory
policy.save_annotations("path/to/dataset")
```

### Visualizing with `lerobot-dataset-visualizer`
The recorded annotations adhere to the schema required by [`lerobot-dataset-visualizer`](https://github.com/dcuevasa/lerobot-dataset-visualizer):
```bash
# Clone and run the dataset visualizer
git clone https://github.com/dcuevasa/lerobot-dataset-visualizer.git
streamlit run lerobot-dataset-visualizer/app.py -- --dataset-dir path/to/dataset
```
Features available in the visualizer:
* **Plan Timeline:** Displays high-level VLM decomposition and reasoning.
* **Subtask Spans:** Shows active micro-expert execution intervals.
* **Interactive Bounding Boxes:** Renders spatial affordance bounding boxes and labels synchronized with video frames.

### Splitting Multi-Stage Datasets with `split_dataset.py`
The generated `meta/lerobot_annotations.json` also includes legacy `subtasks` lists (`label`, `start`, `end`), making it directly compatible with [`FLAG-Embodied-data/split_dataset.py`](https://github.com/dcuevasa/FLAG-Embodied-data):
```bash
# Partition composite demonstration episodes into micro-expert subtask datasets
python path/to/FLAG-Embodied-data/split_dataset.py \
  --input-dataset path/to/dataset \
  --output-dir path/to/micro_expert_subtasks
```
This enables automatically splitting composite tasks into atomic datasets for training individual micro-expert models (ACT, Diffusion, SmolVLA).

---

## 8. Running Tests

Run the test suite using pytest in your virtual environment:
```bash
pytest tests/test_modular_vla.py
```
*(Tests verify bounding boxes, perception detectors, affordance conditioning, HUD rendering, VLM planning, model swapper, CORAL switching option, dataset annotation recording, and closed-loop policy execution).*

---

## 9. License

Apache-2.0
