#!/usr/bin/env python
"""Coffee Task End-to-End Simulation Demo for Modular Hierarchical VLA.

Demonstrates:
1. Configuring high-level VLM coordinator (pluggable: Ollama, OpenAI, Gemini, Claude, Mock).
2. Configuring visual affordance detector (YOLOv8, YOLO-World, Mock).
3. Defining and mixing specialized micro-expert skills (ACT for pick, Diffusion for place, etc.).
4. Dynamic Model Swapping (thesis low-VRAM strategy) or CORAL/LoRA head switching.
5. Executing control loop with live HUD and target bounding box overlays burned onto frames.
6. Saving an annotated playback video of the robot execution.
"""

import os
from pathlib import Path
import time
import cv2
import numpy as np
import torch

from lerobot_policy_modular_vla import (
    ModularVLAConfig,
    ModularVLAPolicy,
    VLMConfig,
    DetectorConfig,
    SkillDefinition,
)


def run_coffee_demo(
    vlm_provider: str = "mock",  # Change to "ollama", "openai", "gemini"
    vlm_model: str = "qwen2.5-vl:7b",
    detector_engine: str = "mock",  # Change to "yolo" or "yolo_world"
    swapping_strategy: str = "swap_on_demand",  # or "coral_switching", "cache_all"
    output_video_path: str = "coffee_rollout_with_hud.mp4",
):
    print("=" * 70)
    print("  MODULAR HIERARCHICAL VLA: COFFEE TASK PIPELINE DEMO")
    print("=" * 70)

    # 1. Define Skills Catalogue (Mixing ACT, Diffusion, and lightweight micro-experts)
    skills = {
        "pick_mug": {
            "name": "pick_mug",
            "description": "Grasps the mug highlighted in the bounding box/mask and lifts it",
            "policy_type": "mock",  # In real deployment: "act"
            "pretrained_path": None,  # e.g., "bendca61/so101-subtask-pick-mug"
            "target_concept": "mug",
            "max_steps": 40,
            "conditioning_mode": "crop",
            "crop_margin": 0.15,
            # Optional CORAL / LoRA switching parameter:
            "lora_adapter_path": None,
        },
        "place_mug": {
            "name": "place_mug",
            "description": "Moves mug to coffee machine base and deposits it securely",
            "policy_type": "mock",  # In real deployment: "diffusion" (mixing architectures!)
            "pretrained_path": None,  # e.g., "bendca61/so101-subtask-place-mug"
            "target_concept": "coffee_machine",
            "max_steps": 50,
            "conditioning_mode": "crop",
            "crop_margin": 0.20,
            "lora_adapter_path": None,
        },
        "press_button": {
            "name": "press_button",
            "description": "Reaches out and presses the brewing button on the coffee machine",
            "policy_type": "mock",  # In real deployment: "act"
            "pretrained_path": None,  # e.g., "bendca61/so101-subtask-press-button"
            "target_concept": "button",
            "max_steps": 25,
            "conditioning_mode": "crop",
            "crop_margin": 0.10,
            "lora_adapter_path": None,
        },
    }

    # 2. Build Policy Configuration
    config = ModularVLAConfig(
        vlm=VLMConfig(
            provider=vlm_provider,
            model_name=vlm_model,
            api_base="http://localhost:11434" if vlm_provider == "ollama" else None,
        ),
        detector=DetectorConfig(
            engine=detector_engine,
            conf_threshold=0.25,
        ),
        skills=skills,
        swapping_strategy=swapping_strategy,
        default_task="Prepara un café: agarra la taza, colócala en la cafetera y presiona el botón",
        burn_overlays_into_recording=True,  # Burn HUD & detections onto output frames!
        primary_camera_key="observation.images.cam_high",
    )

    print(f"[Config] VLM Provider      : {config.vlm.provider} (Model: {config.vlm.model_name})")
    print(f"[Config] Vision Detector   : {config.detector.engine}")
    print(f"[Config] Swapping Strategy : {config.swapping_strategy}")
    print(f"[Config] Available Skills  : {list(config.skills.keys())}")
    print(f"[Config] Goal Instruction  : '{config.default_task}'\n")

    # 3. Instantiate Policy
    policy = ModularVLAPolicy(config)
    policy.reset()

    # 4. Create Simulated Table Workspace Scene
    h, w = 480, 640
    base_canvas = np.full((h, w, 3), 40, dtype=np.uint8)
    # Draw table
    cv2.rectangle(base_canvas, (40, 80), (600, 440), (70, 75, 80), -1)
    # Draw cup (blue)
    cv2.circle(base_canvas, (180, 260), 40, (180, 100, 50), -1)
    cv2.putText(base_canvas, "CUP", (160, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # Draw coffee machine (gray)
    cv2.rectangle(base_canvas, (420, 140), (560, 360), (120, 120, 130), -1)
    cv2.putText(base_canvas, "MACHINE", (440, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # Draw button (red)
    cv2.circle(base_canvas, (490, 180), 12, (50, 50, 220), -1)
    cv2.putText(base_canvas, "BTN", (475, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # 5. Execute Simulation Rollout Loop (120 steps)
    total_steps = 120
    print(f"Starting closed-loop execution for {total_steps} steps...\n")

    recorded_frames = []
    fps = 30
    dt = 1.0 / fps

    for step in range(total_steps):
        # Convert scene to PyTorch tensor (B, C, H, W)
        tensor_img = torch.from_numpy(base_canvas).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        batch = {
            "observation.images.cam_high": tensor_img,
            "observation.state": torch.zeros((1, 6)),
        }

        # Policy Step (VLM coordinates, detector grounds, micro-expert acts, HUD renders)
        action = policy.select_action(batch)
        info = policy.latest_info

        # Retrieve burned overlay frame
        vis_frame = info.get("visualized_frame")
        if vis_frame is not None:
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
            recorded_frames.append(bgr_frame)

        if step % 15 == 0 or step == total_steps - 1:
            step_idx = info.get("step_index", 0) + 1
            total_plan = info.get("total_steps", 1)
            skill = info.get("skill_name", "N/A")
            target = info.get("target_object", "N/A")
            prog = info.get("step_progress", 0)
            max_p = info.get("max_steps", 1)
            status = info.get("status", "N/A")
            print(f"Step {step:03d} | [{step_idx}/{total_plan}] Skill='{skill:12s}' on '{target:14s}' | Progress: {prog:02d}/{max_p:02d} | Status: {status}")

    print("\nExecution complete!")

    # 6. Save Video Recording with Live Overlays
    if recorded_frames:
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        final_video_path = out_dir / Path(output_video_path).name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(final_video_path), fourcc, fps, (w, h))
        for frame in recorded_frames:
            writer.write(frame)
        writer.release()
        print(f"[Video Saved] Live overlay recording saved to: {final_video_path.resolve()}")

    return recorded_frames


if __name__ == "__main__":
    run_coffee_demo()
