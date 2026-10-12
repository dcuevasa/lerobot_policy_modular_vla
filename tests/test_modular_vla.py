import json
from pathlib import Path
import numpy as np
import pytest
import torch

from lerobot_policy_modular_vla import (
    AffordanceConditioner,
    BoundingBox,
    AffordanceDetection,
    DatasetAnnotationRecorder,
    DetectorConfig,
    MockAffordanceDetector,
    MockSkillPolicy,
    MockVLMClient,
    ModelSwapper,
    ModularVLAConfig,
    ModularVLAPolicy,
    PlanCoordinator,
    SkillCatalog,
    SkillDefinition,
    SkillRouter,
    TelemetryVisualizer,
    VLMConfig,
    make_detector,
    make_vlm_client,
)


def test_bounding_box_and_padding():
    box = BoundingBox(ymin=0.2, xmin=0.3, ymax=0.6, xmax=0.7, normalized=True)
    assert box.width == pytest.approx(0.4)
    assert box.height == pytest.approx(0.4)
    assert box.center == (pytest.approx(0.4), pytest.approx(0.5))

    pbox = box.to_pixel(img_height=100, img_width=200)
    assert pbox.ymin == pytest.approx(20.0)
    assert pbox.xmin == pytest.approx(60.0)
    assert pbox.ymax == pytest.approx(60.0)
    assert pbox.xmax == pytest.approx(140.0)

    padded = box.pad(margin_ratio=0.1)
    assert padded.ymin < box.ymin
    assert padded.ymax > box.ymax


def test_mock_detector():
    detector = MockAffordanceDetector()
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = detector.detect(dummy_img, candidate_labels=["cup", "coffee_machine"])
    assert len(dets) >= 2
    labels = [d.label for d in dets]
    assert "cup" in labels
    assert "coffee_machine" in labels
    assert dets[0].mask is not None


def test_affordance_conditioner():
    conditioner = AffordanceConditioner(default_mode="crop", crop_margin=0.1, target_size=(100, 100))
    img_t = torch.rand((1, 3, 200, 200))
    det = AffordanceDetection(
        id=0,
        label="cup",
        box=BoundingBox(ymin=0.2, xmin=0.2, ymax=0.6, xmax=0.6, normalized=True),
    )

    # 1. Crop mode
    cropped = conditioner.condition_tensor(img_t, det, mode="crop")
    assert cropped.shape == (1, 3, 100, 100)

    # 2. BBox overlay mode
    overlay = conditioner.condition_tensor(img_t, det, mode="bbox_overlay")
    assert overlay.shape == img_t.shape

    # 3. Batch conditioning
    batch = {"observation.images.cam_high": img_t, "state": torch.zeros((1, 6))}
    cond_batch = conditioner.condition_observation_batch(batch, det, target_camera_keys=["observation.images.cam_high"])
    assert cond_batch["observation.images.cam_high"].shape == (1, 3, 100, 100)
    assert "target_affordance_bbox" in cond_batch


def test_telemetry_visualizer():
    visualizer = TelemetryVisualizer(enabled=True)
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    det = AffordanceDetection(
        id=0,
        label="mug",
        box=BoundingBox(ymin=0.3, xmin=0.3, ymax=0.7, xmax=0.7, normalized=True),
    )
    rendered = visualizer.render_overlay(
        image=img,
        active_detection=det,
        all_detections=[det],
        task="Test Task",
        active_skill="pick",
        current_step_idx=0,
        total_steps=3,
        skill_step_count=10,
        max_skill_steps=100,
        status="EXECUTING",
    )
    assert rendered.shape == (240, 320, 3)
    assert rendered.dtype == np.uint8


def test_mock_vlm_client_planning():
    client = MockVLMClient()
    dets = [
        AffordanceDetection(id=0, label="cup", box=BoundingBox(0.2, 0.2, 0.5, 0.5)),
        AffordanceDetection(id=1, label="coffee_machine", box=BoundingBox(0.3, 0.6, 0.8, 0.9)),
    ]
    skills = [
        {"name": "pick_mug", "description": "pick mug", "max_steps": 80},
        {"name": "place_mug", "description": "place mug", "max_steps": 100},
    ]
    plan = client.generate_plan(task="prepare coffee", detected_objects=dets, available_skills=skills)
    assert len(plan.steps) >= 2
    assert plan.steps[0].skill_name == "pick_mug"
    assert plan.steps[1].skill_name == "place_mug"


def test_vlm_factory_providers():
    mock_c = make_vlm_client("mock")
    assert isinstance(mock_c, MockVLMClient)

    ollama_c = make_vlm_client("ollama", model_name="qwen2.5-vl:7b", api_base="http://localhost:11434")
    assert ollama_c.model_name == "qwen2.5-vl:7b"

    openai_c = make_vlm_client("openai", model_name="gpt-4o", api_key="sk-test")
    assert openai_c.model_name == "gpt-4o"

    gemini_c = make_vlm_client("gemini", model_name="gemini-2.0-flash", api_key="test-key")
    assert gemini_c.model_name == "gemini-2.0-flash"


def test_model_swapper_and_router():
    skills = [
        SkillDefinition(name="skill_a", description="Skill A", policy_type="mock", max_steps=2),
        SkillDefinition(name="skill_b", description="Skill B", policy_type="mock", max_steps=3),
    ]
    catalog = SkillCatalog(skills)
    swapper = ModelSwapper(strategy="swap_on_demand", default_action_dim=6)
    router = SkillRouter(catalog, swapper)

    router.set_active_skill("skill_a")
    assert router.active_skill_name == "skill_a"
    assert not router.is_current_skill_finished()

    batch = {"observation.images.cam_high": torch.zeros((1, 3, 100, 100))}
    action1 = router.select_action(batch)
    assert action1.shape == (1, 6)
    action2 = router.select_action(batch)
    assert router.is_current_skill_finished()

    # Switch to skill_b
    router.set_active_skill("skill_b")
    assert router.active_skill_name == "skill_b"
    assert router.skill_step_count == 0


def test_coral_switching_option():
    skill = SkillDefinition(
        name="coral_skill",
        description="Skill using LoRA head",
        policy_type="mock",
        lora_adapter_path="path/to/adapter",
    )
    swapper = ModelSwapper(strategy="coral_switching", default_action_dim=6)
    policy = swapper.get_or_load(skill)
    assert policy is not None
    assert swapper._active_skill_name == "coral_skill"


def test_modular_vla_policy_end_to_end():
    config = ModularVLAConfig(
        vlm=VLMConfig(provider="mock"),
        detector=DetectorConfig(engine="mock"),
        default_task="Prepare coffee with cup and coffee machine",
        burn_overlays_into_recording=True,
    )
    # Shorten steps for quick test
    for s in config.skills.values():
        s["max_steps"] = 3

    policy = ModularVLAPolicy(config)
    policy.reset()

    batch = {
        "observation.images.cam_high": torch.zeros((1, 3, 224, 224)),
        "observation.state": torch.zeros((1, 6)),
    }

    # Execute several steps to verify execution and subtask transitions
    for step in range(10):
        action = policy.select_action(batch)
        assert action.shape == (1, 6)
        assert policy.latest_info["status"] in ("EXECUTING", "COMPLETED")
        assert "visualized_frame" in policy.latest_info
        assert policy.latest_info["visualized_frame"].shape == (224, 224, 3)

    text = policy.generate_text(batch)
    assert "Plan:" in text or len(text) > 0


def test_dataset_annotation_recorder(tmp_path: Path):
    from lerobot_policy_modular_vla.vlm.base import PlanResult, PlanStep

    recorder = DatasetAnnotationRecorder(fps=30.0)
    recorder.start_episode(episode_index=0)

    # 1. Record plan
    plan = PlanResult(
        task="prepare coffee",
        steps=[
            PlanStep(step_index=0, skill_name="pick_mug", target_object="cup", max_steps=10),
            PlanStep(step_index=1, skill_name="place_mug", target_object="coffee_machine", max_steps=10),
        ],
        reasoning="Pick then place",
    )
    recorder.record_plan(plan, timestamp=0.0)

    # 2. Record subtask 1 and detection
    recorder.record_subtask_start(plan.steps[0], timestamp=0.0)
    det = AffordanceDetection(
        id=1,
        label="cup",
        box=BoundingBox(ymin=0.2, xmin=0.3, ymax=0.5, xmax=0.6, normalized=True),
        score=0.95,
    )
    recorder.record_affordance_detection(det, camera_key="observation.images.cam_high", timestamp=0.0)

    # Step forward
    for _ in range(15):
        recorder.step()

    # 3. Record subtask 2
    recorder.record_subtask_start(plan.steps[1], timestamp=0.5)
    for _ in range(15):
        recorder.step()

    # 4. Save and inspect JSON
    saved_file = recorder.save_to_json(tmp_path)
    assert saved_file.exists()
    assert saved_file.name == "lerobot_annotations.json"

    with open(saved_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["version"] == 2
    assert "schema" in data
    assert "0" in data["episodes"]
    ep0 = data["episodes"]["0"]

    atoms = ep0["atoms"]
    assert len(atoms) >= 3
    styles = [a["style"] for a in atoms]
    assert "plan" in styles
    assert "subtask" in styles
    assert "vqa" in styles

    # Check vqa bounding box tool call
    vqa_atoms = [a for a in atoms if a["style"] == "vqa"]
    assert len(vqa_atoms) == 1
    tool_call = vqa_atoms[0]["tool_calls"][0]
    assert tool_call["type"] == "bbox"
    assert tool_call["label"] == "cup"
    assert len(tool_call["box"]) == 4

    # Check legacy subtasks for split_dataset.py
    assert "subtasks" in ep0
    subtasks = ep0["subtasks"]
    assert len(subtasks) == 2
    assert subtasks[0]["label"] == "pick_mug"
    assert subtasks[1]["label"] == "place_mug"
    assert subtasks[0]["start"] <= subtasks[0]["end"]


def test_modular_vla_policy_annotations_e2e(tmp_path: Path):
    config = ModularVLAConfig(
        vlm=VLMConfig(provider="mock"),
        detector=DetectorConfig(engine="mock"),
        default_task="Prepare coffee",
        annotations_dir=str(tmp_path),
    )
    for s in config.skills.values():
        s["max_steps"] = 2

    policy = ModularVLAPolicy(config)
    policy.reset(episode_index=0)

    batch = {
        "observation.images.cam_high": torch.zeros((1, 3, 224, 224)),
        "observation.state": torch.zeros((1, 6)),
    }

    for _ in range(6):
        policy.select_action(batch)

    # Save annotations
    saved_path = policy.save_annotations()
    assert saved_path.exists()

    with open(saved_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "0" in data["episodes"]
    atoms = data["episodes"]["0"]["atoms"]
    assert any(a["style"] == "plan" for a in atoms)
    assert any(a["style"] == "subtask" for a in atoms)

