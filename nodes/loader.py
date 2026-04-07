import os
import sys
import json
import torch
import folder_paths

VOXCPM_MODEL_TYPE = "voxcpm"
folder_paths.add_model_folder_path(
    VOXCPM_MODEL_TYPE,
    os.path.join(folder_paths.models_dir, VOXCPM_MODEL_TYPE),
)


def _list_model_dirs():
    """List VoxCPM model directories that contain config.json."""
    base_dirs = folder_paths.get_folder_paths(VOXCPM_MODEL_TYPE)
    seen = set()
    results = []
    for base in base_dirs:
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            if name in seen:
                continue
            full = os.path.join(base, name)
            if os.path.isdir(full) and os.path.isfile(os.path.join(full, "config.json")):
                seen.add(name)
                results.append(name)
    return results if results else ["None"]


def _resolve_model_path(model_name):
    """Resolve model name to full path."""
    base_dirs = folder_paths.get_folder_paths(VOXCPM_MODEL_TYPE)
    for base in base_dirs:
        full = os.path.join(base, model_name)
        if os.path.isdir(full) and os.path.isfile(os.path.join(full, "config.json")):
            return full
    raise FileNotFoundError(
        f"VoxCPM model '{model_name}' not found. "
        f"Please place model directory under: {base_dirs}"
    )


class VoxCPMLoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_list_model_dirs(),),
                "optimize": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("VOXCPM_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "audio/voxcpm"

    def load_model(self, model_name, optimize):
        from voxcpm import VoxCPM

        model_path = _resolve_model_path(model_name)

        model = VoxCPM(
            voxcpm_model_path=model_path,
            zipenhancer_model_path=None,
            enable_denoiser=False,
            optimize=optimize,
        )

        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        arch = config.get("architecture", "voxcpm").lower()

        model_info = {
            "model": model,
            "sample_rate": model.tts_model.sample_rate,
            "architecture": arch,
            "model_path": model_path,
        }

        return (model_info,)
