import json
import logging
import os

import folder_paths

logger = logging.getLogger("RunningHub.VoxCPM")

VOXCPM_MODEL_TYPE = "voxcpm"
LORA_MODEL_TYPE = "voxcpm_lora"

folder_paths.add_model_folder_path(
    VOXCPM_MODEL_TYPE,
    os.path.join(folder_paths.models_dir, VOXCPM_MODEL_TYPE),
)
folder_paths.add_model_folder_path(
    LORA_MODEL_TYPE,
    os.path.join(folder_paths.models_dir, VOXCPM_MODEL_TYPE, "loras"),
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
    if not results:
        return ["None"]
    preferred = "VoxCPM2"
    if preferred in results:
        results.remove(preferred)
        results.insert(0, preferred)
    return results


def _list_lora_files():
    """List LoRA weight files (.pth, .ckpt) and directories."""
    base_dirs = folder_paths.get_folder_paths(LORA_MODEL_TYPE)
    seen = set()
    results = ["None"]
    for base in base_dirs:
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            if name in seen:
                continue
            full = os.path.join(base, name)
            is_weight_file = os.path.isfile(full) and name.endswith((".pth", ".ckpt", ".safetensors"))
            is_weight_dir = os.path.isdir(full) and (
                os.path.isfile(os.path.join(full, "lora_weights.ckpt"))
                or os.path.isfile(os.path.join(full, "lora_weights.safetensors"))
                or os.path.isfile(os.path.join(full, "lora_weights.pth"))
            )
            if is_weight_file or is_weight_dir:
                seen.add(name)
                results.append(name)
    return results


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


def _resolve_lora_path(lora_name):
    """Resolve LoRA name to full path. Returns None if 'None' or file missing.

    Missing LoRA files only trigger a warning instead of raising, so workflows
    that reference a LoRA name which is not present on the current machine can
    still execute (falling back to the base model without LoRA).
    """
    if not lora_name or lora_name == "None":
        return None
    base_dirs = folder_paths.get_folder_paths(LORA_MODEL_TYPE)
    for base in base_dirs:
        full = os.path.join(base, lora_name)
        if os.path.isfile(full) or os.path.isdir(full):
            return full
    logger.warning(
        "VoxCPM LoRA '%s' not found under %s; loading without LoRA.",
        lora_name,
        base_dirs,
    )
    return None


def _load_pipeline(model_name, optimize, lora_name="None"):
    from voxcpm import VoxCPM

    model_path = _resolve_model_path(model_name)
    lora_path = _resolve_lora_path(lora_name)

    logger.info("Loading VoxCPM from %s (optimize=%s, lora=%s)", model_path, optimize, lora_path)

    model = VoxCPM(
        voxcpm_model_path=model_path,
        zipenhancer_model_path=None,
        enable_denoiser=False,
        optimize=optimize,
        lora_weights_path=lora_path,
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
    return model_info


class RunningHubVoxCPMLoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_list_model_dirs(),),
                "optimize": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "lora_name": (_list_lora_files(),),
            },
        }

    RETURN_TYPES = ("VOXCPM_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "RunningHub/VoxCPM"

    @classmethod
    def VALIDATE_INPUTS(cls, lora_name="None", **_kwargs):
        # Only declare lora_name so ComfyUI skips its enum validation for it
        # (model_name keeps the default validation). Workflows may reference a
        # LoRA file that is missing on this machine; we warn at load time and
        # fall back to no-LoRA instead of blocking submission.
        return True

    def load_model(self, model_name, optimize, lora_name="None"):
        model_info = _load_pipeline(model_name, optimize, lora_name)
        return (model_info,)
