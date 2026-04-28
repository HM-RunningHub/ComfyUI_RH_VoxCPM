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


def _infer_rank_from_safetensors(weights_path):
    """Reverse-engineer the LoRA rank from raw tensor shapes.

    LoRA stores each adapted linear layer as a pair of low-rank
    factors. Across the two voxcpm code paths the names look like
    ``...lora_A.weight`` / ``...lora_B.weight`` (or ``lora_a`` /
    ``lora_b``); shapes are ``[r, in]`` and ``[out, r]`` respectively.
    Both contain ``r`` as their *smaller* dimension, so we just take
    ``min(shape)`` of any candidate tensor and majority-vote across
    the file. This survives misnamed / mixed-case keys and never
    needs the upstream class hierarchy.

    Returns the inferred ``r`` (int >= 1) or ``None`` if the file
    cannot be parsed or contains no recognisable LoRA tensors.
    """
    if not weights_path or not os.path.isfile(weights_path):
        return None
    if not weights_path.lower().endswith(".safetensors"):
        return None
    try:
        from safetensors import safe_open
    except ImportError:
        return None
    counts = {}
    try:
        with safe_open(weights_path, framework="pt") as f:
            for key in f.keys():
                low = key.lower()
                if "lora" not in low:
                    continue
                if not (low.endswith(".weight") or low.endswith("weight")):
                    continue
                tensor = f.get_tensor(key)
                shape = tuple(tensor.shape)
                if len(shape) < 2:
                    continue
                r = int(min(shape))
                if r <= 0:
                    continue
                counts[r] = counts.get(r, 0) + 1
    except Exception as e:
        logger.warning("Failed to inspect safetensors shapes from %s: %s", weights_path, e)
        return None
    if not counts:
        return None
    # Majority vote: pick the most common r across all LoRA tensors.
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _infer_rank_from_ckpt(weights_path):
    """Same shape-vote trick as above, for legacy ``.ckpt`` / ``.pth`` files."""
    if not weights_path or not os.path.isfile(weights_path):
        return None
    ext = os.path.splitext(weights_path)[1].lower()
    if ext not in (".ckpt", ".pth"):
        return None
    try:
        import torch
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception:
        return None
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else None
    if not isinstance(state, dict):
        return None
    counts = {}
    for key, tensor in state.items():
        low = str(key).lower()
        if "lora" not in low:
            continue
        if not hasattr(tensor, "shape"):
            continue
        shape = tuple(tensor.shape)
        if len(shape) < 2:
            continue
        r = int(min(shape))
        if r <= 0:
            continue
        counts[r] = counts.get(r, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _read_safetensors_lora_config(weights_path):
    """Read embedded LoRAConfig metadata from a ``.safetensors`` file.

    The plugin's Train LoRA node writes ``lora_config`` (rank/alpha/...)
    into the safetensors ``__metadata__`` block, so a freshly-uploaded
    single-file LoRA can still self-describe its rank even when the
    sidecar ``lora_config.json`` was lost in transit (a common pain point
    on RunningHub since the upload widget only accepts one file).

    Returns the parsed config dict, or ``None`` when the file is not a
    safetensors archive, has no metadata, or doesn't contain a JSON blob
    under the ``lora_config`` key.
    """
    if not weights_path or not os.path.isfile(weights_path):
        return None
    if not weights_path.lower().endswith(".safetensors"):
        return None
    try:
        from safetensors import safe_open
    except ImportError:
        return None
    try:
        with safe_open(weights_path, framework="pt") as f:
            md = f.metadata() or {}
    except Exception as e:
        logger.warning("Failed to read safetensors metadata from %s: %s", weights_path, e)
        return None
    blob = md.get("lora_config")
    if not blob:
        return None
    try:
        cfg = json.loads(blob)
    except Exception as e:
        logger.warning("safetensors lora_config metadata is not valid JSON (%s): %s", weights_path, e)
        return None
    if isinstance(cfg, dict) and ("r" in cfg or "alpha" in cfg):
        return cfg
    return None


def _read_ckpt_lora_config(weights_path):
    """Read embedded ``lora_config`` from a torch ``.ckpt`` / ``.pth`` file.

    The Train LoRA node also writes ``{"state_dict": ..., "lora_config": ...}``
    when safetensors is unavailable. We load with ``weights_only=True`` so
    the read is hardened against malicious pickle payloads (in line with
    upstream commit ``ec2acec``).
    """
    if not weights_path or not os.path.isfile(weights_path):
        return None
    ext = os.path.splitext(weights_path)[1].lower()
    if ext not in (".ckpt", ".pth"):
        return None
    try:
        import torch
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception as e:
        logger.warning("Failed to read torch checkpoint metadata from %s: %s", weights_path, e)
        return None
    if not isinstance(ckpt, dict):
        return None
    cfg = ckpt.get("lora_config")
    if isinstance(cfg, dict) and ("r" in cfg or "alpha" in cfg):
        return cfg
    return None


def _read_lora_config(lora_path):
    """Locate the LoRA rank/alpha metadata for ``lora_path``.

    Resolution order (first hit wins):

    1. ``lora_path`` is a *directory* containing ``lora_config.json`` (the
       layout produced by this plugin's Train LoRA node and by the
       upstream training script).
    2. ``lora_path`` is a *file* — first try the embedded metadata inside
       the file itself (``safetensors.__metadata__`` for ``.safetensors``,
       ``ckpt['lora_config']`` for ``.ckpt`` / ``.pth``). This survives a
       single-file re-upload through UIs (such as RunningHub's) that
       cannot carry a sidecar JSON.
    3. ``lora_path`` is a *file* — fall back to a sidecar JSON in the
       same directory (``<stem>.json`` or ``lora_config.json``).

    Returns the parsed ``lora_config`` dict (subset accepted by
    ``LoRAConfig``) or ``None`` when no usable config is found.

    Aligned with upstream commit ``19b6bf7`` (lora_ft_webui rank-mismatch
    fix): if we don't pass the correct ``r/alpha`` to VoxCPM, the auto
    LoRAConfig defaults to ``r=8, alpha=16`` and silently skips every
    LoRA parameter trained at e.g. ``r=32``.
    """
    if not lora_path:
        return None

    json_candidates = []
    weight_files_for_shape_probe = []
    if os.path.isdir(lora_path):
        json_candidates.append(os.path.join(lora_path, "lora_config.json"))
        for fn in ("lora_weights.safetensors", "lora_weights.ckpt", "lora_weights.pth"):
            full = os.path.join(lora_path, fn)
            if os.path.isfile(full):
                weight_files_for_shape_probe.append(full)
    elif os.path.isfile(lora_path):
        embedded = _read_safetensors_lora_config(lora_path) or _read_ckpt_lora_config(lora_path)
        if embedded is not None:
            return embedded
        d = os.path.dirname(lora_path)
        stem = os.path.splitext(os.path.basename(lora_path))[0]
        json_candidates.append(os.path.join(d, f"{stem}.json"))
        json_candidates.append(os.path.join(d, "lora_config.json"))
        weight_files_for_shape_probe.append(lora_path)

    for cand in json_candidates:
        if not os.path.isfile(cand):
            continue
        try:
            with open(cand, "r", encoding="utf-8") as f:
                data = json.load(f)
            cfg = data.get("lora_config", data)
            if isinstance(cfg, dict) and ("r" in cfg or "alpha" in cfg):
                return cfg
        except Exception as e:
            logger.warning("Failed to read LoRA config %s: %s", cand, e)

    # Last-resort: probe the weight tensors directly to discover r.
    # This is what saves the user when a "naked" .safetensors is uploaded
    # through a single-file UI and no metadata/sidecar made it across.
    for wf in weight_files_for_shape_probe:
        inferred = _infer_rank_from_safetensors(wf) or _infer_rank_from_ckpt(wf)
        if inferred is not None:
            logger.info(
                "Auto-inferred LoRA rank=%d (alpha defaulted to %d) "
                "from tensor shapes in %s — no metadata/sidecar found.",
                inferred, inferred, wf,
            )
            # alpha is a *scaling* hyperparameter and is not encoded in
            # the saved tensors. Convention in this plugin (and most
            # community LoRAs) is alpha == r when the user didn't pick
            # something else. This restores 1.0× scaling and avoids the
            # silent dimension-skip caused by VoxCPM's default r=8 path.
            return {"r": inferred, "alpha": inferred}
    return None


def _build_lora_config_for_model(model_path, lora_cfg_dict):
    """Instantiate the right ``LoRAConfig`` (V1 or V2) for the base model.

    Honors the rank / alpha / target switches recorded in the saved
    ``lora_config.json`` so VoxCPM materializes LoRA tensors with shapes
    matching the checkpoint we are about to load.
    """
    config_path = os.path.join(model_path, "config.json")
    arch = "voxcpm"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            arch = (json.load(f).get("architecture") or "voxcpm").lower()
    except Exception:
        pass

    if arch == "voxcpm2":
        from voxcpm.model.voxcpm2 import LoRAConfig
    else:
        from voxcpm.model.voxcpm import LoRAConfig

    kwargs = {
        "enable_lm": bool(lora_cfg_dict.get("enable_lm", True)),
        "enable_dit": bool(lora_cfg_dict.get("enable_dit", True)),
        "enable_proj": bool(lora_cfg_dict.get("enable_proj", False)),
    }
    if "r" in lora_cfg_dict:
        kwargs["r"] = int(lora_cfg_dict["r"])
    if "alpha" in lora_cfg_dict:
        kwargs["alpha"] = int(lora_cfg_dict["alpha"])
    if "dropout" in lora_cfg_dict:
        kwargs["dropout"] = float(lora_cfg_dict["dropout"])
    return LoRAConfig(**kwargs)


def _load_pipeline(model_name, optimize, lora_name="None"):
    from voxcpm import VoxCPM

    model_path = _resolve_model_path(model_name)
    lora_path = _resolve_lora_path(lora_name)

    lora_config = None
    if lora_path is not None:
        cfg_dict = _read_lora_config(lora_path)
        if cfg_dict:
            try:
                lora_config = _build_lora_config_for_model(model_path, cfg_dict)
                logger.info(
                    "Using LoRA rank=%s alpha=%s for %s",
                    cfg_dict.get("r"), cfg_dict.get("alpha"), lora_path,
                )
            except Exception as e:
                logger.warning(
                    "Failed to build LoRAConfig (%s); falling back to "
                    "VoxCPM auto config (r=8, alpha=16). Rank mismatch may "
                    "cause LoRA params to be skipped.",
                    e,
                )
        else:
            # _read_lora_config already tries to infer r from tensor shapes,
            # so reaching here means the file genuinely has no LoRA tensors
            # (or something unreadable). Just let VoxCPM run its default.
            logger.warning(
                "LoRA '%s' has no metadata/sidecar AND no detectable LoRA "
                "tensors. Loading with VoxCPM defaults (r=8, alpha=16); the "
                "weights may be silently skipped if they don't match.",
                lora_path,
            )

    logger.info("Loading VoxCPM from %s (optimize=%s, lora=%s)", model_path, optimize, lora_path)

    model = VoxCPM(
        voxcpm_model_path=model_path,
        zipenhancer_model_path=None,
        enable_denoiser=False,
        optimize=optimize,
        lora_config=lora_config,
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
