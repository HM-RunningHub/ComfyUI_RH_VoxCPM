"""
LoRA / Full fine-tuning nodes for VoxCPM, wrapping the training code that
ships with the upstream `VoxCPM` project (see scripts/train_voxcpm_finetune.py).

These nodes execute the training loop **in-process** so that users can drive
a complete LoRA fine-tune from a ComfyUI workflow. For large-scale training
it is still recommended to run the upstream script directly, but small voice
adaptation jobs (a few hundred iterations on a few minutes of audio) can be
done entirely inside ComfyUI.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchaudio
import folder_paths
import comfy.utils

from .generate import _recognize_audio, _safe_save_wav

logger = logging.getLogger("RunningHub.VoxCPM.Train")

VOXCPM_MODEL_TYPE = "voxcpm"
LORA_MODEL_TYPE = "voxcpm_lora"


def _resolve_model_path(model_name: str) -> str:
    base_dirs = folder_paths.get_folder_paths(VOXCPM_MODEL_TYPE)
    for base in base_dirs:
        full = os.path.join(base, model_name)
        if os.path.isdir(full) and os.path.isfile(os.path.join(full, "config.json")):
            return full
    raise FileNotFoundError(
        f"VoxCPM model '{model_name}' not found. "
        f"Please place model directory under: {base_dirs}"
    )


def _list_model_dirs() -> List[str]:
    base_dirs = folder_paths.get_folder_paths(VOXCPM_MODEL_TYPE)
    seen = set()
    results: List[str] = []
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


def _get_output_root() -> str:
    """Return the root directory for training outputs under ComfyUI output dir."""
    root = os.path.join(folder_paths.get_output_directory(), "voxcpm_train")
    os.makedirs(root, exist_ok=True)
    return root


def _get_lora_dir() -> str:
    """Return first LoRA directory under models/voxcpm/loras (create if missing)."""
    base_dirs = folder_paths.get_folder_paths(LORA_MODEL_TYPE)
    target = base_dirs[0] if base_dirs else os.path.join(
        folder_paths.models_dir, VOXCPM_MODEL_TYPE, "loras"
    )
    os.makedirs(target, exist_ok=True)
    return target


def _zip_checkpoint_to_output(source_dir: Path, zip_name: str) -> Optional[dict]:
    """Zip ``source_dir`` under ComfyUI's ``output/`` for download.

    Follows the RH onboarding convention (`glut-shared-rh-onboard`, §8
    "Output Node Standardization"): the destination path is resolved via
    :func:`folder_paths.get_save_image_path`, which handles:

    * Anti-traversal of ``output/``
    * Automatic ``_00001_`` style counter suffix so concurrent training
      tasks don't overwrite each other
    * RunningHub-side filename rewriting (the caller should surface the
      resulting ``filename``/``subfolder`` pair so downstream nodes and
      the web UI can locate the artifact)

    The archive is written to ``output/voxcpm_train/<zip_name>_NNNNN_.zip``
    (i.e. the ``voxcpm_train`` subfolder is kept isolated from plain image
    outputs). On failure the function never raises; it just returns ``None``.

    Returns
    -------
    dict or None
        ``{"zip_path": Path, "filename": str, "subfolder": str}`` on
        success, ``None`` otherwise.
    """
    import shutil

    try:
        safe_prefix = zip_name.strip().replace(" ", "_") or "voxcpm_lora"
        # RH convention: treat zip like an image output so RunningHub's
        # filename/subfolder rewriter can reach it. The subfolder is
        # deliberately scoped to the plugin so we never pollute the root
        # of `output/` alongside generated images.
        prefix_with_subfolder = f"voxcpm_train/{safe_prefix}"

        full_output_folder, filename, counter, subfolder, _ = (
            folder_paths.get_save_image_path(
                prefix_with_subfolder,
                folder_paths.get_output_directory(),
            )
        )
        os.makedirs(full_output_folder, exist_ok=True)

        archive_basename = f"{filename}_{counter:05d}_"
        archive_base_path = os.path.join(full_output_folder, archive_basename)
        archive = shutil.make_archive(
            archive_base_path, "zip", root_dir=str(source_dir)
        )

        final_filename = os.path.basename(archive)
        return {
            "zip_path": Path(archive),
            "filename": final_filename,
            "subfolder": subfolder,
        }
    except Exception as e:
        logger.warning("Failed to zip training output to ComfyUI output dir: %s", e)
        return None


def _detect_sample_rate(pretrained_path: str) -> Optional[int]:
    """Read audio_vae_config.sample_rate from config.json; None if unavailable."""
    cfg_file = os.path.join(pretrained_path, "config.json")
    if not os.path.isfile(cfg_file):
        return None
    try:
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return int(cfg["audio_vae_config"]["sample_rate"])
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        logger.warning("Failed to detect sample_rate from %s: %s", cfg_file, e)
        return None


def _detect_out_sample_rate(pretrained_path: str) -> int:
    cfg_file = os.path.join(pretrained_path, "config.json")
    if not os.path.isfile(cfg_file):
        return 0
    try:
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return int(cfg.get("audio_vae_config", {}).get("out_sample_rate") or 0)
    except Exception:
        return 0


def _detect_architecture(pretrained_path: str) -> str:
    cfg_file = os.path.join(pretrained_path, "config.json")
    if not os.path.isfile(cfg_file):
        return "voxcpm"
    try:
        with open(cfg_file, "r", encoding="utf-8") as f:
            return json.load(f).get("architecture", "voxcpm").lower()
    except Exception:
        return "voxcpm"


def _ensure_voxcpm_src_importable():
    """
    The upstream training modules live under `voxcpm.training.*`. The VoxCPM
    PyPI package ships the inference side only; make sure training utilities
    are importable by trying both package and local-source paths.
    """
    try:
        import voxcpm.training  # noqa: F401
        return
    except ImportError:
        pass

    # Try local sibling checkouts of the upstream repo.
    candidates = []
    plugin_root = Path(__file__).resolve().parent.parent
    # Allow a local vendored copy under the plugin directory.
    candidates.append(plugin_root / "voxcpm" / "src")
    # ComfyUI/custom_nodes/VoxCPM/src (common side-by-side layout)
    candidates.append(plugin_root.parent / "VoxCPM" / "src")

    for cand in candidates:
        if (cand / "voxcpm" / "training").is_dir():
            import sys

            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            try:
                import voxcpm.training  # noqa: F401
                logger.info("Loaded voxcpm training utilities from %s", cand)
                return
            except ImportError:
                continue

    raise ImportError(
        "voxcpm.training module not found. The training nodes require the "
        "upstream VoxCPM source tree. Install the full repo, or place a "
        "checkout next to this plugin (e.g. "
        "ComfyUI/custom_nodes/VoxCPM/src/voxcpm/training)."
    )


def _audio_dict_to_wav(audio_dict: dict, target_sr: int) -> "torch.Tensor":
    """Convert ComfyUI AUDIO dict to mono float32 1-D tensor at target_sr."""
    waveform = audio_dict["waveform"]
    sr = int(audio_dict["sample_rate"])
    if waveform.dim() == 3:
        waveform = waveform.squeeze(0)
    if waveform.dim() == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.squeeze(0).cpu().float()


# --------------------------------------------------------------------------- #
# Dataset nodes
# --------------------------------------------------------------------------- #

def _asr_audio_dict(audio_dict: dict) -> str:
    """Run funasr SenseVoiceSmall on a ComfyUI AUDIO dict, return the text.

    The reference audio is resampled to 16kHz mono and written to a temp
    WAV, then handed to :func:`generate._recognize_audio`. The caller is
    responsible for raising if the result is empty.
    """
    wav = _audio_dict_to_wav(audio_dict, 16000)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    try:
        _safe_save_wav(tmp.name, wav.unsqueeze(0), 16000)
        return (_recognize_audio(tmp.name) or "").strip()
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


class RunningHubVoxCPMDatasetEntry:
    """Create a single (audio, text) training entry to feed into Dataset Build.

    If ``text`` is left blank, the node runs funasr SenseVoiceSmall on
    ``audio`` to obtain the transcript automatically — useful when batching
    a folder of clips without pre-written labels.

    Optionally attach a ``ref_audio`` sample that VoxCPM's training pipeline
    will use as the reference utterance for voice-style conditioning (upstream
    commit ``e4e0496``).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "dataset_id": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
                "ref_audio": ("AUDIO",),
            },
            "optional_inputs": {},
        }

    RETURN_TYPES = ("VOXCPM_DATA_ENTRY", "STRING")
    RETURN_NAMES = ("entry", "text")
    FUNCTION = "build"
    CATEGORY = "RunningHub/VoxCPM/Train"

    # Minimum characters for an ASR result to be considered usable. Anything
    # shorter (e.g. ``그.``, ``.``) almost certainly means SenseVoiceSmall
    # failed to transcribe the clip and we must not silently feed the
    # training loop garbage.
    _MIN_ASR_TEXT_LEN = 2

    def build(self, audio, text="", dataset_id=0, ref_audio=None):
        text = (text or "").strip()
        if not text:
            logger.info("Dataset entry text is empty, running funasr ASR on audio...")
            text = _asr_audio_dict(audio)
            logger.info("ASR transcript: %s", text[:120] + ("..." if len(text) > 120 else ""))
            if not text:
                raise ValueError(
                    "Dataset entry text was empty and automatic ASR returned no "
                    "transcript. Please provide the text manually."
                )
            if len(text) < self._MIN_ASR_TEXT_LEN:
                raise ValueError(
                    f"Automatic ASR returned a suspiciously short transcript "
                    f"({text!r}). This usually means the clip is silent, the "
                    f"language is not supported, or the SenseVoiceSmall model "
                    f"is missing. Please provide the text manually."
                )
        entry = {"audio": audio, "text": text, "dataset_id": int(dataset_id)}
        if ref_audio is not None:
            entry["ref_audio"] = ref_audio
        return (entry, text)


class RunningHubVoxCPMDatasetBuild:
    """Materialize (audio, text) entries into a jsonl manifest on disk.

    Accepts entries via dynamic `entry_1..entry_N` slots, plus an optional
    extra manifest path to append user-curated data.
    """

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "entry_1": ("VOXCPM_DATA_ENTRY",),
        }
        optional = {
            "entry_2": ("VOXCPM_DATA_ENTRY",),
            "entry_3": ("VOXCPM_DATA_ENTRY",),
            "entry_4": ("VOXCPM_DATA_ENTRY",),
            "entry_5": ("VOXCPM_DATA_ENTRY",),
            "entry_6": ("VOXCPM_DATA_ENTRY",),
            "entry_7": ("VOXCPM_DATA_ENTRY",),
            "entry_8": ("VOXCPM_DATA_ENTRY",),
            "extra_manifest": ("STRING", {"default": "", "multiline": False}),
            "sample_rate": ("INT", {
                "default": 16000,
                "min": 8000,
                "max": 48000,
                "step": 1000,
            }),
            "dataset_name": ("STRING", {"default": "voxcpm_dataset"}),
        }
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("manifest_path", "num_samples")
    FUNCTION = "build"
    CATEGORY = "RunningHub/VoxCPM/Train"

    def build(self, entry_1, **kwargs):
        sample_rate = int(kwargs.get("sample_rate", 16000))
        dataset_name = (kwargs.get("dataset_name") or "voxcpm_dataset").strip()

        entries: List[dict] = [entry_1]
        for i in range(2, 9):
            e = kwargs.get(f"entry_{i}")
            if e is not None:
                entries.append(e)

        ts = time.strftime("%Y%m%d_%H%M%S")
        out_root = Path(_get_output_root()) / f"{dataset_name}_{ts}"
        wav_dir = out_root / "wavs"
        wav_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = out_root / "train.jsonl"
        num = 0
        with manifest_path.open("w", encoding="utf-8") as f:
            for idx, ent in enumerate(entries):
                audio = ent["audio"]
                text = ent["text"]
                dataset_id = int(ent.get("dataset_id", 0))
                wav = _audio_dict_to_wav(audio, sample_rate)
                wav_file = wav_dir / f"sample_{idx:05d}.wav"
                _safe_save_wav(str(wav_file), wav.unsqueeze(0), sample_rate)
                duration = float(wav.shape[-1]) / float(sample_rate)
                record = {
                    "audio": str(wav_file),
                    "text": text,
                    "duration": duration,
                    "dataset_id": dataset_id,
                }
                ref_audio = ent.get("ref_audio")
                if ref_audio is not None:
                    ref_wav = _audio_dict_to_wav(ref_audio, sample_rate)
                    ref_file = wav_dir / f"sample_{idx:05d}_ref.wav"
                    _safe_save_wav(str(ref_file), ref_wav.unsqueeze(0), sample_rate)
                    record["ref_audio"] = str(ref_file)
                    record["ref_duration"] = float(ref_wav.shape[-1]) / float(sample_rate)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                num += 1

            extra = (kwargs.get("extra_manifest") or "").strip()
            if extra:
                if not os.path.isfile(extra):
                    raise FileNotFoundError(f"extra_manifest not found: {extra}")
                with open(extra, "r", encoding="utf-8") as ef:
                    for line in ef:
                        line = line.strip()
                        if not line:
                            continue
                        f.write(line + "\n")
                        num += 1

        logger.info("Built manifest %s with %d samples", manifest_path, num)
        return (str(manifest_path), num)


# --------------------------------------------------------------------------- #
# Training nodes
# --------------------------------------------------------------------------- #


def _build_lora_config(arch: str, *, enable_lm, enable_dit, enable_proj, r, alpha, dropout):
    if arch == "voxcpm2":
        from voxcpm.model.voxcpm2 import LoRAConfig
    else:
        from voxcpm.model.voxcpm import LoRAConfig
    return LoRAConfig(
        enable_lm=bool(enable_lm),
        enable_dit=bool(enable_dit),
        enable_proj=bool(enable_proj),
        r=int(r),
        alpha=int(alpha),
        dropout=float(dropout),
    )


def _save_lora_checkpoint(model, save_dir: Path, pretrained_path: str):
    """Save LoRA weights + config to save_dir. Mirrors upstream save_checkpoint()."""
    try:
        from safetensors.torch import save_file as safe_save
        has_safetensors = True
    except ImportError:
        has_safetensors = False

    save_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = model.module if hasattr(model, "module") else model
    full_state = unwrapped.state_dict()
    lora_cfg = unwrapped.lora_config
    state_dict = {k: v for k, v in full_state.items() if "lora_" in k}

    if has_safetensors:
        safe_save(state_dict, str(save_dir / "lora_weights.safetensors"))
    else:
        torch.save({"state_dict": state_dict}, save_dir / "lora_weights.ckpt")

    lora_info = {
        "base_model": str(pretrained_path) if pretrained_path else None,
        "lora_config": (
            lora_cfg.model_dump() if hasattr(lora_cfg, "model_dump") else vars(lora_cfg)
        ),
    }
    with open(save_dir / "lora_config.json", "w", encoding="utf-8") as f:
        json.dump(lora_info, f, indent=2, ensure_ascii=False)


def _save_full_checkpoint(model, save_dir: Path, pretrained_path: str):
    """Save full finetune weights (excluding audio_vae) and copy config files."""
    import shutil
    try:
        from safetensors.torch import save_file as safe_save
        has_safetensors = True
    except ImportError:
        has_safetensors = False

    save_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = model.module if hasattr(model, "module") else model
    full_state = unwrapped.state_dict()
    state_dict = {k: v for k, v in full_state.items() if not k.startswith("audio_vae.")}

    if has_safetensors:
        safe_save(state_dict, str(save_dir / "model.safetensors"))
    else:
        torch.save({"state_dict": state_dict}, save_dir / "pytorch_model.bin")

    if pretrained_path:
        pretrained_dir = Path(pretrained_path)
        for fname in [
            "config.json",
            "audiovae.pth",
            "audiovae.safetensors",
            "tokenizer.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
        ]:
            src = pretrained_dir / fname
            if src.exists():
                shutil.copy2(src, save_dir / fname)


def _run_training(
    *,
    pretrained_path: str,
    train_manifest: str,
    val_manifest: str,
    num_iters: int,
    batch_size: int,
    grad_accum_steps: int,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    max_grad_norm: float,
    num_workers: int,
    log_interval: int,
    save_interval: int,
    save_path: str,
    lora_enable: bool,
    lora_cfg: Optional[object],
    pbar: "comfy.utils.ProgressBar",
) -> Path:
    """In-process VoxCPM training loop.

    Adapted from `scripts/train_voxcpm_finetune.py`, stripped of distributed
    support (single-GPU / CPU only) and the validation / TensorBoard paths,
    while preserving the core optimizer + scheduler + gradient accumulation
    pipeline.

    Returns the path to the final checkpoint folder.

    .. note::
        ComfyUI wraps the entire prompt execution with
        ``torch.inference_mode()``, which silently turns every tensor created
        inside the workflow into an "inference tensor". Those tensors never
        participate in autograd — forward still runs and the loss value looks
        correct, but ``grad_fn`` stays ``None`` and ``backward`` fails with
        ``element 0 of tensors does not require grad and does not have a
        grad_fn``. We must therefore explicitly disable inference_mode and
        re-enable grad for the whole training loop, otherwise loss.backward()
        raises that misleading error (observed on RunningHub, 2026-04).
    """
    with torch.inference_mode(False), torch.enable_grad():
        return _run_training_inner(
            pretrained_path=pretrained_path,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            num_iters=num_iters,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            max_grad_norm=max_grad_norm,
            num_workers=num_workers,
            log_interval=log_interval,
            save_interval=save_interval,
            save_path=save_path,
            lora_enable=lora_enable,
            lora_cfg=lora_cfg,
            pbar=pbar,
        )


def _run_training_inner(
    *,
    pretrained_path: str,
    train_manifest: str,
    val_manifest: str,
    num_iters: int,
    batch_size: int,
    grad_accum_steps: int,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    max_grad_norm: float,
    num_workers: int,
    log_interval: int,
    save_interval: int,
    save_path: str,
    lora_enable: bool,
    lora_cfg: Optional[object],
    pbar: "comfy.utils.ProgressBar",
) -> Path:
    _ensure_voxcpm_src_importable()

    from voxcpm.model import VoxCPMModel, VoxCPM2Model
    from voxcpm.training import (
        Accelerator,
        BatchProcessor,
        TrainingTracker,
        build_dataloader,
        load_audio_text_datasets,
    )
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    arch = _detect_architecture(pretrained_path)
    model_cls = VoxCPM2Model if arch == "voxcpm2" else VoxCPMModel
    logger.info("Loading base model %s as %s", pretrained_path, model_cls.__name__)

    base_model = model_cls.from_local(
        pretrained_path,
        optimize=False,
        training=True,
        lora_config=lora_cfg if lora_enable else None,
    )
    tokenizer = base_model.text_tokenizer

    sample_rate = base_model.audio_vae.sample_rate

    train_ds, val_ds = load_audio_text_datasets(
        train_manifest=train_manifest,
        val_manifest=val_manifest or "",
        sample_rate=sample_rate,
    )

    def tokenize(batch):
        return {"text_ids": [tokenizer(t) for t in batch["text"]]}

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    if val_ds is not None:
        val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])

    dataset_cnt = (
        int(max(train_ds["dataset_id"])) + 1
        if "dataset_id" in train_ds.column_names
        else 1
    )

    accelerator = Accelerator(amp=torch.cuda.is_available())
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    tracker = TrainingTracker(writer=None, log_file=str(save_dir / "train.log"), rank=0)

    train_loader = build_dataloader(
        train_ds,
        accelerator=accelerator,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        drop_last=True,
    )

    batch_processor = BatchProcessor(
        config=base_model.config,
        audio_vae=base_model.audio_vae,
        dataset_cnt=dataset_cnt,
        device=accelerator.device,
    )

    # Drop audio_vae from the trainable tree (matches upstream behavior).
    del base_model.audio_vae
    model = accelerator.prepare_model(base_model)
    unwrapped_model = accelerator.unwrap(model)
    unwrapped_model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    tracker.print(
        f"Trainable parameters: {num_trainable:,} (LoRA={lora_enable}, arch={arch})"
    )
    if num_trainable == 0:
        raise RuntimeError(
            "No trainable parameters were found. "
            "When training in LoRA mode make sure at least one of "
            "enable_lm / enable_dit / enable_proj is True and lora_rank > 0."
        )

    optimizer = AdamW(
        trainable_params, lr=float(learning_rate), weight_decay=float(weight_decay)
    )

    total_steps = int(num_iters)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_steps),
        num_training_steps=total_steps,
    )

    grad_accum_steps = max(int(grad_accum_steps), 1)
    data_epoch = 0
    train_iter = iter(train_loader)

    def get_next_batch():
        nonlocal train_iter, data_epoch
        try:
            return next(train_iter)
        except StopIteration:
            data_epoch += 1
            sampler = getattr(train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(data_epoch)
            train_iter = iter(train_loader)
            return next(train_iter)

    lambdas = {"loss/diff": 1.0, "loss/stop": 1.0}

    last_folder: Optional[Path] = None
    start_t = time.time()

    for step in range(total_steps):
        tracker.step = step
        optimizer.zero_grad(set_to_none=True)
        loss_dict: Dict[str, torch.Tensor] = {}

        for micro_step in range(grad_accum_steps):
            batch = get_next_batch()
            processed = batch_processor(batch)

            is_last = micro_step == grad_accum_steps - 1
            sync_ctx = contextlib.nullcontext() if is_last else accelerator.no_sync()
            with sync_ctx:
                with accelerator.autocast(dtype=torch.bfloat16):
                    outputs = model(
                        processed["text_tokens"],
                        processed["text_mask"],
                        processed["audio_feats"],
                        processed["audio_mask"],
                        processed["loss_mask"],
                        processed["position_ids"],
                        processed["labels"],
                        progress=step / max(1, total_steps),
                    )
                total_loss = 0.0
                for k, v in outputs.items():
                    if k.startswith("loss/"):
                        total_loss = total_loss + v * lambdas.get(k, 1.0) / grad_accum_steps
                        loss_dict[k] = v.detach()

                if (
                    not isinstance(total_loss, torch.Tensor)
                    or not total_loss.requires_grad
                    or total_loss.grad_fn is None
                ):
                    raise RuntimeError(
                        "Training step produced a loss with no autograd graph. "
                        "Common causes: (1) the batch has no valid loss tokens "
                        "(e.g. text too short compared to audio, or manifest text "
                        "is empty/ASR-failed); (2) LoRA adapters are disabled or "
                        "lora_rank=0; (3) the model was loaded under no_grad. "
                        f"loss={total_loss!r}, outputs_keys={list(outputs.keys())}"
                    )

                accelerator.backward(total_loss)

        scaler = getattr(accelerator, "scaler", None)
        if scaler is not None:
            scaler.unscale_(optimizer)
        effective_max_norm = float(max_grad_norm) if float(max_grad_norm) > 0 else 1e9
        grad_norm = torch.nn.utils.clip_grad_norm_(
            unwrapped_model.parameters(), max_norm=effective_max_norm
        )

        accelerator.step(optimizer)
        accelerator.update()
        scheduler.step()

        if step % int(log_interval) == 0 or step == total_steps - 1:
            loss_values = {
                k: (v.item() if isinstance(v, torch.Tensor) else float(v))
                for k, v in loss_dict.items()
            }
            loss_values["lr"] = float(optimizer.param_groups[0]["lr"])
            loss_values["grad_norm"] = float(grad_norm)
            elapsed = time.time() - start_t
            loss_values["elapsed_s"] = float(elapsed)
            tracker.log_metrics(loss_values, split="train")

        if save_interval > 0 and (
            (step > 0 and step % int(save_interval) == 0) or step == total_steps - 1
        ):
            folder = save_dir / f"step_{step + 1:07d}"
            folder.mkdir(parents=True, exist_ok=True)
            if lora_enable:
                _save_lora_checkpoint(model, folder, pretrained_path)
            else:
                _save_full_checkpoint(model, folder, pretrained_path)
            last_folder = folder
            tracker.print(f"Saved checkpoint to {folder}")

        pbar.update_absolute(step + 1, total_steps)

    # Final save if nothing was written yet (e.g. save_interval=0)
    if last_folder is None:
        folder = save_dir / f"step_{total_steps:07d}"
        folder.mkdir(parents=True, exist_ok=True)
        if lora_enable:
            _save_lora_checkpoint(model, folder, pretrained_path)
        else:
            _save_full_checkpoint(model, folder, pretrained_path)
        last_folder = folder
        tracker.print(f"Saved final checkpoint to {folder}")

    # Copy last checkpoint to `latest/` for easy reloading.
    import shutil
    latest_folder = save_dir / "latest"
    try:
        if latest_folder.exists():
            shutil.rmtree(latest_folder)
        shutil.copytree(last_folder, latest_folder)
    except Exception as e:
        tracker.print(f"Warning: failed to update latest checkpoint: {e}")

    return last_folder


class RunningHubVoxCPMTrainLoRA:
    """LoRA fine-tune VoxCPM on a jsonl manifest (single-GPU, in-process)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_list_model_dirs(),),
                "train_manifest": ("STRING", {"default": "", "multiline": False}),
                "output_name": ("STRING", {"default": "VoxCPM2_lora"}),
                "num_iters": ("INT", {"default": 500, "min": 1, "max": 200000, "step": 1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "grad_accum_steps": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "learning_rate": ("FLOAT", {
                    "default": 1e-4,
                    "min": 1e-7,
                    "max": 1e-1,
                    "step": 1e-5,
                }),
                "lora_rank": ("INT", {"default": 32, "min": 1, "max": 256, "step": 1}),
                "lora_alpha": ("INT", {"default": 32, "min": 1, "max": 512, "step": 1}),
            },
            "optional": {
                "val_manifest": ("STRING", {"default": "", "multiline": False}),
                "warmup_steps": ("INT", {"default": 100, "min": 0, "max": 10000, "step": 1}),
                "weight_decay": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001}),
                "max_grad_norm": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "num_workers": ("INT", {"default": 2, "min": 0, "max": 16, "step": 1}),
                "log_interval": ("INT", {"default": 10, "min": 1, "max": 10000, "step": 1}),
                "save_interval": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "lora_dropout": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01}),
                "enable_lm": ("BOOLEAN", {"default": True}),
                "enable_dit": ("BOOLEAN", {"default": True}),
                "enable_proj": ("BOOLEAN", {"default": False}),
                "copy_to_loras_dir": ("BOOLEAN", {"default": True}),
                "zip_to_output": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("lora_path", "info")
    FUNCTION = "train"
    CATEGORY = "RunningHub/VoxCPM/Train"
    OUTPUT_NODE = True

    def __init__(self):
        self.type = "output"

    def train(
        self,
        model_name,
        train_manifest,
        output_name,
        num_iters,
        batch_size,
        grad_accum_steps,
        learning_rate,
        lora_rank,
        lora_alpha,
        val_manifest="",
        warmup_steps=100,
        weight_decay=0.01,
        max_grad_norm=1.0,
        num_workers=2,
        log_interval=10,
        save_interval=0,
        lora_dropout=0.0,
        enable_lm=True,
        enable_dit=True,
        enable_proj=False,
        copy_to_loras_dir=True,
        zip_to_output=True,
    ):
        _ensure_voxcpm_src_importable()
        train_manifest = (train_manifest or "").strip()
        if not train_manifest or not os.path.isfile(train_manifest):
            raise FileNotFoundError(
                f"train_manifest '{train_manifest}' not found. "
                f"Build one with the 'VoxCPM Dataset Build' node or provide a jsonl path."
            )

        pretrained_path = _resolve_model_path(model_name)
        arch = _detect_architecture(pretrained_path)
        lora_cfg = _build_lora_config(
            arch,
            enable_lm=enable_lm,
            enable_dit=enable_dit,
            enable_proj=enable_proj,
            r=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )

        ts = time.strftime("%Y%m%d_%H%M%S")
        safe_name = (output_name or "voxcpm_lora").strip().replace(" ", "_")
        run_dir = Path(_get_output_root()) / f"{safe_name}_{ts}"
        ckpt_dir = run_dir / "checkpoints"

        total_pbar_steps = int(num_iters)
        pbar = comfy.utils.ProgressBar(total_pbar_steps)

        last_folder = _run_training(
            pretrained_path=pretrained_path,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            num_iters=num_iters,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            max_grad_norm=max_grad_norm,
            num_workers=num_workers,
            log_interval=log_interval,
            save_interval=save_interval,
            save_path=str(ckpt_dir),
            lora_enable=True,
            lora_cfg=lora_cfg,
            pbar=pbar,
        )

        final_path = last_folder
        if copy_to_loras_dir:
            import shutil

            loras_root = Path(_get_lora_dir())
            src_safetensors = Path(last_folder) / "lora_weights.safetensors"
            src_ckpt = Path(last_folder) / "lora_weights.ckpt"
            flat_name = f"{safe_name}_{ts}.safetensors"
            target_file = loras_root / flat_name
            try:
                loras_root.mkdir(parents=True, exist_ok=True)
                if src_safetensors.is_file():
                    src = src_safetensors
                elif src_ckpt.is_file():
                    src = src_ckpt
                    target_file = loras_root / f"{safe_name}_{ts}.ckpt"
                else:
                    src = None
                    logger.warning(
                        "Could not find lora_weights.{safetensors,ckpt} under %s; "
                        "skipping copy to models/voxcpm/loras",
                        last_folder,
                    )

                if src is not None:
                    if target_file.exists():
                        target_file.unlink()
                    shutil.copy2(src, target_file)
                    final_path = target_file
                    logger.info(
                        "Copied LoRA weights to %s (renamed from %s)",
                        target_file,
                        src.name,
                    )
            except Exception as e:
                logger.warning("Failed to copy LoRA to models/voxcpm/loras: %s", e)

        zip_info: Optional[dict] = None
        if zip_to_output:
            zip_info = _zip_checkpoint_to_output(
                Path(last_folder), f"{safe_name}_{ts}"
            )
            if zip_info is not None:
                logger.info("Wrote LoRA zip to %s", zip_info["zip_path"])

        info_lines = [
            "LoRA training complete.",
            f"  base model : {model_name} ({arch})",
            f"  iterations : {num_iters}",
            f"  batch size : {batch_size} x grad_accum {grad_accum_steps}",
            f"  lr         : {learning_rate}",
            f"  rank/alpha : {lora_rank}/{lora_alpha}",
            f"  output dir : {final_path}",
        ]
        if zip_info is not None:
            info_lines.append(f"  output zip : {zip_info['zip_path']}")
        info = "\n".join(info_lines)
        logger.info(info)

        ui_payload = {}
        if zip_info is not None:
            # Surface the zip under a dedicated ``voxcpm_lora`` UI key rather
            # than ``images`` so the RunningHub web UI can pick it up for
            # download while avoiding any "is this an image?" confusion.
            ui_payload = {
                "voxcpm_lora": [
                    {
                        "filename": zip_info["filename"],
                        "subfolder": zip_info["subfolder"],
                        "type": "output",
                    }
                ]
            }

        return {"ui": ui_payload, "result": (str(final_path), info)}


class RunningHubVoxCPMTrainFull:
    """Full-parameter fine-tune (no LoRA). Warning: requires lots of VRAM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_list_model_dirs(),),
                "train_manifest": ("STRING", {"default": "", "multiline": False}),
                "output_name": ("STRING", {"default": "VoxCPM2_full"}),
                "num_iters": ("INT", {"default": 500, "min": 1, "max": 200000, "step": 1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "grad_accum_steps": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
                "learning_rate": ("FLOAT", {
                    "default": 1e-5,
                    "min": 1e-7,
                    "max": 1e-1,
                    "step": 1e-6,
                }),
            },
            "optional": {
                "val_manifest": ("STRING", {"default": "", "multiline": False}),
                "warmup_steps": ("INT", {"default": 100, "min": 0, "max": 10000, "step": 1}),
                "weight_decay": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001}),
                "max_grad_norm": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "num_workers": ("INT", {"default": 2, "min": 0, "max": 16, "step": 1}),
                "log_interval": ("INT", {"default": 10, "min": 1, "max": 10000, "step": 1}),
                "save_interval": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "zip_to_output": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("checkpoint_path", "info")
    FUNCTION = "train"
    CATEGORY = "RunningHub/VoxCPM/Train"
    OUTPUT_NODE = True

    def __init__(self):
        self.type = "output"

    def train(
        self,
        model_name,
        train_manifest,
        output_name,
        num_iters,
        batch_size,
        grad_accum_steps,
        learning_rate,
        val_manifest="",
        warmup_steps=100,
        weight_decay=0.01,
        max_grad_norm=1.0,
        num_workers=2,
        log_interval=10,
        save_interval=0,
        zip_to_output=True,
    ):
        _ensure_voxcpm_src_importable()
        train_manifest = (train_manifest or "").strip()
        if not train_manifest or not os.path.isfile(train_manifest):
            raise FileNotFoundError(
                f"train_manifest '{train_manifest}' not found. "
                f"Build one with the 'VoxCPM Dataset Build' node or provide a jsonl path."
            )

        pretrained_path = _resolve_model_path(model_name)
        arch = _detect_architecture(pretrained_path)

        ts = time.strftime("%Y%m%d_%H%M%S")
        safe_name = (output_name or "voxcpm_full").strip().replace(" ", "_")
        run_dir = Path(_get_output_root()) / f"{safe_name}_{ts}"
        ckpt_dir = run_dir / "checkpoints"

        total_pbar_steps = int(num_iters)
        pbar = comfy.utils.ProgressBar(total_pbar_steps)

        last_folder = _run_training(
            pretrained_path=pretrained_path,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            num_iters=num_iters,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            max_grad_norm=max_grad_norm,
            num_workers=num_workers,
            log_interval=log_interval,
            save_interval=save_interval,
            save_path=str(ckpt_dir),
            lora_enable=False,
            lora_cfg=None,
            pbar=pbar,
        )

        zip_info: Optional[dict] = None
        if zip_to_output:
            zip_info = _zip_checkpoint_to_output(
                Path(last_folder), f"{safe_name}_{ts}"
            )
            if zip_info is not None:
                logger.info("Wrote full-finetune zip to %s", zip_info["zip_path"])

        info_lines = [
            "Full fine-tune complete.",
            f"  base model : {model_name} ({arch})",
            f"  iterations : {num_iters}",
            f"  batch size : {batch_size} x grad_accum {grad_accum_steps}",
            f"  lr         : {learning_rate}",
            f"  output dir : {last_folder}",
        ]
        if zip_info is not None:
            info_lines.append(f"  output zip : {zip_info['zip_path']}")
        info = "\n".join(info_lines)
        logger.info(info)

        ui_payload = {}
        if zip_info is not None:
            ui_payload = {
                "voxcpm_full": [
                    {
                        "filename": zip_info["filename"],
                        "subfolder": zip_info["subfolder"],
                        "type": "output",
                    }
                ]
            }

        return {"ui": ui_payload, "result": (str(last_folder), info)}
