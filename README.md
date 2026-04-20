# ComfyUI_RH_VoxCPM

![License](https://img.shields.io/badge/License-Apache%202.0-green)

[中文说明](README_zh.md)

ComfyUI custom nodes for [VoxCPM](https://github.com/OpenBMB/VoxCPM) — Tokenizer-Free TTS for Context-Aware Speech Generation and True-to-Life Voice Cloning.

Run this node online: [RunningHub (CN)](https://www.runninghub.cn/?inviteCode=rh-v1367) | [RunningHub (Global)](https://www.runninghub.ai/?inviteCode=rh-v1367)

GitHub Repository: [HM-RunningHub/ComfyUI_RH_VoxCPM](https://github.com/HM-RunningHub/ComfyUI_RH_VoxCPM)

## ✨ Features

- **Voice Design**: Create unique voices from text descriptions (gender, age, tone, emotion, pace)
- **Controllable Cloning**: Clone a voice with optional style guidance via reference audio
- **Ultimate Cloning**: Reproduce every vocal nuance through audio continuation (VoxCPM2 only)
- **LoRA Fine-tuning**: Load custom LoRA weights for personalized voice generation
- **LoRA / Full Training**: Train VoxCPM LoRA (or full fine-tune) directly from a ComfyUI workflow, reusing the upstream training loop
- **Auto ASR**: Automatically recognize reference audio text via FunASR SenseVoiceSmall when `reference_audio_text` is empty
- **Reference Denoising**: Optional ZipEnhancer denoising for reference audio before cloning

## 🛠️ Installation

### Method 1: Clone from GitHub

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/HM-RunningHub/ComfyUI_RH_VoxCPM.git
cd ComfyUI_RH_VoxCPM
pip install -r requirements.txt
```

### Method 2: ComfyUI Manager

Search for `ComfyUI_RH_VoxCPM` in ComfyUI Manager and install.

## 📦 Model Download & Installation

### VoxCPM Models (required, pick one)

| Model | Params | Size | Recommended |
|-------|--------|------|-------------|
| [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | 2B | ~4.6 GB | ✅ Best quality |
| [VoxCPM1.5](https://huggingface.co/openbmb/VoxCPM1.5) | 800M | ~1.9 GB | Good balance |
| [VoxCPM-0.5B](https://huggingface.co/openbmb/VoxCPM-0.5B) | 640M | ~1.5 GB | Lightweight |

#### Method 1: Download from HuggingFace (Recommended)

```bash
huggingface-cli download openbmb/VoxCPM2 --local-dir ComfyUI/models/voxcpm/VoxCPM2
```

#### Method 2: Download from ModelScope (For China users)

```bash
pip install modelscope
modelscope download --model openbmb/VoxCPM2 --local_dir ComfyUI/models/voxcpm/VoxCPM2
```

### Model Directory Structure

```
ComfyUI/
└── models/
    └── voxcpm/
        ├── VoxCPM2/                # Main model (required)
        │   ├── config.json
        │   ├── model.safetensors
        │   ├── audiovae.pth
        │   ├── tokenizer.json
        │   ├── tokenizer_config.json
        │   └── special_tokens_map.json
        ├── loras/                  # LoRA weights (optional)
        │   └── my_custom_voice.pth
        └── speech_zipenhancer_ans_multiloss_16k_base/  # Denoiser (optional)
```

### SenseVoiceSmall (required for auto ASR)

```bash
# From ModelScope
modelscope download --model iic/SenseVoiceSmall --local_dir ComfyUI/models/SenseVoice/SenseVoiceSmall
```

### ZipEnhancer (optional, for reference audio denoising)

```bash
# From ModelScope
modelscope download --model iic/speech_zipenhancer_ans_multiloss_16k_base --local_dir ComfyUI/models/voxcpm/speech_zipenhancer_ans_multiloss_16k_base
```

## 🚀 Usage

### Example Workflows

Download example workflows from the [`examples/`](examples/) directory and import into ComfyUI:

1. **[Basic Workflow](examples/VoxCPM2%20基础工作流.json)** — Single-speaker speech generation with voice design / cloning
2. **[Multi-Speaker Workflow](examples/VoxCPM2%20多人工作流.json)** — Fixed 5-speaker multi-speaker dialogue generation with per-speaker voice control
3. **[LoRA Training Workflow](examples/VoxCPM2%20LoRA%20训练工作流.json)** — Build a tiny dataset from two audio clips and run a LoRA fine-tune

Notes:

- `RunningHub VoxCPM Multi-Speaker` is the fixed 5-speaker version
- `RunningHub VoxCPM Multi-Speaker (Dynamic Audio)` uses the same script format but grows reference-audio inputs automatically
- If the dynamic inputs do not appear after updating the plugin, refresh the ComfyUI frontend page or reopen the workflow

### Three Modes

- **Voice Design**: Fill `control_instruction` (e.g. "A warm young woman"), leave `reference_audio` empty. The model creates a brand-new voice from your description alone.
- **Controllable Cloning**: Upload `reference_audio`, keep `ultimate_clone` OFF. Use `control_instruction` to steer emotion, pace, and style while preserving the reference timbre.
- **Ultimate Cloning**: Upload `reference_audio`, turn `ultimate_clone` ON. The model treats the reference as a spoken prefix and continues from it, faithfully reproducing every vocal detail. `control_instruction` is ignored in this mode. If `reference_audio_text` is empty, ASR will auto-recognize it.

## 📝 Node Reference

### RunningHub VoxCPM Load Model

Load VoxCPM/VoxCPM2 model from local directory with optional LoRA weights.

| Input | Type | Description |
|-------|------|-------------|
| model_name | COMBO | Model directory under `models/voxcpm/` |
| optimize | BOOLEAN | Enable torch.compile optimization (default: off) |
| lora_name | COMBO | LoRA weights under `models/voxcpm/loras/` (optional, default: None) |

### RunningHub VoxCPM Generate Speech

Generate speech with voice design, controllable cloning, or ultimate cloning.

| Input | Type | Description |
|-------|------|-------------|
| model | VOXCPM_MODEL | Model from Load Model node |
| text | STRING | Target text to synthesize |
| cfg_value | FLOAT | Guidance scale (default: 2.0) |
| inference_steps | INT | LocDiT flow-matching steps (default: 10) |
| seed | INT | Random seed for reproducibility |
| control_instruction | STRING | Voice description for voice design mode (optional) |
| reference_audio | AUDIO | Reference audio for cloning (optional) |
| ultimate_clone | BOOLEAN | Enable ultimate cloning mode (default: off) |
| reference_audio_text | STRING | Transcript of reference audio; auto ASR if empty (optional) |
| normalize_text | BOOLEAN | Text normalization (default: off) |
| denoise_reference | BOOLEAN | Denoise reference audio via ZipEnhancer (default: off) |
| max_len | INT | Maximum token length during generation (default: 4096) |
| retry_badcase | BOOLEAN | Auto-retry when output quality is poor (default: on) |

### RunningHub VoxCPM Multi-Speaker

Generate multi-speaker dialogue from a tagged script. Supports up to 5 speakers with individual voice control.

| Input | Type | Description |
|-------|------|-------------|
| model | VOXCPM_MODEL | Model from Load Model node |
| script | STRING | Tagged script, e.g. `[spk1]Hello[spk2]Hi there` |
| cfg_value | FLOAT | Guidance scale (default: 2.0) |
| inference_steps | INT | LocDiT flow-matching steps (default: 10) |
| seed | INT | Random seed for reproducibility |
| audio_1 ~ audio_5 | AUDIO | Reference audio for each speaker (optional) |
| control_1 ~ control_5 | STRING | Voice description for each speaker (optional) |
| normalize_text | BOOLEAN | Text normalization (default: off) |
| denoise_reference | BOOLEAN | Denoise reference audio via ZipEnhancer (default: off) |
| max_len | INT | Maximum token length during generation (default: 4096) |
| retry_badcase | BOOLEAN | Auto-retry when output quality is poor (default: on) |

### RunningHub VoxCPM Multi-Speaker (Dynamic Audio)

For multi-speaker reference-audio workflows. The script still uses `[spk1]...[spk2]...` tags, while speaker control instructions are merged into a single multiline input using the same tag format. The node shows 2 reference-audio inputs by default and automatically adds the next one when all current inputs are connected, with no fixed upper limit. At execution time, `audio_1` maps to `spk1`, `audio_2` maps to `spk2`, and so on, so tags like `spk10` and `spk20` are supported as well.

Usage tips:

- You need to connect all currently visible `audio_*` inputs before the next one is added
- This auto-growth behavior depends on the frontend extension script; if it does not update after installing a new version, refresh the page

| Input | Type | Description |
|-------|------|-------------|
| model | VOXCPM_MODEL | Model from Load Model node |
| script | STRING | Tagged script, e.g. `[spk1]Hello[spk2]Hi there` |
| speaker_controls | STRING | Multiline tagged controls, e.g. `[spk1]Sichuan accent\n[spk2]Adult female, northeastern accent` |
| cfg_value | FLOAT | Guidance scale (default: 2.0) |
| inference_steps | INT | LocDiT flow-matching steps (default: 10) |
| seed | INT | Random seed for reproducibility |
| audio_1 ~ audio_N | AUDIO | Dynamic reference-audio inputs mapped to `spk1 ~ spkN` by slot order; starts with 2, auto-grows when filled, and has no fixed upper limit |
| normalize_text | BOOLEAN | Text normalization (default: off) |
| denoise_reference | BOOLEAN | Denoise reference audio via ZipEnhancer (default: off) |
| max_len | INT | Maximum token length during generation (default: 4096) |
| retry_badcase | BOOLEAN | Auto-retry when output quality is poor (default: on) |

## 🎓 Training Nodes (LoRA / Full Fine-tuning)

> ⚠️ The training nodes rely on the upstream training modules (`voxcpm.training.*`). They pull `transformers / datasets / safetensors / argbind` via `requirements.txt`, and require a full [VoxCPM](https://github.com/OpenBMB/VoxCPM) source tree to be available — either install the full repo, or drop a checkout next to this plugin (e.g. `ComfyUI/custom_nodes/VoxCPM/src/voxcpm/training/`) or inside `<plugin>/voxcpm/src/`.

Typical workflow:
1. **Dataset Entry** wraps a single (audio, text) pair into a training sample.
2. **Dataset Build** aggregates samples into a `train.jsonl` manifest (an existing jsonl path also works).
3. **Train LoRA** / **Train Full** runs the training loop. Artifacts are written to `ComfyUI/output/voxcpm_train/<name>_<timestamp>/`; with `copy_to_loras_dir` enabled LoRA weights are also copied to `ComfyUI/models/voxcpm/loras/` so the Load Model node picks them up after a frontend refresh.

### RunningHub VoxCPM Dataset Entry

| Input | Type | Description |
|-------|------|-------------|
| audio | AUDIO | Training clip |
| text | STRING | Transcript for the clip |
| dataset_id | INT | Optional dataset id for multi-dataset training (default: 0) |
| ref_audio | AUDIO | Optional voice-style reference audio. When provided it is written to the manifest as `ref_audio` and used by the training pipeline for voice conditioning (requires voxcpm built after 2026-04) |

### RunningHub VoxCPM Dataset Build

| Input | Type | Description |
|-------|------|-------------|
| entry_1, entry_2 | VOXCPM_DATA_ENTRY | At least two samples |
| entry_3 ~ entry_8 | VOXCPM_DATA_ENTRY | Additional samples (optional) |
| extra_manifest | STRING | Path to an existing jsonl to append (optional) |
| sample_rate | INT | Sample rate to save WAVs at; match the base model AudioVAE (default: 16000) |
| dataset_name | STRING | Output directory prefix |

Outputs `manifest_path` (path to `train.jsonl`) and `num_samples`.

### RunningHub VoxCPM Train LoRA

| Input | Type | Description |
|-------|------|-------------|
| model_name | COMBO | Base model directory under `models/voxcpm/` |
| train_manifest | STRING | Training manifest (jsonl) path (use Dataset Build output) |
| output_name | STRING | Output name prefix (the final folder is suffixed with a timestamp) |
| num_iters | INT | Total training steps (default: 500) |
| batch_size | INT | Per-step batch size (default: 1) |
| grad_accum_steps | INT | Gradient accumulation steps (default: 1) |
| learning_rate | FLOAT | Learning rate (default: 1e-4) |
| lora_rank | INT | LoRA rank (default: 32) |
| lora_alpha | INT | LoRA alpha (default: 32) |
| val_manifest | STRING | Optional validation manifest |
| warmup_steps | INT | Warmup steps (default: 100) |
| weight_decay | FLOAT | Weight decay (default: 0.01) |
| max_grad_norm | FLOAT | Gradient clipping; 0 = disabled (default: 1.0) |
| num_workers | INT | Data loader workers (default: 2) |
| log_interval | INT | Log interval in steps (default: 10) |
| save_interval | INT | Checkpoint interval; 0 = save only at the end (default: 0) |
| lora_dropout | FLOAT | LoRA dropout (default: 0.0) |
| enable_lm | BOOLEAN | Apply LoRA to the LM (default: on) |
| enable_dit | BOOLEAN | Apply LoRA to the DiT (default: on) |
| enable_proj | BOOLEAN | Apply LoRA to projection layers (default: off) |
| copy_to_loras_dir | BOOLEAN | Copy final LoRA to `models/voxcpm/loras/` (default: on) |

Outputs `lora_path` (folder containing `lora_weights.safetensors` + `lora_config.json`) and `info` (summary string).

### RunningHub VoxCPM Train Full

Mirrors the LoRA node without LoRA-specific inputs. ⚠️ Full fine-tuning is memory-heavy; prefer the LoRA node for voice adaptation.

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

## 🔗 Links

- [RunningHub](https://www.runninghub.cn)
- [VoxCPM (Original Project)](https://github.com/OpenBMB/VoxCPM)
- [VoxCPM2 on HuggingFace](https://huggingface.co/openbmb/VoxCPM2)

## 🙏 Acknowledgements

This project is based on [VoxCPM](https://github.com/OpenBMB/VoxCPM), developed by [OpenBMB](https://github.com/OpenBMB) / [ModelBest](https://modelbest.cn).
