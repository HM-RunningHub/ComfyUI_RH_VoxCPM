# ComfyUI_RH_VoxCPM

![License](https://img.shields.io/badge/License-Apache%202.0-green)

[中文说明](README_zh.md)

ComfyUI custom nodes for [VoxCPM](https://github.com/OpenBMB/VoxCPM) — Tokenizer-Free TTS for Context-Aware Speech Generation and True-to-Life Voice Cloning.

Run this node online: [RunningHub (CN)](https://www.runninghub.cn/?inviteCode=rh-v1367) | [RunningHub (Global)](https://www.runninghub.ai/?inviteCode=rh-v1367)

## ✨ Features

- **Voice Design**: Create unique voices from text descriptions (gender, age, tone, emotion, pace)
- **Controllable Cloning**: Clone a voice with optional style guidance via reference audio
- **Ultimate Cloning**: Reproduce every vocal nuance through audio continuation (VoxCPM2 only)
- **LoRA Fine-tuning**: Load custom LoRA weights for personalized voice generation
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
2. **[Multi-Speaker Workflow](examples/VoxCPM2%20多人工作流.json)** — Multi-speaker dialogue generation with per-speaker voice control

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

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

## 🔗 Links

- [RunningHub](https://www.runninghub.cn)
- [VoxCPM (Original Project)](https://github.com/OpenBMB/VoxCPM)
- [VoxCPM2 on HuggingFace](https://huggingface.co/openbmb/VoxCPM2)

## 🙏 Acknowledgements

This project is based on [VoxCPM](https://github.com/OpenBMB/VoxCPM), developed by [OpenBMB](https://github.com/OpenBMB) / [ModelBest](https://modelbest.cn).
