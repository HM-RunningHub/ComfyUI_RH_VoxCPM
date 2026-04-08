# ComfyUI_RH_VoxCPM

[中文说明](README_zh.md)

ComfyUI custom nodes for [VoxCPM](https://github.com/OpenBMB/VoxCPM) — Tokenizer-Free TTS for Context-Aware Speech Generation and True-to-Life Voice Cloning.

Run this node online: [RunningHub (CN)](https://www.runninghub.cn/?inviteCode=rh-v1367) | [RunningHub (Global)](https://www.runninghub.ai/?inviteCode=rh-v1367)

## Features

- **Voice Design**: Create unique voices from text descriptions (gender, age, tone, emotion, pace)
- **Controllable Cloning**: Clone a voice with optional style guidance via reference audio
- **Ultimate Cloning**: Reproduce every vocal nuance through audio continuation (VoxCPM2 only)
- **LoRA Fine-tuning**: Load custom LoRA weights for personalized voice generation
- **Auto ASR**: Automatically recognize reference audio text via FunASR SenseVoiceSmall when `reference_audio_text` is empty
- **Reference Denoising**: Optional ZipEnhancer denoising for reference audio before cloning

## Nodes

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

## Model Setup

### VoxCPM Models (required, pick one)

Place model directory under `ComfyUI/models/voxcpm/`:

| Model | Size | Download |
|-------|------|----------|
| VoxCPM2 (2B, recommended) | ~4.6 GB | https://huggingface.co/openbmb/VoxCPM2 |
| VoxCPM1.5 (800M) | ~1.9 GB | https://huggingface.co/openbmb/VoxCPM1.5 |
| VoxCPM-0.5B (640M) | ~1.5 GB | https://huggingface.co/openbmb/VoxCPM-0.5B |

```
ComfyUI/models/voxcpm/
  VoxCPM2/
    config.json
    model.safetensors
    audiovae.pth
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
```

### LoRA Weights (optional)

Place LoRA weight files (`.pth`, `.ckpt`, `.safetensors`) or directories under `ComfyUI/models/voxcpm/loras/`:

```
ComfyUI/models/voxcpm/loras/
  my_custom_voice.pth
  another_lora/
    lora_weights.ckpt
```

### SenseVoiceSmall (required for auto ASR)

```
ComfyUI/models/SenseVoice/SenseVoiceSmall/
```

Download: ModelScope `iic/SenseVoiceSmall`

### ZipEnhancer (optional, for reference audio denoising)

```
ComfyUI/models/voxcpm/speech_zipenhancer_ans_multiloss_16k_base/
```

Download: ModelScope `iic/speech_zipenhancer_ans_multiloss_16k_base`

## Dependencies

```
pip install voxcpm
```

## Usage

1. Add **RunningHub VoxCPM Load Model** node, select your model directory (and optionally a LoRA)
2. Add **RunningHub VoxCPM Generate Speech** node, connect the model output
3. Enter target text and optionally provide control instruction or reference audio
4. Connect output to a **Preview Audio** or **Save Audio** node

### Three Modes

- **Voice Design**: Fill `control_instruction` (e.g. "A warm young woman"), leave `reference_audio` empty. The model creates a brand-new voice from your description alone.
- **Controllable Cloning**: Upload `reference_audio`, keep `ultimate_clone` OFF. Use `control_instruction` to steer emotion, pace, and style while preserving the reference timbre.
- **Ultimate Cloning**: Upload `reference_audio`, turn `ultimate_clone` ON. The model treats the reference as a spoken prefix and continues from it, faithfully reproducing every vocal detail. `control_instruction` is ignored in this mode. If `reference_audio_text` is empty, ASR will auto-recognize it.
