# ComfyUI_RH_VoxCPM

ComfyUI custom nodes for [VoxCPM](https://github.com/OpenBMB/VoxCPM) — Tokenizer-Free TTS for Context-Aware Speech Generation and True-to-Life Voice Cloning.

## Features

- **Voice Design**: Create unique voices from text descriptions (gender, age, tone, emotion, pace)
- **Controllable Cloning**: Clone a voice with optional style guidance
- **Ultimate Cloning**: Reproduce every vocal nuance through audio continuation (VoxCPM2 only)

## Nodes

| Node | Description |
|------|-------------|
| VoxCPM: Load Model | Load VoxCPM/VoxCPM2 model from local directory |
| VoxCPM: Generate Speech | Generate speech with voice design, controllable cloning, or ultimate cloning |

## Model Setup

Place your VoxCPM model directory under `ComfyUI/models/voxcpm/`. The directory should contain `config.json`, model weights, and tokenizer files.

Example structure:
```
ComfyUI/models/voxcpm/
  VoxCPM2/
    config.json
    model.safetensors
    audiovae.safetensors
    tokenizer.json
    ...
```

## Dependencies

```
pip install voxcpm
```

## Usage

1. Add **VoxCPM: Load Model** node, select your model directory
2. Add **VoxCPM: Generate Speech** node, connect the model output
3. Enter target text and optionally provide control instruction or reference audio
4. Connect output to a **Preview Audio** or **Save Audio** node
