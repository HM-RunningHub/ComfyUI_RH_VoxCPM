# ComfyUI_RH_VoxCPM

![License](https://img.shields.io/badge/License-Apache%202.0-green)

[English](README.md)

[VoxCPM](https://github.com/OpenBMB/VoxCPM) 的 ComfyUI 自定义节点 — 无分词器 TTS，支持上下文感知语音生成与高保真声音克隆。

在线使用：[RunningHub 国内版](https://www.runninghub.cn/?inviteCode=rh-v1367) | [RunningHub 国际版](https://www.runninghub.ai/?inviteCode=rh-v1367)

## ✨ 功能特性

- **声音设计**：通过文字描述创造全新声音（性别、年龄、语调、情感、语速）
- **可控克隆**：上传参考音频克隆音色，同时可用文字指令控制风格
- **极致克隆**：以音频续写方式复刻每一个声音细节（仅 VoxCPM2 支持）
- **LoRA 微调**：加载自定义 LoRA 权重，实现个性化语音生成
- **自动 ASR**：参考音频文本为空时，自动通过 FunASR SenseVoiceSmall 识别
- **参考音频降噪**：可选 ZipEnhancer 对参考音频进行降噪处理

## 🛠️ 安装

### 方法一：从 GitHub 克隆

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/HM-RunningHub/ComfyUI_RH_VoxCPM.git
cd ComfyUI_RH_VoxCPM
pip install -r requirements.txt
```

### 方法二：ComfyUI Manager

在 ComfyUI Manager 中搜索 `ComfyUI_RH_VoxCPM` 安装。

## 📦 模型下载与安装

### VoxCPM 模型（必需，选其一）

| 模型 | 参数量 | 大小 | 推荐 |
|------|--------|------|------|
| [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | 20 亿 | ~4.6 GB | ✅ 最佳质量 |
| [VoxCPM1.5](https://huggingface.co/openbmb/VoxCPM1.5) | 8 亿 | ~1.9 GB | 均衡之选 |
| [VoxCPM-0.5B](https://huggingface.co/openbmb/VoxCPM-0.5B) | 6.4 亿 | ~1.5 GB | 轻量级 |

#### 方法一：从 HuggingFace 下载（推荐）

```bash
huggingface-cli download openbmb/VoxCPM2 --local-dir ComfyUI/models/voxcpm/VoxCPM2
```

#### 方法二：从 ModelScope 下载（国内用户推荐）

```bash
pip install modelscope
modelscope download --model openbmb/VoxCPM2 --local_dir ComfyUI/models/voxcpm/VoxCPM2
```

### 模型目录结构

```
ComfyUI/
└── models/
    └── voxcpm/
        ├── VoxCPM2/                # 主模型（必需）
        │   ├── config.json
        │   ├── model.safetensors
        │   ├── audiovae.pth
        │   ├── tokenizer.json
        │   ├── tokenizer_config.json
        │   └── special_tokens_map.json
        ├── loras/                  # LoRA 权重（可选）
        │   └── my_custom_voice.pth
        └── speech_zipenhancer_ans_multiloss_16k_base/  # 降噪模型（可选）
```

### SenseVoiceSmall（自动 ASR 必需）

```bash
# 从 ModelScope 下载
modelscope download --model iic/SenseVoiceSmall --local_dir ComfyUI/models/SenseVoice/SenseVoiceSmall
```

### ZipEnhancer（可选，用于参考音频降噪）

```bash
# 从 ModelScope 下载
modelscope download --model iic/speech_zipenhancer_ans_multiloss_16k_base --local_dir ComfyUI/models/voxcpm/speech_zipenhancer_ans_multiloss_16k_base
```

## 🚀 使用方法

### 示例工作流

从 [`examples/`](examples/) 目录下载示例工作流并导入 ComfyUI：

1. **[声音设计](examples/voxcpm_voice_design.json)** — 通过文字描述创造声音
2. **[极致克隆](examples/voxcpm_ultimate_clone.json)** — 从参考音频克隆声音

### 三种模式

- **声音设计**：填写 `control_instruction`（如"一个温柔的年轻女性"），不上传 `reference_audio`。模型仅根据文字描述从零创造一个全新的声音。
- **可控克隆**：上传 `reference_audio`，保持 `ultimate_clone` 关闭。通过 `control_instruction` 控制情感、语速和风格，同时保留参考音频的音色。
- **极致克隆**：上传 `reference_audio`，开启 `ultimate_clone`。模型将参考音频视为已说出的前缀并从中续写，忠实复刻每一个声音细节。此模式下 `control_instruction` 会被忽略。若 `reference_audio_text` 为空，将自动进行 ASR 识别。

## 📝 节点参考

### RunningHub VoxCPM Load Model（加载模型）

从本地目录加载 VoxCPM/VoxCPM2 模型，可选加载 LoRA 权重。

| 参数 | 类型 | 说明 |
|------|------|------|
| model_name | COMBO | `models/voxcpm/` 下的模型目录 |
| optimize | BOOLEAN | 启用 torch.compile 优化（默认：关） |
| lora_name | COMBO | `models/voxcpm/loras/` 下的 LoRA 权重（可选，默认：None） |

### RunningHub VoxCPM Generate Speech（语音生成）

支持声音设计、可控克隆、极致克隆三种模式。

| 参数 | 类型 | 说明 |
|------|------|------|
| model | VOXCPM_MODEL | 来自 Load Model 节点的模型 |
| text | STRING | 要合成的目标文本 |
| cfg_value | FLOAT | 引导强度（默认：2.0） |
| inference_steps | INT | LocDiT 流匹配步数（默认：10） |
| seed | INT | 随机种子 |
| control_instruction | STRING | 声音描述，用于声音设计模式（可选） |
| reference_audio | AUDIO | 参考音频，用于克隆模式（可选） |
| ultimate_clone | BOOLEAN | 启用极致克隆模式（默认：关） |
| reference_audio_text | STRING | 参考音频的文字内容；为空时自动 ASR 识别（可选） |
| normalize_text | BOOLEAN | 文本规范化（默认：关） |
| denoise_reference | BOOLEAN | 通过 ZipEnhancer 对参考音频降噪（默认：关） |
| max_len | INT | 生成时最大 token 长度（默认：4096） |
| retry_badcase | BOOLEAN | 输出质量差时自动重试（默认：开） |

## 📄 许可证

本项目基于 [Apache License 2.0](LICENSE) 许可证开源。

## 🔗 相关链接

- [RunningHub](https://www.runninghub.cn)
- [VoxCPM（原始项目）](https://github.com/OpenBMB/VoxCPM)
- [VoxCPM2 on HuggingFace](https://huggingface.co/openbmb/VoxCPM2)

## 🙏 致谢

本项目基于 [VoxCPM](https://github.com/OpenBMB/VoxCPM)，由 [OpenBMB](https://github.com/OpenBMB) / [面壁智能](https://modelbest.cn) 开发。
