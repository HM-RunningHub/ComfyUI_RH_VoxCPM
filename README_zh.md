# ComfyUI_RH_VoxCPM

![License](https://img.shields.io/badge/License-Apache%202.0-green)

[English](README.md)

[VoxCPM](https://github.com/OpenBMB/VoxCPM) 的 ComfyUI 自定义节点 — 无分词器 TTS，支持上下文感知语音生成与高保真声音克隆。

在线使用：[RunningHub 国内版](https://www.runninghub.cn/?inviteCode=rh-v1367) | [RunningHub 国际版](https://www.runninghub.ai/?inviteCode=rh-v1367)

GitHub 仓库地址：[HM-RunningHub/ComfyUI_RH_VoxCPM](https://github.com/HM-RunningHub/ComfyUI_RH_VoxCPM)

## ✨ 功能特性

- **声音设计**：通过文字描述创造全新声音（性别、年龄、语调、情感、语速）
- **可控克隆**：上传参考音频克隆音色，同时可用文字指令控制风格
- **极致克隆**：以音频续写方式复刻每一个声音细节（仅 VoxCPM2 支持）
- **LoRA 微调**：加载自定义 LoRA 权重，实现个性化语音生成
- **LoRA/全量训练**：直接在 ComfyUI 工作流中训练 VoxCPM LoRA 或全量微调（复用原项目的训练循环）
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

1. **[基础工作流](examples/VoxCPM2%20基础工作流.json)** — 单人语音生成，支持声音设计 / 克隆
2. **[多人工作流](examples/VoxCPM2%20多人工作流.json)** — 固定 5 人版本的多说话人对话生成，每位说话人可独立控制声音
3. **[LoRA 训练工作流](examples/VoxCPM2%20LoRA%20训练工作流.json)** — 从两段音频构建迷你数据集并执行 LoRA 微调

说明：

- `RunningHub VoxCPM Multi-Speaker` 是固定 5 人输入版本
- `RunningHub VoxCPM Multi-Speaker (Dynamic Audio)` 是动态音频输入版本，脚本格式相同，但参考音频输入会自动增长
- 如果插件更新后没有看到动态输入效果，请刷新 ComfyUI 前端页面或重新打开工作流

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

### RunningHub VoxCPM Multi-Speaker（多人语音）

根据带标签的脚本生成多说话人对话，最多支持 5 位说话人，每人可独立控制声音。

| 参数 | 类型 | 说明 |
|------|------|------|
| model | VOXCPM_MODEL | 来自 Load Model 节点的模型 |
| script | STRING | 带标签的脚本，如 `[spk1]你好[spk2]你好啊` |
| cfg_value | FLOAT | 引导强度（默认：2.0） |
| inference_steps | INT | LocDiT 流匹配步数（默认：10） |
| seed | INT | 随机种子 |
| audio_1 ~ audio_5 | AUDIO | 各说话人的参考音频（可选） |
| control_1 ~ control_5 | STRING | 各说话人的声音描述（可选） |
| normalize_text | BOOLEAN | 文本规范化（默认：关） |
| denoise_reference | BOOLEAN | 通过 ZipEnhancer 对参考音频降噪（默认：关） |
| max_len | INT | 生成时最大 token 长度（默认：4096） |
| retry_badcase | BOOLEAN | 输出质量差时自动重试（默认：开） |

### RunningHub VoxCPM Multi-Speaker (Dynamic Audio)（多人语音，动态参考音频）

适用于多说话人参考音频场景。脚本仍然使用 `[spk1]...[spk2]...` 标签；控制指令合并为一个多行文本输入，同样用标签区分不同说话人。节点默认显示 2 个参考音频输入；当当前输入全部接满时，会自动新增下一个输入，不设上限。执行时会按槽位编号把 `audio_1` 映射到 `spk1`、`audio_2` 映射到 `spk2`，依此类推，因此也支持 `spk10`、`spk20` 这类标签。

使用提示：

- 先接满当前可见的 `audio_*` 输入，节点才会自动长出下一个输入
- 这个“自动增长”依赖前端扩展脚本；更新插件后若行为未变化，请刷新页面

| 参数 | 类型 | 说明 |
|------|------|------|
| model | VOXCPM_MODEL | 来自 Load Model 节点的模型 |
| script | STRING | 带标签的脚本，如 `[spk1]你好[spk2]你好啊` |
| speaker_controls | STRING | 多行控制文本，如 `[spk1]四川话\n[spk2]成年女性，东北话` |
| cfg_value | FLOAT | 引导强度（默认：2.0） |
| inference_steps | INT | LocDiT 流匹配步数（默认：10） |
| seed | INT | 随机种子 |
| audio_1 ~ audio_N | AUDIO | 动态参考音频输入，按槽位顺序映射到 `spk1 ~ spkN`；默认显示 2 个，接满后自动增加，无上限 |
| normalize_text | BOOLEAN | 文本规范化（默认：关） |
| denoise_reference | BOOLEAN | 通过 ZipEnhancer 对参考音频降噪（默认：关） |
| max_len | INT | 生成时最大 token 长度（默认：4096） |
| retry_badcase | BOOLEAN | 输出质量差时自动重试（默认：开） |

## 🎓 训练节点（LoRA / 全量微调）

> ⚠️ 训练节点依赖 VoxCPM 原项目的训练代码（`voxcpm.training.*`）。安装插件时会随 `requirements.txt` 自动拉取 `transformers / datasets / safetensors / argbind` 等依赖；另需在 `ComfyUI/custom_nodes/VoxCPM/src/` 或插件目录下 `voxcpm/src/` 放一份 [VoxCPM](https://github.com/OpenBMB/VoxCPM) 源码（包含 `voxcpm/training/`）。

训练流程通常分三步：
1. 用 **Dataset Entry** 把单条（音频 + 文本）包装成训练样本；
2. 用 **Dataset Build** 把若干样本聚合为 `train.jsonl` 训练清单；也可以直接提供已有的 jsonl 文件路径；
3. 用 **Train LoRA** 或 **Train Full** 执行训练，产物默认写入 `ComfyUI/output/voxcpm_train/<name>_<timestamp>/`。启用 `copy_to_loras_dir` 后 LoRA 会自动拷贝到 `ComfyUI/models/voxcpm/loras/`，刷新页面即可在 Load Model 节点里直接选用。

### RunningHub VoxCPM Dataset Entry（构造训练样本）

| 参数 | 类型 | 说明 |
|------|------|------|
| audio | AUDIO | 单条训练音频 |
| text | STRING | 对应的文本转写 |
| dataset_id | INT | 可选：多数据集训练时的数据集编号（默认 0） |
| ref_audio | AUDIO | 可选：声音风格参考音频；填入后会被写入 manifest 的 `ref_audio` 字段，训练时用于条件输入（要求 voxcpm ≥ 2026-04 构建） |

### RunningHub VoxCPM Dataset Build（构建训练清单）

| 参数 | 类型 | 说明 |
|------|------|------|
| entry_1, entry_2 | VOXCPM_DATA_ENTRY | 至少 2 条样本 |
| entry_3 ~ entry_8 | VOXCPM_DATA_ENTRY | 更多样本（可选） |
| extra_manifest | STRING | 已有 jsonl 清单路径，将被追加到末尾（可选） |
| sample_rate | INT | 写入 wav 的采样率，建议与模型 AudioVAE 一致（默认 16000） |
| dataset_name | STRING | 输出目录名前缀 |

输出：`manifest_path` 指向生成的 `train.jsonl`，`num_samples` 为样本总数。

### RunningHub VoxCPM Train LoRA（LoRA 微调）

| 参数 | 类型 | 说明 |
|------|------|------|
| model_name | COMBO | 基底模型目录（`models/voxcpm/` 下） |
| train_manifest | STRING | 训练清单 jsonl 路径（可用 Dataset Build 的输出） |
| output_name | STRING | 输出名前缀，最终目录带时间戳 |
| num_iters | INT | 训练总步数（默认 500） |
| batch_size | INT | 单步 batch 大小（默认 1） |
| grad_accum_steps | INT | 梯度累积步数（默认 1） |
| learning_rate | FLOAT | 学习率（默认 1e-4） |
| lora_rank | INT | LoRA 秩（默认 32） |
| lora_alpha | INT | LoRA alpha（默认 32） |
| val_manifest | STRING | 验证集清单（可选） |
| warmup_steps | INT | warmup 步数（默认 100） |
| weight_decay | FLOAT | 权重衰减（默认 0.01） |
| max_grad_norm | FLOAT | 梯度裁剪上限，0 关闭（默认 1.0） |
| num_workers | INT | 数据加载线程数（默认 2） |
| log_interval | INT | 日志打印间隔步数（默认 10） |
| save_interval | INT | 检查点保存间隔步数；0 表示只在结束时保存（默认 0） |
| lora_dropout | FLOAT | LoRA dropout（默认 0.0） |
| enable_lm | BOOLEAN | 对语言模型部分启用 LoRA（默认 开） |
| enable_dit | BOOLEAN | 对 DiT 部分启用 LoRA（默认 开） |
| enable_proj | BOOLEAN | 对投影层启用 LoRA（默认 关） |
| copy_to_loras_dir | BOOLEAN | 训练结束自动拷贝到 `models/voxcpm/loras/`（默认 开） |

输出：`lora_path`（LoRA 权重目录，含 `lora_weights.safetensors` + `lora_config.json`）、`info`（训练摘要）。

### RunningHub VoxCPM Train Full（全量微调）

参数与 Train LoRA 类似，但不含 LoRA 相关项。⚠️ 全量微调显存/存储开销极大，建议仅在确有必要时使用；日常声音适配请优先使用 LoRA。

## 📄 许可证

本项目基于 [Apache License 2.0](LICENSE) 许可证开源。

## 🔗 相关链接

- [RunningHub](https://www.runninghub.cn)
- [VoxCPM（原始项目）](https://github.com/OpenBMB/VoxCPM)
- [VoxCPM2 on HuggingFace](https://huggingface.co/openbmb/VoxCPM2)

## 🙏 致谢

本项目基于 [VoxCPM](https://github.com/OpenBMB/VoxCPM)，由 [OpenBMB](https://github.com/OpenBMB) / [面壁智能](https://modelbest.cn) 开发。
