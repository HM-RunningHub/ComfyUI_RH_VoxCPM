import logging
import os
import re
import tempfile

import numpy as np
import torch
import torchaudio
import folder_paths
import comfy.utils

from .generate import _recognize_audio, _denoise_audio

logger = logging.getLogger("RunningHub.VoxCPM")

NUM_SPEAKERS = 5
TARGET_RMS = 0.08
RMS_FLOOR = 1e-6
SENTENCE_DELIMITERS = re.compile(r"(?<=[。！？.!?])")


def _normalize_rms(wav, target_rms=TARGET_RMS):
    """Normalize a 1-D numpy waveform to a target RMS level."""
    rms = np.sqrt(np.mean(wav ** 2))
    if rms < RMS_FLOOR:
        return wav
    return wav * (target_rms / rms)


def _split_by_sentences(wav, text, num_parts):
    """Split a waveform into num_parts roughly equal chunks by duration.

    Uses sentence boundaries in text to estimate proportional lengths.
    """
    if num_parts <= 1:
        return [wav]

    sentences = SENTENCE_DELIMITERS.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) < num_parts:
        chunk_size = len(wav) // num_parts
        parts = []
        for i in range(num_parts):
            start = i * chunk_size
            end = start + chunk_size if i < num_parts - 1 else len(wav)
            parts.append(wav[start:end])
        return parts

    # Distribute sentences into num_parts groups by character count
    total_chars = sum(len(s) for s in sentences)
    target_chars = total_chars / num_parts

    groups = []
    current_group = []
    current_chars = 0
    for s in sentences:
        current_group.append(s)
        current_chars += len(s)
        if current_chars >= target_chars and len(groups) < num_parts - 1:
            groups.append(current_chars)
            current_group = []
            current_chars = 0
    groups.append(current_chars)

    # Split wav proportionally
    total_group_chars = sum(groups)
    parts = []
    offset = 0
    for i, gc in enumerate(groups):
        if i == len(groups) - 1:
            parts.append(wav[offset:])
        else:
            length = int(len(wav) * gc / total_group_chars)
            parts.append(wav[offset:offset + length])
            offset += length
    return parts


def _save_audio_to_temp(audio_dict):
    """Save ComfyUI AUDIO dict to a temp wav file, return path."""
    waveform = audio_dict["waveform"]
    sr = audio_dict["sample_rate"]
    if waveform.dim() == 3:
        waveform = waveform.squeeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    torchaudio.save(tmp.name, waveform.cpu(), sr)
    return tmp.name


def _parse_script(text):
    """Parse multi-speaker script into ordered segments.

    Format: [spk1]Hello world[spk2]How are you?[spk1]I'm fine.
    Returns: [(1, "Hello world"), (2, "How are you?"), (1, "I'm fine.")]
    """
    pattern = re.compile(r"\[spk(\d)\]", re.IGNORECASE)
    parts = pattern.split(text)

    # parts = ['prefix_before_any_tag', '1', 'text1', '2', 'text2', ...]
    # If text starts with [spk1], parts[0] is empty string
    segments = []

    # Handle text before any tag — treat as spk1
    if parts[0].strip():
        segments.append((1, parts[0].strip()))

    for i in range(1, len(parts), 2):
        spk_idx = int(parts[i])
        if spk_idx < 1 or spk_idx > NUM_SPEAKERS:
            raise ValueError(f"Invalid speaker index: [spk{spk_idx}], must be 1-{NUM_SPEAKERS}")
        segment_text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if segment_text:
            segments.append((spk_idx, segment_text))

    if not segments:
        raise ValueError(
            "No valid segments found. Use [spk1]...[spk2]... format. "
            "Example: [spk1]Hello[spk2]Hi there"
        )
    return segments


class RunningHubVoxCPMMultiSpeaker:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "model": ("VOXCPM_MODEL",),
                "script": ("STRING", {
                    "default": "[spk5]今日AI新闻速报\n[spk1]T8那个瓜娃子，又更新什么了？\n[spk2]管他干啥呀，带派不就行了。\n[spk1]天天只知道做视频，耳都不耳人一哈。\n[spk2]别笑哈，你试你也过不了第二关。",
                    "multiline": True,
                }),
                "cfg_value": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                }),
                "inference_steps": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                }),
            },
            "optional": {},
        }

        control_defaults = {
            1: "四川话",
            2: "成年女性，东北话",
            3: "",
            4: "",
            5: "旁白音，成熟男性",
        }
        for i in range(1, NUM_SPEAKERS + 1):
            inputs["optional"][f"audio_{i}"] = ("AUDIO",)
            inputs["optional"][f"control_{i}"] = ("STRING", {
                "default": control_defaults.get(i, ""),
            })

        inputs["optional"]["normalize_text"] = ("BOOLEAN", {"default": False})
        inputs["optional"]["denoise_reference"] = ("BOOLEAN", {"default": False})
        inputs["optional"]["max_len"] = ("INT", {
            "default": 4096,
            "min": 64,
            "max": 8192,
            "step": 64,
        })
        inputs["optional"]["retry_badcase"] = ("BOOLEAN", {"default": True})

        return inputs

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "RunningHub/VoxCPM"

    def generate(self, model, script, cfg_value, inference_steps, seed, **kwargs):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        voxcpm_model = model["model"]
        sample_rate = model["sample_rate"]
        is_v2 = model["architecture"] == "voxcpm2"

        normalize_text = kwargs.get("normalize_text", False)
        denoise_reference = kwargs.get("denoise_reference", False)
        max_len = kwargs.get("max_len", 4096)
        retry_badcase = kwargs.get("retry_badcase", True)

        segments = _parse_script(script)
        logger.info("Parsed %d segments from script", len(segments))

        speaker_audios = {}
        speaker_controls = {}
        for i in range(1, NUM_SPEAKERS + 1):
            speaker_audios[i] = kwargs.get(f"audio_{i}")
            speaker_controls[i] = (kwargs.get(f"control_{i}") or "").strip()

        # Group segments by speaker: {spk_idx: [(original_index, text), ...]}
        spk_groups = {}
        for orig_idx, (spk_idx, seg_text) in enumerate(segments):
            spk_groups.setdefault(spk_idx, []).append((orig_idx, seg_text))

        unique_speakers = sorted(spk_groups.keys())
        total_steps = len(unique_speakers) * (int(inference_steps) + 2) + 1
        pbar = comfy.utils.ProgressBar(total_steps)
        step_counter = 0

        temp_files = []
        # Will hold (original_index, wav_np) pairs
        indexed_wavs = [None] * len(segments)

        try:
            spk_wav_cache = {}
            for spk_idx, audio in speaker_audios.items():
                if audio is not None:
                    wav_path = _save_audio_to_temp(audio)
                    temp_files.append(wav_path)
                    if denoise_reference:
                        denoised = _denoise_audio(wav_path)
                        temp_files.append(denoised)
                        wav_path = denoised
                    spk_wav_cache[spk_idx] = wav_path

            spk_asr_cache = {}

            for spk_idx in unique_speakers:
                group = spk_groups[spk_idx]
                texts = [t for _, t in group]
                combined_text = "\n".join(texts)

                ref_wav_path = spk_wav_cache.get(spk_idx)
                control = speaker_controls.get(spk_idx, "")
                has_control = bool(control)

                logger.info("Generating spk%d: %d segments, combined %d chars",
                            spk_idx, len(group), len(combined_text))

                if has_control:
                    final_text = f"({control}){combined_text}"
                    generate_kwargs = {
                        "text": final_text,
                        "cfg_value": float(cfg_value),
                        "inference_timesteps": int(inference_steps),
                        "normalize": normalize_text,
                        "denoise": False,
                        "max_len": int(max_len),
                        "retry_badcase": retry_badcase,
                    }
                    if ref_wav_path is not None and is_v2:
                        generate_kwargs["reference_wav_path"] = ref_wav_path
                else:
                    generate_kwargs = {
                        "text": combined_text,
                        "cfg_value": float(cfg_value),
                        "inference_timesteps": int(inference_steps),
                        "normalize": normalize_text,
                        "denoise": False,
                        "max_len": int(max_len),
                        "retry_badcase": retry_badcase,
                    }
                    if ref_wav_path is not None:
                        if is_v2:
                            if spk_idx not in spk_asr_cache:
                                logger.info("Running ASR for spk%d...", spk_idx)
                                spk_asr_cache[spk_idx] = _recognize_audio(ref_wav_path)
                                logger.info("ASR result for spk%d: %s...",
                                            spk_idx, spk_asr_cache[spk_idx][:60])
                            ref_text = spk_asr_cache[spk_idx]
                            generate_kwargs["prompt_wav_path"] = ref_wav_path
                            generate_kwargs["prompt_text"] = ref_text
                            generate_kwargs["reference_wav_path"] = ref_wav_path
                        else:
                            if spk_idx not in spk_asr_cache:
                                spk_asr_cache[spk_idx] = _recognize_audio(ref_wav_path)
                            ref_text = spk_asr_cache[spk_idx]
                            generate_kwargs["prompt_wav_path"] = ref_wav_path
                            generate_kwargs["prompt_text"] = ref_text

                step_counter += 1
                pbar.update_absolute(step_counter, total_steps)

                wav_np = voxcpm_model.generate(**generate_kwargs)

                step_counter += int(inference_steps) + 1
                pbar.update_absolute(step_counter, total_steps)

                # Split the combined waveform by character proportion
                char_lengths = [len(t) for t in texts]
                total_chars = sum(char_lengths)
                parts = []
                offset = 0
                for i, cl in enumerate(char_lengths):
                    if i == len(char_lengths) - 1:
                        parts.append(wav_np[offset:])
                    else:
                        length = int(len(wav_np) * cl / total_chars)
                        parts.append(wav_np[offset:offset + length])
                        offset += length

                for (orig_idx, _), part in zip(group, parts):
                    indexed_wavs[orig_idx] = _normalize_rms(part)

            # Reassemble in original script order
            final_parts = [w for w in indexed_wavs if w is not None]
            combined = np.concatenate(final_parts, axis=-1)
            peak = np.max(np.abs(combined))
            if peak > 0.99:
                combined = combined * (0.99 / peak)
            wav_tensor = torch.from_numpy(combined).float().unsqueeze(0).unsqueeze(0)

            pbar.update_absolute(total_steps, total_steps)
            return ({"waveform": wav_tensor, "sample_rate": sample_rate},)

        finally:
            for tmp_path in temp_files:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
