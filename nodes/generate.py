import logging
import os
import tempfile

import numpy as np
import torch
import torchaudio
import soundfile as sf
import folder_paths
import comfy.utils

logger = logging.getLogger("RunningHub.VoxCPM")


_CONTROL_PAREN_TRANS = str.maketrans({
    "(": "",
    ")": "",
    "（": "",
    "）": "",
})


def _sanitize_control(text):
    """Strip parentheses from a voice-design control instruction.

    VoxCPM expects the format ``(description)text``. If ``description`` itself
    contains ``(`` / ``)`` (or the full-width equivalents ``（`` / ``）``),
    the resulting prompt has unbalanced parentheses such as
    ``(INTP (逻辑学家)：26岁女性...)文本``. The model then treats part of the
    description as text and reads it aloud.

    We remove only the bracket characters and keep their inner content, so the
    description still reads naturally.
    """
    if not text:
        return ""
    cleaned = text.translate(_CONTROL_PAREN_TRANS)
    # Collapse runs of whitespace introduced by the removal.
    return " ".join(cleaned.split()).strip()


def _safe_save_wav(path, waveform, sample_rate):
    """Save waveform tensor to WAV, falling back to soundfile if torchaudio fails."""
    try:
        torchaudio.save(path, waveform, sample_rate)
    except Exception as e:
        logger.warning("torchaudio.save failed (%s), falling back to soundfile", e)
        # soundfile expects (samples, channels) numpy array
        audio_np = waveform.numpy().T
        sf.write(path, audio_np, sample_rate)

SENSEVOICE_MODEL_TYPE = "SenseVoice"
folder_paths.add_model_folder_path(
    SENSEVOICE_MODEL_TYPE,
    os.path.join(folder_paths.models_dir, SENSEVOICE_MODEL_TYPE),
)

VOXCPM_MODEL_TYPE = "voxcpm"
ZIPENHANCER_DIR_NAME = "speech_zipenhancer_ans_multiloss_16k_base"

def _get_asr_model():
    from funasr import AutoModel

    base_dirs = folder_paths.get_folder_paths(SENSEVOICE_MODEL_TYPE)
    model_path = None
    for base in base_dirs:
        candidate = os.path.join(base, "SenseVoiceSmall")
        if os.path.isdir(candidate):
            model_path = candidate
            break

    if model_path is None:
        raise FileNotFoundError(
            "SenseVoiceSmall model not found. "
            "Please place it under: models/SenseVoice/SenseVoiceSmall/"
        )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return AutoModel(
        model=model_path,
        disable_update=True,
        device=device,
    )


def _recognize_audio(wav_path):
    """Run ASR on audio file, return recognized text."""
    asr = _get_asr_model()
    res = asr.generate(input=wav_path, language="auto", use_itn=True)
    return res[0]["text"].split("|>")[-1]


def _get_denoiser():
    from voxcpm.zipenhancer import ZipEnhancer

    base_dirs = folder_paths.get_folder_paths(VOXCPM_MODEL_TYPE)
    model_path = None
    for base in base_dirs:
        candidate = os.path.join(base, ZIPENHANCER_DIR_NAME)
        if os.path.isdir(candidate):
            model_path = candidate
            break

    if model_path is None:
        expected = os.path.join(folder_paths.models_dir, VOXCPM_MODEL_TYPE, ZIPENHANCER_DIR_NAME)
        raise FileNotFoundError(
            f"ZipEnhancer model not found. "
            f"Please download from ModelScope 'iic/speech_zipenhancer_ans_multiloss_16k_base' "
            f"and place it at: {expected}"
        )

    return ZipEnhancer(model_path)


def _denoise_audio(input_path):
    """Denoise audio file, return path to denoised file."""
    denoiser = _get_denoiser()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    denoiser.enhance(input_path, output_path=tmp.name)
    return tmp.name


class RunningHubVoxCPMGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("VOXCPM_MODEL",),
                "control_instruction": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "text": ("STRING", {
                    "default": "Hello, this is a test.",
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
            "optional": {
                "reference_audio": ("AUDIO",),
                "ultimate_clone": ("BOOLEAN", {"default": False}),
                "reference_audio_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "normalize_text": ("BOOLEAN", {"default": False}),
                "denoise_reference": ("BOOLEAN", {"default": False}),
                "max_len": ("INT", {
                    "default": 4096,
                    "min": 64,
                    "max": 8192,
                    "step": 64,
                }),
                "retry_badcase": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "RunningHub/VoxCPM"

    def generate(
        self,
        model,
        control_instruction,
        text,
        cfg_value,
        inference_steps,
        seed,
        reference_audio=None,
        ultimate_clone=False,
        reference_audio_text="",
        normalize_text=False,
        denoise_reference=False,
        max_len=4096,
        retry_badcase=True,
    ):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        voxcpm_model = model["model"]
        sample_rate = model["sample_rate"]
        is_v2 = model["architecture"] == "voxcpm2"

        text = text.strip()
        if not text:
            raise ValueError("Target text must not be empty.")

        control = _sanitize_control((control_instruction or "").strip())

        # ultimate_clone=ON  → 极致克隆：用 prompt_text 做音频续写，control_instruction 被忽略
        # ultimate_clone=OFF → 声音设计 / 可控克隆：control_instruction 拼入文本
        if ultimate_clone:
            ref_text = (reference_audio_text or "").strip() or None
            final_text = text
        else:
            ref_text = None
            final_text = f"({control}){text}" if control else text

        ref_wav_path = None
        temp_files = []

        try:
            if reference_audio is not None:
                waveform = reference_audio["waveform"]
                sr = reference_audio["sample_rate"]

                if waveform.dim() == 3:
                    waveform = waveform.squeeze(0)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                temp_files.append(tmp.name)
                tmp.close()
                _safe_save_wav(tmp.name, waveform.cpu(), sr)
                ref_wav_path = tmp.name

                if denoise_reference:
                    logger.info("Denoising reference audio...")
                    denoised_path = _denoise_audio(ref_wav_path)
                    temp_files.append(denoised_path)
                    ref_wav_path = denoised_path

                if ultimate_clone and ref_text is None:
                    logger.info("Ultimate clone: reference_audio_text is empty, running ASR...")
                    ref_text = _recognize_audio(ref_wav_path)
                    logger.info("ASR result: %s...", ref_text[:80])

            total_steps = int(inference_steps) + 2
            pbar = comfy.utils.ProgressBar(total_steps)

            generate_kwargs = {
                "text": final_text,
                "cfg_value": float(cfg_value),
                "inference_timesteps": int(inference_steps),
                "normalize": normalize_text,
                "denoise": False,
                "max_len": int(max_len),
                "retry_badcase": retry_badcase,
            }

            if ref_wav_path is not None:
                if ultimate_clone and ref_text and is_v2:
                    generate_kwargs["prompt_wav_path"] = ref_wav_path
                    generate_kwargs["prompt_text"] = ref_text
                    generate_kwargs["reference_wav_path"] = ref_wav_path
                elif is_v2:
                    generate_kwargs["reference_wav_path"] = ref_wav_path
                else:
                    if ultimate_clone and ref_text:
                        generate_kwargs["prompt_wav_path"] = ref_wav_path
                        generate_kwargs["prompt_text"] = ref_text

            pbar.update(1)

            wav_np = voxcpm_model.generate(**generate_kwargs)
            if not isinstance(wav_np, np.ndarray):
                wav_np = np.asarray(wav_np, dtype=np.float32)
            wav_np = wav_np.astype(np.float32, copy=False).reshape(-1)
            pbar.update_absolute(total_steps - 1, total_steps)

            wav_tensor = torch.from_numpy(wav_np).float().unsqueeze(0).unsqueeze(0)

            audio_output = {
                "waveform": wav_tensor,
                "sample_rate": sample_rate,
            }

            pbar.update_absolute(total_steps, total_steps)
            return (audio_output,)

        finally:
            for tmp_path in temp_files:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
