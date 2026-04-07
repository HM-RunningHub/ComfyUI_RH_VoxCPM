import os
import tempfile
import numpy as np
import torch
import torchaudio
import folder_paths


class VoxCPMGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("VOXCPM_MODEL",),
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
            },
            "optional": {
                "control_instruction": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "reference_audio": ("AUDIO",),
                "prompt_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "normalize_text": ("BOOLEAN", {"default": False}),
                "denoise_reference": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/voxcpm"

    def generate(
        self,
        model,
        text,
        cfg_value,
        inference_steps,
        control_instruction="",
        reference_audio=None,
        prompt_text="",
        normalize_text=False,
        denoise_reference=False,
    ):
        voxcpm_model = model["model"]
        sample_rate = model["sample_rate"]
        is_v2 = model["architecture"] == "voxcpm2"

        text = text.strip()
        if not text:
            raise ValueError("Target text must not be empty.")

        control = (control_instruction or "").strip()
        prompt_text_clean = (prompt_text or "").strip() or None

        if control and not prompt_text_clean:
            final_text = f"({control}){text}"
        else:
            final_text = text

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
                torchaudio.save(tmp.name, waveform.cpu(), sr)
                ref_wav_path = tmp.name

            generate_kwargs = {
                "text": final_text,
                "cfg_value": float(cfg_value),
                "inference_timesteps": int(inference_steps),
                "normalize": normalize_text,
                "denoise": denoise_reference,
            }

            if ref_wav_path is not None:
                if prompt_text_clean and is_v2:
                    generate_kwargs["prompt_wav_path"] = ref_wav_path
                    generate_kwargs["prompt_text"] = prompt_text_clean
                    generate_kwargs["reference_wav_path"] = ref_wav_path
                elif is_v2:
                    generate_kwargs["reference_wav_path"] = ref_wav_path
                else:
                    if prompt_text_clean:
                        generate_kwargs["prompt_wav_path"] = ref_wav_path
                        generate_kwargs["prompt_text"] = prompt_text_clean

            wav_np = voxcpm_model.generate(**generate_kwargs)

            wav_tensor = torch.from_numpy(wav_np).float().unsqueeze(0).unsqueeze(0)

            audio_output = {
                "waveform": wav_tensor,
                "sample_rate": sample_rate,
            }

            return (audio_output,)

        finally:
            for tmp_path in temp_files:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
