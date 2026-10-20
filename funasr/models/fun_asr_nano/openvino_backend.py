import logging
import os
import random
import re
import string
import time

import numpy as np
import torch
import torch.nn as nn

from funasr.utils.load_utils import load_audio_text_image_video

logger = logging.getLogger(__name__)

_LANGUAGE_PROMPTS = {
    "zh": "中文",
    "zh-cn": "中文",
    "chinese": "中文",
    "en": "英文",
    "english": "英文",
    "ja": "日文",
    "japanese": "日文",
}


class FunASRNanoOpenVINO(nn.Module):
    """FunASR AutoModel adapter backed by Optimum Intel OpenVINO."""

    skip_init_param = True

    def __init__(self, **kwargs):
        super().__init__()
        model_dir = kwargs.get("openvino_model_dir")
        source_dir = kwargs.get("openvino_source_model_dir") or kwargs.get("model_path")
        if not model_dir:
            raise ValueError("backend='openvino' requires openvino_model_dir")
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"OpenVINO model directory not found: {model_dir}")
        if not source_dir or not os.path.isdir(source_dir):
            raise FileNotFoundError(f"Fun-ASR-Nano source directory not found: {source_dir}")

        from optimum.intel import OVModelForSpeechSeq2Seq
        from transformers import AutoTokenizer

        openvino_device = kwargs.get("openvino_device", "CPU")
        openvino_config = kwargs.get("openvino_config")
        use_mixed_gpu_precision = (
            openvino_config is None and str(openvino_device).upper().startswith("GPU")
        )
        if use_mixed_gpu_precision:
            self.__dict__["_ov_model"] = OVModelForSpeechSeq2Seq.from_pretrained(
                model_dir,
                device=openvino_device,
                ov_config={"INFERENCE_PRECISION_HINT": "f16"},
                compile=False,
            )
            for component_name, component in self._ov_model.components.items():
                component.ov_config = {
                    **component.ov_config,
                    "INFERENCE_PRECISION_HINT": "f32" if component_name.startswith("decoder") else "f16",
                }
            self._ov_model.compile()
        else:
            self.__dict__["_ov_model"] = OVModelForSpeechSeq2Seq.from_pretrained(
                model_dir,
                device=openvino_device,
                ov_config=openvino_config,
            )
        self.skip_special_tokens = kwargs.get("skip_special_tokens", True)
        self.__dict__["_ov_detokenizer"] = None
        detokenizer_path = os.path.join(model_dir, "openvino_detokenizer.xml")
        if os.path.isfile(detokenizer_path):
            import openvino_tokenizers  # noqa: F401
            from openvino import Core

            self.__dict__["_ov_detokenizer"] = Core().compile_model(detokenizer_path, "CPU")
            logger.info("Using OpenVINO detokenizer IR: %s", detokenizer_path)
        else:
            logger.info("OpenVINO detokenizer IR not found; using AutoTokenizer.decode()")
        if self._ov_detokenizer is None or not self.skip_special_tokens:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                trust_remote_code=True,
            )
        else:
            self.tokenizer = None
        self.max_new_tokens = int(kwargs.get("max_length", 512))
        self.model_path = source_dir

    def inference(
        self,
        data_in,
        data_lengths=None,
        key=None,
        frontend=None,
        **kwargs,
    ):
        if kwargs.get("data_type") == "fbank":
            raise NotImplementedError("OpenVINO Fun-ASR-Nano backend accepts waveform audio, not fbank")
        if kwargs.get("hotwords") or kwargs.get("hotword"):
            raise NotImplementedError(
                "FunASR Nano hotwords are not yet supported by the Optimum Intel OpenVINO API"
            )

        if not isinstance(data_in, (list, tuple)):
            data_in = [data_in]
        if key is None:
            key = [
                "rand_key_" + "".join(random.choice(string.ascii_letters + string.digits) for _ in range(13))
                for _ in data_in
            ]
        elif isinstance(key, str):
            key = [key]

        target_fs = int(getattr(frontend, "fs", 16000))
        source_fs = int(kwargs.get("fs", target_fs))
        language = kwargs.get("language")
        language = _LANGUAGE_PROMPTS.get(str(language).lower(), language or "中文")
        use_itn = kwargs.get("itn", True)
        max_new_tokens = int(kwargs.get("max_length", self.max_new_tokens))
        generation_kwargs = dict(kwargs.get("llm_kwargs") or {})
        generation_kwargs["max_new_tokens"] = max_new_tokens

        results = []
        audio_duration = 0.0
        start_time = time.perf_counter()
        for index, item in enumerate(data_in):
            waveform = load_audio_text_image_video(
                item,
                fs=target_fs,
                audio_fs=source_fs,
                data_type="sound",
            )
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.detach().cpu().numpy()
            waveform = np.asarray(waveform, dtype=np.float32)
            if waveform.ndim == 2:
                channel_axis = 0 if waveform.shape[0] <= 8 else 1
                waveform = waveform.mean(axis=channel_axis)
            if waveform.ndim != 1 or waveform.size == 0:
                raise ValueError("OpenVINO Fun-ASR-Nano expects nonempty mono waveform audio")

            inputs = self._ov_model.preprocess_input(
                waveform,
                target_fs,
                language=language,
                itn=use_itn,
            )
            generated = self._ov_model.generate(**inputs, **generation_kwargs)
            if hasattr(generated, "sequences"):
                generated = generated.sequences
            prompt_length = inputs["decoder_input_ids"].shape[1]
            token_ids = generated[0][prompt_length:].detach().cpu().tolist()
            if self._ov_detokenizer is not None and self.skip_special_tokens:
                detokenizer_inputs = {
                    self._ov_detokenizer.input(0).get_any_name(): np.asarray(
                        [token_ids], dtype=np.int64
                    )
                }
                detokenizer_output = self._ov_detokenizer(detokenizer_inputs)[
                    self._ov_detokenizer.output(0)
                ][0]
                text = (
                    detokenizer_output.decode("utf-8")
                    if isinstance(detokenizer_output, bytes)
                    else str(detokenizer_output)
                )
            else:
                text = self.tokenizer.decode(
                    token_ids,
                    skip_special_tokens=self.skip_special_tokens,
                )
            text = text.strip()
            key_i = key[index] if index < len(key) else f"rand_{index}"
            results.append(
                {
                    "key": key_i,
                    "text": re.sub(r"\s+", " ", text.replace("/sil", " ")).strip(),
                    "text_tn": re.sub(r"[^\w\s\u3000\u4e00-\u9fff]+", "", text),
                }
            )
            audio_duration += waveform.size / target_fs

        elapsed = time.perf_counter() - start_time
        return results, {
            "batch_data_time": audio_duration,
            "load_data": 0.0,
            "extract_feat": 0.0,
            "forward": elapsed,
        }