import sys
import types

import numpy as np
import torch
import yaml

from funasr import AutoModel
from funasr.models.fun_asr_nano.openvino_backend import FunASRNanoOpenVINO


class FakeOVModel:
    load_calls = []
    compile_calls = 0

    def __init__(self):
        self.preprocess_calls = []
        self.generate_calls = []
        self.components = {
            "encoder": types.SimpleNamespace(ov_config={}),
            "decoder": types.SimpleNamespace(ov_config={}),
        }

    @classmethod
    def from_pretrained(cls, model_dir, device, ov_config=None, **kwargs):
        cls.load_calls.append((model_dir, device, ov_config, kwargs))
        return cls()

    def compile(self):
        type(self).compile_calls += 1

    def preprocess_input(self, waveform, sampling_rate, language, itn):
        self.preprocess_calls.append((waveform.copy(), sampling_rate, language, itn))
        return {"decoder_input_ids": torch.tensor([[1, 2]])}

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return torch.tensor([[1, 2, 7, 8]])


class FakeTokenizer:
    @classmethod
    def from_pretrained(cls, model_dir, trust_remote_code):
        assert model_dir.endswith("openvino")
        assert trust_remote_code is True
        return cls()

    def decode(self, token_ids, skip_special_tokens):
        assert token_ids == [7, 8]
        assert skip_special_tokens is True
        return "hello world"


def test_openvino_backend_returns_fun_asr_inference_result(monkeypatch, tmp_path):
    FakeOVModel.load_calls.clear()
    model_dir = tmp_path / "openvino"
    source_dir = tmp_path / "source"
    model_dir.mkdir()
    source_dir.mkdir()

    optimum_module = types.ModuleType("optimum.intel")
    optimum_module.OVModelForSpeechSeq2Seq = FakeOVModel
    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoTokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "optimum", types.ModuleType("optimum"))
    monkeypatch.setitem(sys.modules, "optimum.intel", optimum_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    model = FunASRNanoOpenVINO(
        openvino_model_dir=str(model_dir),
        openvino_source_model_dir=str(source_dir),
        openvino_device="CPU",
    )
    results, metadata = model.inference(
        data_in=[np.ones(16000, dtype=np.float32)],
        key=["sample"],
        language="en",
        fs=16000,
    )

    assert results == [
        {"key": "sample", "text": "hello world", "text_tn": "hello world"}
    ]
    assert model._ov_model.preprocess_calls[0][1:] == (16000, "英文", True)
    assert model._ov_model.generate_calls[0]["max_new_tokens"] == 512
    assert metadata["batch_data_time"] == 1.0


def test_openvino_gpu_defaults_to_fp16_encoder_and_fp32_decoder(monkeypatch, tmp_path):
    model_dir = tmp_path / "openvino"
    source_dir = tmp_path / "source"
    model_dir.mkdir()
    source_dir.mkdir()
    optimum_module = types.ModuleType("optimum.intel")
    optimum_module.OVModelForSpeechSeq2Seq = FakeOVModel
    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoTokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "optimum", types.ModuleType("optimum"))
    monkeypatch.setitem(sys.modules, "optimum.intel", optimum_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    FakeOVModel.load_calls.clear()
    FakeOVModel.compile_calls = 0

    model = FunASRNanoOpenVINO(
        openvino_model_dir=str(model_dir),
        openvino_source_model_dir=str(source_dir),
        openvino_device="GPU.1",
    )

    assert FakeOVModel.load_calls == [
        (str(model_dir), "GPU.1", {"INFERENCE_PRECISION_HINT": "f16"}, {"compile": False})
    ]
    assert model._ov_model.components["encoder"].ov_config["INFERENCE_PRECISION_HINT"] == "f16"
    assert model._ov_model.components["decoder"].ov_config["INFERENCE_PRECISION_HINT"] == "f32"
    assert FakeOVModel.compile_calls == 1


def test_openvino_config_overrides_gpu_precision_default(monkeypatch, tmp_path):
    model_dir = tmp_path / "openvino"
    source_dir = tmp_path / "source"
    model_dir.mkdir()
    source_dir.mkdir()
    optimum_module = types.ModuleType("optimum.intel")
    optimum_module.OVModelForSpeechSeq2Seq = FakeOVModel
    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoTokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "optimum", types.ModuleType("optimum"))
    monkeypatch.setitem(sys.modules, "optimum.intel", optimum_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    FakeOVModel.load_calls.clear()
    config = {"INFERENCE_PRECISION_HINT": "f16"}

    FunASRNanoOpenVINO(
        openvino_model_dir=str(model_dir),
        openvino_source_model_dir=str(source_dir),
        openvino_device="GPU.1",
        openvino_config=config,
    )

    assert FakeOVModel.load_calls == [(str(model_dir), "GPU.1", config, {})]


def test_openvino_backend_uses_detokenizer_ir_when_present(monkeypatch, tmp_path):
    model_dir = tmp_path / "openvino"
    source_dir = tmp_path / "source"
    model_dir.mkdir()
    source_dir.mkdir()
    (model_dir / "openvino_detokenizer.xml").write_text("fake IR", encoding="utf-8")

    class NoFallbackTokenizer(FakeTokenizer):
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise AssertionError("AutoTokenizer should not load when detokenizer IR exists")

        def decode(self, *_args, **_kwargs):
            raise AssertionError("AutoTokenizer.decode should not run when detokenizer IR exists")

    class FakeTensorName:
        def __init__(self, name):
            self.name = name

        def get_any_name(self):
            return self.name

    class FakeDetokenizer:
        def input(self, _index):
            return FakeTensorName("input_ids")

        def output(self, _index):
            return "decoded_text"

        def __call__(self, inputs):
            assert inputs["input_ids"].tolist() == [[7, 8]]
            return {"decoded_text": np.asarray(["OpenVINO decoded"])}

    class FakeCore:
        def compile_model(self, path, device):
            assert path.endswith("openvino_detokenizer.xml")
            assert device == "CPU"
            return FakeDetokenizer()

    optimum_module = types.ModuleType("optimum.intel")
    optimum_module.OVModelForSpeechSeq2Seq = FakeOVModel
    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoTokenizer = NoFallbackTokenizer
    openvino_module = types.ModuleType("openvino")
    openvino_module.Core = FakeCore
    monkeypatch.setitem(sys.modules, "optimum", types.ModuleType("optimum"))
    monkeypatch.setitem(sys.modules, "optimum.intel", optimum_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "openvino", openvino_module)
    monkeypatch.setitem(sys.modules, "openvino_tokenizers", types.ModuleType("openvino_tokenizers"))

    model = FunASRNanoOpenVINO(
        openvino_model_dir=str(model_dir),
        openvino_source_model_dir=str(source_dir),
    )
    results, _ = model.inference(
        data_in=[np.ones(16000, dtype=np.float32)],
        key=["sample"],
        fs=16000,
    )

    assert results[0]["text"] == "OpenVINO decoded"


def test_openvino_backend_rejects_unsupported_hotwords():
    model = object.__new__(FunASRNanoOpenVINO)

    try:
        model.inference(data_in=[np.ones(10, dtype=np.float32)], hotwords=["name"])
    except NotImplementedError as error:
        assert "hotwords" in str(error)
    else:
        raise AssertionError("OpenVINO backend must not silently ignore Nano hotwords")


def test_auto_model_openvino_backend_uses_local_config_without_loading_checkpoint(
    monkeypatch, tmp_path
):
    source_dir = tmp_path / "nano-source"
    model_dir = tmp_path / "openvino"
    source_dir.mkdir()
    model_dir.mkdir()
    checkpoint = source_dir / "model.pt"
    checkpoint.write_bytes(b"not a torch checkpoint")
    (source_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": "FunASRNano",
                "model_conf": {},
                "frontend": None,
                "frontend_conf": {},
                "tokenizer": None,
                "init_param": str(checkpoint),
            }
        ),
        encoding="utf-8",
    )

    class LocalFakeOVModel:
        @classmethod
        def from_pretrained(cls, _model_dir, device, ov_config=None):
            assert device == "CPU"
            return cls()

    optimum_module = types.ModuleType("optimum.intel")
    optimum_module.OVModelForSpeechSeq2Seq = LocalFakeOVModel
    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoTokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "optimum", types.ModuleType("optimum"))
    monkeypatch.setitem(sys.modules, "optimum.intel", optimum_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    from funasr.auto import auto_model as auto_model_module

    def checkpoint_load_must_not_run(**_kwargs):
        raise AssertionError("OpenVINO backend should not load the PyTorch checkpoint")

    monkeypatch.setattr(auto_model_module, "load_pretrained_model", checkpoint_load_must_not_run)
    model, config = AutoModel.build_model(
        model="FunAudioLLM/Fun-ASR-Nano-2512",
        backend="openvino",
        openvino_source_model_dir=str(source_dir),
        openvino_model_dir=str(model_dir),
        device="cpu",
        disable_update=True,
    )

    assert isinstance(model, FunASRNanoOpenVINO)
    assert config["model"] == "FunASRNano"
    assert config["model_path"] == str(source_dir)


def test_auto_model_openvino_backend_downloads_source_for_hub_model_id(monkeypatch, tmp_path):
    source_dir = tmp_path / "nano-source"
    model_dir = tmp_path / "openvino"
    source_dir.mkdir()
    model_dir.mkdir()
    (source_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": "FunASRNano",
                "model_conf": {},
                "frontend": None,
                "frontend_conf": {},
                "tokenizer": None,
                "init_param": str(source_dir / "model.pt"),
            }
        ),
        encoding="utf-8",
    )

    class LocalFakeOVModel:
        @classmethod
        def from_pretrained(cls, _model_dir, device, ov_config=None):
            assert device == "CPU"
            return cls()

    optimum_module = types.ModuleType("optimum.intel")
    optimum_module.OVModelForSpeechSeq2Seq = LocalFakeOVModel
    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoTokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "optimum", types.ModuleType("optimum"))
    monkeypatch.setitem(sys.modules, "optimum.intel", optimum_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    from funasr.auto import auto_model as auto_model_module

    calls = []

    def fake_download_model(**kwargs):
        calls.append(kwargs["model"])
        return {**kwargs, "model_path": str(source_dir)}

    monkeypatch.setattr(auto_model_module, "download_model", fake_download_model)
    model, config = AutoModel.build_model(
        model="FunAudioLLM/Fun-ASR-Nano-2512",
        backend="openvino",
        openvino_model_dir=str(model_dir),
        device="cpu",
        disable_update=True,
    )

    assert calls == ["FunAudioLLM/Fun-ASR-Nano-2512"]
    assert isinstance(model, FunASRNanoOpenVINO)
    assert config["model_path"] == str(source_dir)