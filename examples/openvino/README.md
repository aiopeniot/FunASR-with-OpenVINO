# Fun-ASR-Nano OpenVINO with FunASR AutoModel

This example uses FunASR's existing `AutoModel.generate()` API. `backend="openvino"` selects Optimum Intel for Nano ASR inference; FunASR continues to handle audio loading, optional VAD, CAM++ speaker clustering, and result assembly.

## Model Conversion

Optimum Intel detects the FunASR Nano checkpoint and exports its audio encoder/adaptor and Qwen3 decoder as OpenVINO IR. The decoder IR includes the stateful KV-cache path used for autoregressive generation. It also exports tokenizer and detokenizer IRs. The current FunASR adapter uses the tokenizer IR for prompt encoding and prefers the detokenizer IR for generated token IDs; it falls back to Transformers decoding if the detokenizer IR is absent.

The source snapshot used here is `examples/openvino/models/source/models/FunAudioLLM--Fun-ASR-Nano-2512/snapshots/master/`. The validated export environment was Python 3.10, FunASR 1.4.16, Optimum Intel 2.2.0, OpenVINO 2026.4.1, `openvino-tokenizers` 2026.4.1.0, and Transformers 4.57.6. Keep Transformers/OpenVINO versions aligned between conversion and inference, and use the patched FunASR checkout so the AutoModel backend is available.

Export from the repository root into a fresh output directory:

```bash
python -m pip install -e .
python -m pip install \
  "optimum-intel==2.2.0" \
  "openvino==2026.4.1" \
  "openvino-tokenizers==2026.4.1.0" \
  "transformers==4.57.6"

optimum-cli export openvino \
  -m examples/openvino/models/source/models/FunAudioLLM--Fun-ASR-Nano-2512/snapshots/master \
  examples/openvino/models/openvino/nano \
  --trust-remote-code
```

The output should contain `openvino_encoder_model.xml`, `openvino_decoder_model.xml`, `openvino_tokenizer.xml`, and `openvino_detokenizer.xml` with their corresponding `.bin` weights. Record the source model revision and hashes alongside the exported files.

## Use AutoModel

```python
from funasr import AutoModel

model = AutoModel(
  model="examples/openvino/models/source/models/FunAudioLLM--Fun-ASR-Nano-2512/snapshots/master",
    backend="openvino",
  openvino_model_dir="examples/openvino/models/openvino/nano",
    openvino_device="CPU",
    device="cpu",
    vad_model="fsmn-vad",
    spk_model="cam++",
)
result = model.generate(input="meeting.wav", language="zh")
print(result[0]["text"])
print(result[0].get("sentence_info", []))
```

`device` controls FunASR's PyTorch VAD/CAM++ components; `openvino_device` controls Nano inference. A hub ID can be passed as `model`; FunASR downloads the source snapshot for configuration and skips loading its PyTorch checkpoint. `openvino_source_model_dir` may be used to point at an existing local snapshot explicitly.

OpenVINO Nano currently returns transcript text without CTC forced-alignment timestamps or model hotwords. With `spk_model`, the existing AutoModel pipeline assigns speakers to VAD segments; this is not word-level speaker alignment.

## Smoke Test

```bash
python examples/openvino/smoke_test.py \
  --model examples/openvino/models/source/models/FunAudioLLM--Fun-ASR-Nano-2512/snapshots/master \
  --openvino-model-dir examples/openvino/models/openvino/nano \
  --audio examples/openvino/testdata/three-speaker-conversation-en-croak.wav \
  --language en
```