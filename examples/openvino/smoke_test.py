import argparse
import json

from funasr import AutoModel


def main():
    parser = argparse.ArgumentParser(description="Run Fun-ASR-Nano through FunASR AutoModel/OpenVINO")
    parser.add_argument("--model", default="FunAudioLLM/Fun-ASR-Nano-2512")
    parser.add_argument("--openvino-model-dir", required=True)
    parser.add_argument("--openvino-source-model-dir")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--openvino-device", default="CPU")
    parser.add_argument("--vad-model", default="fsmn-vad")
    parser.add_argument("--spk-model", default="cam++")
    args = parser.parse_args()

    model = AutoModel(
        model=args.model,
        backend="openvino",
        openvino_model_dir=args.openvino_model_dir,
        openvino_source_model_dir=args.openvino_source_model_dir,
        openvino_device=args.openvino_device,
        device=args.device,
        vad_model=args.vad_model,
        spk_model=args.spk_model,
        disable_update=True,
    )
    result = model.generate(input=args.audio, language=args.language)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()