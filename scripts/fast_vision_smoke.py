"""Check one known object on CUDA; overwrite the latest smoke-test result"""

import hashlib
import json
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    fixture = root / "tests/fixtures/fast-vision"
    output = root / "results/fast-vision-smoke/result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)

    # Import after clearing the old result
    import torch
    import torchvision
    from PIL import Image, __version__ as pillow_version

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    expected = json.loads((fixture / "expected.json").read_text())
    image_path = fixture / expected["image"]
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()

    if image_sha256 != expected["image_sha256"]:
        raise ValueError("Input image does not match expected.json")
    
    with Image.open(image_path) as source:
        image = source.convert("RGB")

    weights = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
    model = model.eval().to("cuda:0")
    tensor = weights.transforms()(image).to("cuda:0")
    with torch.inference_mode():
        prediction = model([tensor])[0]

    # Blocking copies complete the outputs on this default-stream path
    prediction = {key: value.cpu() for key, value in prediction.items()}
    boxes, labels, scores = (prediction[key] for key in ("boxes", "labels", "scores"))
    reference = torch.tensor([expected["box_xyxy"]], dtype=boxes.dtype)
    ious = torchvision.ops.box_iou(boxes, reference)[:, 0]
    categories = weights.meta["categories"]
    matches = (
        (labels == categories.index(expected["class"]))
        & (scores >= expected["score_min"])
        & (ious >= expected["iou_min"])
    )
    passed = bool(matches.any())
    detections = [
        {"class": categories[label], "score": score, "box_xyxy": box,
         "iou": iou, "match": match}
        for label, score, box, iou, match in zip(
            labels.tolist(), scores.tolist(), boxes.tolist(), ious.tolist(), matches.tolist()
        )
    ]
    result = {
        "passed": passed,
        "image_sha256": image_sha256,
        "weights": str(weights),
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": torchvision.__version__, 
            "pillow": pillow_version
        },
        "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "expected": expected,
        "detections": detections,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"{'PASS' if passed else 'FAIL'}: {output.relative_to(root)}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
