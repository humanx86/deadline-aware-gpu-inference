"""Check one image against a local running vLLM server"""

import base64
import hashlib
import json
import mimetypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/vlm"
OUTPUT = ROOT / "results/vlm-smoke/result.json"

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
PROMPT = (
    "Which animal is visible? "
    "Reply with exactly one label: cat, dog, bird, or none"
)
SAMPLING = {"temperature": 0, "max_tokens": 16}


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.unlink(missing_ok=True)

    from openai import OpenAI

    expected = json.loads((FIXTURE / "expected.json").read_text(encoding="utf-8"))
    image_path = FIXTURE / expected["image"]
    image_bytes = image_path.read_bytes()
    image_hash = hashlib.sha256(image_bytes).hexdigest()

    if image_hash != expected["image_sha256"]:
        raise ValueError("Input image does not match expected.json")

    mime_type = mimetypes.guess_type(image_path.name)[0]
    if mime_type not in {"image/png", "image/jpeg", "image/webp"}:
        raise ValueError("Unsupported image format")

    image_url = (
        f"data:{mime_type};base64," + base64.b64encode(image_bytes).decode("ascii")
    )

    with OpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        max_retries=0,
        timeout=120.0,
    ) as client:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": PROMPT},
                ],
            }],
            **SAMPLING,
        )

    choices = response.choices
    answer = (
        (choices[0].message.content or "").strip().lower()
        if choices else ""
    )
    passed = (
        len(choices) == 1
        and answer == expected["expected_label"].strip().lower()
        and choices[0].finish_reason == "stop"
    )

    result = {
        "passed": passed,
        "image_sha256": image_hash,
        "expected_label": expected["expected_label"],
        "model": MODEL,
        "prompt": PROMPT,
        "sampling": SAMPLING,
        "response": response.model_dump(mode="json"),
    }

    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{'PASS' if passed else 'FAIL'}: {OUTPUT.relative_to(ROOT)}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
