from __future__ import annotations

import argparse
import json
from pathlib import Path


EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}


def main() -> int:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=project_dir / "label-studio-data" / "images",
    )
    parser.add_argument("--output", type=Path, default=project_dir / "tasks.json")
    parser.add_argument("--limit", type=int, default=60)
    args = parser.parse_args()

    images_dir = args.images_dir.resolve()
    images_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p
        for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )[: args.limit]

    if not images:
        print(f"No images found in {images_dir}")
        print("Put dataset images into label-studio-data/images and run again.")
        return 1

    tasks = [
        {
            "data": {
                "image": f"/data/local-files/?d=images/{image.relative_to(images_dir).as_posix()}"
            }
        }
        for image in images
    ]

    args.output.write_text(
        json.dumps(tasks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Created {len(tasks)} tasks: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())