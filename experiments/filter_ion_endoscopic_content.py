"""Remove navigation/planning interface frames from the generated ION candidate set.

The filter acts only on frames created in the workspace, never on the source
collection.  It uses conservative colour statistics: valid endoscopic views
have a substantial warm-tissue component, whereas ION planning panels are
cyan/blue dominated.  Human privacy QC remains mandatory after this filter.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset" / "ion_bronchoscopy_candidate"


def is_endoscopic_view(image: np.ndarray) -> bool:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    coloured = (hsv[..., 1] > 45) & (hsv[..., 2] > 40)
    warm = (((hsv[..., 0] < 28) | (hsv[..., 0] > 170)) & coloured).mean()
    cyan = ((hsv[..., 0] > 65) & (hsv[..., 0] < 110) & coloured).mean()
    return warm >= 0.25 and warm > 1.5 * cyan


def main() -> None:
    manifest_path = DATASET / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    retained, rejected = 0, 0
    for case in manifest["cases"]:
        for video in case["videos"]:
            frames = video.get("frames", [])
            if not frames:
                continue
            directory = DATASET / video["output_path"]
            kept = []
            for frame in frames:
                path = directory / frame["frame"]
                image = cv2.imread(str(path))
                if image is not None and is_endoscopic_view(image):
                    kept.append(frame)
                    retained += 1
                else:
                    path.unlink(missing_ok=True)
                    rejected += 1
            video["frames"] = kept
            video["retained_frames"] = len(kept)
            video["rejected_non_endoscopic_interface"] = (
                video.get("rejected_non_endoscopic_interface", 0)
                + len(frames)
                - len(kept)
            )
    manifest["status"] = (
        "endoscopic_content_filtered_candidate_requires_visual_deidentification_qc"
    )
    manifest["automatic_content_filter"] = {
        "method": "warm tissue colour fraction >=0.25 and >1.5x cyan fraction",
        "retained_frames": retained,
        "rejected_navigation_or_nonendoscopic_frames": rejected,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Retained {retained} endoscopic candidates; rejected {rejected} "
        "navigation/non-endoscopic frames."
    )


if __name__ == "__main__":
    main()
