"""Create a minimal review pack for human de-identification QC of ION frames."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset" / "ion_bronchoscopy_candidate"
OUTPUT = ROOT / "results" / "ion_visual_qc"
SAMPLES_PER_VIDEO = 12


def contact_sheet(frame_paths: list[Path], output: Path) -> None:
    thumbnails = []
    for path in frame_paths:
        image = Image.open(path).convert("RGB").resize((128, 128))
        thumbnails.append((path.name, image))
    sheet = Image.new("RGB", (4 * 128, 3 * 148), "black")
    draw = ImageDraw.Draw(sheet)
    for index, (name, image) in enumerate(thumbnails):
        x, y = (index % 4) * 128, (index // 4) * 148
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + 130), name, fill="white")
    sheet.save(output, quality=92)


def main() -> None:
    manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    checklist = [
        "# ION visual de-identification QC",
        "",
        "For every contact sheet, confirm all criteria before setting a video to approved:",
        "- no patient name, MRN, date of birth, accession number, or timestamp;",
        "- no readable planning/workstation overlay or burned-in free text;",
        "- the crop contains operative endoscopic content rather than an interface;",
        "- no unexpected external scene or staff identity is visible.",
        "",
        "| Case | Video | Frames | Reviewer | Status (approve/reject) | Notes |",
        "|---|---|---:|---|---|---|",
    ]
    for case in manifest["cases"]:
        for video in case["videos"]:
            if "frames" not in video or not video["frames"]:
                continue
            frames = video["frames"]
            indices = [
                round(index * (len(frames) - 1) / (SAMPLES_PER_VIDEO - 1))
                for index in range(SAMPLES_PER_VIDEO)
            ]
            selected = [
                DATASET / video["output_path"] / frames[index]["frame"]
                for index in indices
            ]
            contact_sheet(
                selected, OUTPUT / f"{video['video_id']}_contact_sheet.jpg"
            )
            checklist.append(
                f"| {case['case_id']} | {video['video_id']} | "
                f"{video['retained_frames']} |  | pending |  |"
            )
    (OUTPUT / "CHECKLIST.md").write_text("\n".join(checklist) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
