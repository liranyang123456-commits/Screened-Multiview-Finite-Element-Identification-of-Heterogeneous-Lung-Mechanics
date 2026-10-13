"""Export per-comment plain-text replies for the Elsevier online peer review platform.

The decision letter states: "Before you can submit your revision, you must respond to
reviews in our peer review platform." The platform asks for one reply per reviewer
comment. This script converts Response_Reviewer[1-4].md into plain text (no Markdown
markup), one block per comment, so each block can be pasted into the corresponding
reply box. The wording is identical to the uploaded response PDFs
(02b_Response_to_Reviewer1.pdf ... 02e_Response_to_Reviewer4.pdf).

Output: responses/platform_replies/Platform_Replies_Reviewer<n>.txt
"""
from pathlib import Path
import re

base = Path(__file__).resolve().parent
source = base / "responses"
target = source / "platform_replies"
target.mkdir(parents=True, exist_ok=True)


def plain(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = text.replace("`", "")
    return text.strip()


def upload_name(reviewer: str) -> str:
    """Submission-system file name of the uploaded response PDF (02b-02e)."""
    return f"02{'bcde'[int(reviewer) - 1]}_Response_to_Reviewer{reviewer}.pdf"


for md in sorted(source.glob("Response_Reviewer[1-4].md")):
    lines = md.read_text(encoding="utf-8-sig").splitlines()
    reviewer = re.search(r"Reviewer(\d)", md.stem).group(1)
    blocks = []
    current = None
    mode = None
    for raw in lines:
        line = raw.rstrip()
        if line.startswith("### "):
            if current:
                blocks.append(current)
            current = {"title": plain(line[4:]), "quote": [], "response": [], "location": []}
            mode = None
            continue
        if current is None:
            continue
        if line.startswith("> "):
            body = plain(line[2:])
            if body.lower().startswith("reviewer's comment:"):
                continue
            if body:
                current["quote"].append(body)
            continue
        if line.strip().startswith("**Author Response:**"):
            mode = "response"
            rest = plain(line.split("**Author Response:**", 1)[1])
            if rest:
                current["response"].append(rest)
            continue
        if line.strip().startswith("**Location of revisions:**"):
            mode = "location"
            rest = plain(line.split("**Location of revisions:**", 1)[1])
            if rest:
                current["location"].append(rest)
            continue
        if line.strip() == "---" or not line.strip():
            continue
        if mode:
            current[mode].append(plain(line))
    if current:
        blocks.append(current)

    out = target / f"Platform_Replies_Reviewer{reviewer}.txt"
    with out.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"Replies for Reviewer {reviewer} - paste each block into the matching comment box "
                f"of the online peer review platform ({len(blocks)} comments).\n")
        f.write(f"Wording is identical to the uploaded {upload_name(reviewer)}.\n\n")
        for k, b in enumerate(blocks, 1):
            f.write("=" * 78 + "\n")
            f.write(f"{b['title']}\n")
            f.write("=" * 78 + "\n")
            f.write("[Reviewer's comment, for orientation only - do not paste]\n")
            f.write(" ".join(b["quote"]) + "\n\n")
            f.write("[REPLY - paste from here]\n")
            if k == 1:
                f.write("(Notation used in all replies: counts written as a/b/c give the numbers of cases "
                        "that worsened/improved/tied relative to the stated reference.)\n")
            f.write("\n".join(b["response"]) + "\n")
            if b["location"]:
                f.write("Location of revisions: " + " ".join(b["location"]) + "\n")
            f.write("[end of reply]\n\n")
    print(out, len(blocks), "comments")
