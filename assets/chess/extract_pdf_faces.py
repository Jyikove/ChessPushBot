"""Extract native chess artwork from page 8 of the first-round notice."""
import argparse
import hashlib
import json
from pathlib import Path

import fitz


# The last pictures of the red and white row are repeated rank illustrations.
POSITIONS = (
    ("红_帥", 0, 0), ("红_相", 1, 0), ("白_將", 2, 0), ("白_象", 3, 0),
    ("红_車", 0, 1), ("红_仕", 1, 1), ("白_車", 2, 1), ("白_士", 3, 1),
    ("红_馬", 0, 2), ("红_兵", 1, 2), ("白_馬", 2, 2), ("白_卒", 3, 2),
    ("红_炮", 0, 3), ("白_炮", 2, 3),
)
PAGE = 8


def extract(pdf, output):
    output.mkdir(exist_ok=True)
    with fitz.open(pdf) as doc:
        page = doc[PAGE-1]
        if "字符与颜色示" not in page.get_text():
            raise ValueError("the chess illustration page did not match the notice")
        images = {}
        for info in page.get_image_info(xrefs=True):
            if not 60 <= info["width"] <= 80 or not 60 <= info["height"] <= 80:
                continue
            x, y, *_ = info["bbox"]
            col = round((x-126)/97)
            row = min(range(4), key=lambda i: abs(y-(193, 263, 344, 424)[i]))
            if (row, col) in images:
                raise ValueError("more than one chess face at the same position")
            images[row, col] = info
        if len(images) != 16:
            raise ValueError(f"expected 16 artwork placements, got {len(images)}")
        entries = []
        for name, row, col in POSITIONS:
            info = images[row, col]
            native = fitz.Pixmap(doc, info["xref"])
            filename = f"{name}.png"
            native.save(output / filename)
            entries.append(dict(id=name, file=filename, xref=info["xref"],
                                size_px=[native.width, native.height]))
    info = dict(pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
                notice="2026-11-17 组委会〔2026〕1号", pdf_page=PAGE, faces=entries)
    (output / "source.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    args = parser.parse_args()
    extract(args.pdf, Path(__file__).resolve().parent / "faces")
