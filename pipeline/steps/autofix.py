"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict  # noqa: F401

import yaml  # noqa: F401

from ..common import ERROR, parse_yolo_line  # noqa: F401

DROP_CODES = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
REMOVE_IMAGE_CODES = {"LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}


def _is_remove_image(code):
    return code in REMOVE_IMAGE_CODES or code.startswith("DUP_IMAGE")


def _fix(image, line, action, before, after):
    return {"image": image, "line": line, "action": action, "before": before, "after": after}


def _remove_image(ds, stem):
    """Xoá ảnh + nhãn của `stem` khỏi ds. Trả về True nếu thực sự xoá được thứ gì."""
    removed = False
    img = ds.image_path(stem)
    if img is not None and img.exists():
        img.unlink()
        removed = True
    lbl = ds.label_path(stem)
    if lbl.exists():
        lbl.unlink()
        removed = True
    ds._size_cache.pop(stem, None)
    return removed


def _clip_line(text):
    """Cắt box về [0,1]. Trả về dòng mới, hoặc None nếu box không còn diện tích."""
    cls, cx, cy, w, h = parse_yolo_line(text)
    x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
    x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
    nw, nh = x2 - x1, y2 - y1
    if nw <= 0 or nh <= 0:
        return None
    return f"{cls} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {nw:.6f} {nh:.6f}"


def fix(ds, issues, cfg):
    fixes = []
    errors = [i for i in issues if i.severity == ERROR]    # WARN là việc của người

    # 1) Xoá ảnh (DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE)
    removed = set()
    for i in errors:
        if _is_remove_image(i.code) and i.image not in removed:
            removed.add(i.image)
            before = ds.image_path(i.image)
            before_txt = before.name if before is not None else i.image
            if _remove_image(ds, i.image):
                fixes.append(_fix(i.image, None, "remove_image", before_txt, "(đã xoá ảnh + nhãn)"))

    # 2) data.yaml: ghi lại names đúng cfg["classes"]
    if any(i.code == "BAD_CLASS_NAME" for i in errors):
        want = {k: v for k, v in enumerate(cfg["classes"])}
        have = ds.yaml_names()
        if have != want:
            d = {}
            if ds.yaml_path.exists():
                d = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) or {}
            d["names"] = want
            ds.yaml_path.write_text(
                yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8"
            )
            fixes.append(_fix("data.yaml", None, "rename_classes", str(have), str(want)))

    # 3) Sửa dòng nhãn: gom theo file, số dòng là số dòng GỐC, ghi lại MỘT lần
    by_image = defaultdict(lambda: defaultdict(set))       # stem -> số dòng -> {mã lỗi}
    for i in errors:
        if i.image in removed or i.line is None:
            continue
        if i.code in DROP_CODES or i.code == "BOX_OUT_OF_IMAGE":
            by_image[i.image][i.line].add(i.code)

    for stem in sorted(by_image):
        path = ds.label_path(stem)
        if not path.exists():
            continue
        orig = path.read_text(encoding="utf-8", errors="replace").splitlines()
        new_lines = list(orig)
        to_delete = set()
        changed = False

        for ln in sorted(by_image[stem]):
            if not 1 <= ln <= len(orig):
                continue                                   # số dòng lạ (vd. file đã đổi): bỏ qua êm
            text = orig[ln - 1]
            codes = by_image[stem][ln]

            if codes & DROP_CODES:                         # xoá dòng thắng clip
                to_delete.add(ln)
                fixes.append(_fix(stem, ln, "drop_line", text, "(xoá dòng)"))
                changed = True
            else:                                          # chỉ còn BOX_OUT_OF_IMAGE
                try:
                    clipped = _clip_line(text)
                except ValueError:
                    continue
                if clipped is None:                        # nằm hẳn ngoài ảnh: xoá, không ghi box rỗng
                    to_delete.add(ln)
                    fixes.append(_fix(stem, ln, "drop_line", text, "(xoá dòng)"))
                    changed = True
                elif clipped.split() != text.split():
                    new_lines[ln - 1] = clipped
                    fixes.append(_fix(stem, ln, "clip_box", text, clipped))
                    changed = True

        if changed:
            kept = [t for k, t in enumerate(new_lines, 1) if k not in to_delete]
            path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    # 4) Thiếu file nhãn: KHÔNG tạo file, chỉ ghi lại để chuyển cho người
    seen = set()
    for i in errors:
        if i.code == "MISSING_LABEL_FILE" and i.image not in removed and i.image not in seen:
            seen.add(i.image)
            fixes.append(_fix(i.image, None, "needs_label", "(không có file)", "chuyển người gán"))

    return fixes