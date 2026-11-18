"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401

CHECK = "label_format"


def _issue(sev, code, stem, line, msg, extra=None):
    return Issue(CHECK, sev, code, stem, msg, line=line, extra=extra or {})


def run(ds, cfg):
    t = cfg["label_format"]
    overflow_tol = t["overflow_tol"]
    dup_iou = t["dup_box_iou"]
    min_box_px = t["min_box_px"]
    max_box_frac = t["max_box_frac"]

    out = []
    for path in ds.label_files():
        stem = path.stem
        lines = ds.read_label_lines(stem) or []   # (số dòng từ 1, nội dung), đã bỏ dòng trống
        size = ds.image_size(stem)                 # (rộng, cao) hoặc None nếu ảnh hỏng/thiếu
        prev = []                                  # các box hợp lệ ở dòng trước: (cls, cx, cy, w, h)

        for ln, text in lines:
            # 1) Sai định dạng -> báo rồi bỏ qua dòng này
            try:
                cls, cx, cy, w, h = parse_yolo_line(text)
            except ValueError as e:
                out.append(_issue(ERROR, "BAD_FORMAT", stem, ln, f"dòng nhãn sai định dạng: {e}"))
                continue

            # 2) ID lớp ngoài khoảng
            if not 0 <= cls < len(ds.classes):
                out.append(_issue(ERROR, "BAD_CLASS_ID", stem, ln,
                                  f"id lớp {cls} ngoài 0..{len(ds.classes) - 1}"))
                continue

            # 3) Box rỗng
            if w <= 0 or h <= 0:
                out.append(_issue(ERROR, "BOX_ZERO", stem, ln, f"box rỗng (w={w}, h={h})"))
                continue

            # 4) Box tràn ảnh: mức tràn lớn nhất trong 4 mép
            overflow = max(-(cx - w / 2), (cx + w / 2) - 1, -(cy - h / 2), (cy + h / 2) - 1)
            if overflow > overflow_tol:
                out.append(_issue(ERROR, "BOX_OUT_OF_IMAGE", stem, ln,
                                  f"box tràn ảnh {overflow:.4f} (> {overflow_tol})",
                                  {"overflow": overflow}))

            # 5) Box lặp: cùng lớp, IoU lớn với một dòng TRƯỚC ĐÓ -> báo ở dòng SAU
            box = (cx, cy, w, h)
            for pcls, *pbox in prev:
                if pcls == cls:
                    iou = iou_xywh(pbox, box)
                    if iou > dup_iou:
                        out.append(_issue(ERROR, "DUP_BOX", stem, ln,
                                          f"box lặp với dòng trước (IoU={iou:.2f})",
                                          {"iou": round(iou, 4)}))
                        break
            prev.append((cls, cx, cy, w, h))

            # 6) Cảnh báo: box quá nhỏ (cần kích thước ảnh)
            if size is not None:
                short_px = min(w * size[0], h * size[1])
                if short_px < min_box_px:
                    out.append(_issue(WARN, "BOX_TINY", stem, ln,
                                      f"cạnh ngắn của box {short_px:.1f}px < {min_box_px}px"))

            # 7) Cảnh báo: box quá lớn
            if w * h > max_box_frac:
                out.append(_issue(WARN, "BOX_HUGE", stem, ln,
                                  f"box chiếm {w * h:.2f} diện tích ảnh (> {max_box_frac})"))
    return out
