"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Thuật toán gợi ý, với từng ảnh (bỏ qua ảnh không có trong preds hoặc không có file nhãn):
  P = dự đoán có conf >= conf  VÀ diện tích px (w*W * h*H) >= min_area_px
  Với mỗi dự đoán r trong P: lấy các nhãn có IoU(r, nhãn) >= iou_missing  (cands)
   - không có cands                            -> POSSIBLE_MISSING     (model thấy vật mà nhãn không có box)
   - có cands cùng lớp, tốt nhất IoU < iou_loose -> POSSIBLE_LOOSE_BOX  (đúng vật, box lỏng/lệch)
   - không có cands cùng lớp, nhưng cands tốt nhất IoU >= iou_match (khác lớp) -> POSSIBLE_WRONG_CLASS
  Với mỗi dòng nhãn hợp lệ: nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                             -> POSSIBLE_FALSE_POS  (box nhãn không khớp vật nào model thấy; Issue.line = số dòng)

Chi tiết các ca biên (test kiểm đúng các ca này)
- `cands` gồm nhãn MỌI lớp có IoU >= iou_missing; "tốt nhất cùng lớp" = cands cùng lớp có IoU lớn nhất; "tốt nhất khác lớp" tương tự với cands khác lớp.
- Thứ tự quyết định cho mỗi dự đoán: (1) cands rỗng -> MISSING. (2) có cands cùng lớp: IoU tốt nhất < iou_loose -> LOOSE_BOX, ngược lại KHÔNG cờ
  (nhãn đúng lớp đã phủ vật này, kể cả khi còn nhãn khác lớp chồng lên). (3) không có cands cùng lớp: nếu IoU tốt nhất khác lớp >= iou_match -> WRONG_CLASS;
  nếu chỉ có khác lớp với IoU trong [iou_missing, iou_match) -> không cờ gì (mơ hồ: để cho FALSE_POS / lần duyệt khác).
- Một dự đoán sinh tối đa MỘT issue (MISSING / LOOSE_BOX / WRONG_CLASS); một ảnh có thể có nhiều issue.
- Thứ tự issue: theo `ds.image_files()` rồi theo thứ tự dự đoán trong `preds[stem]`, cuối cùng là các FALSE_POS theo số dòng; `Issue.image` = stem, `Issue.line` chỉ đặt cho FALSE_POS.

Lưu ý số dòng: dùng `ds.read_label_lines(stem)` (giữ số dòng gốc, kể cả dòng hỏng/trống) để lấy `Issue.line`; `valid_boxes` KHÔNG giữ số dòng.

Tất cả severity WARN; `Issue.extra` nên có {"conf": ...} hoặc {"iou": ...} (hàng đợi duyệt dùng conf để xếp ưu tiên).

Gợi ý
- `iou_xywh` có sẵn. Dòng nhãn hỏng không được làm script chết (dùng `ds.valid_boxes` và `parse_yolo_line` trong try).
- Kết quả phụ thuộc mạnh vào ngưỡng: calibrate chúng ở Bước 3 thay vì đoán.
"""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401
from ..model_cache import get_preds  # noqa: F401

CHECK = "second_opinion"


def _issue(code, stem, msg, line=None, extra=None):
    return Issue(CHECK, WARN, code, stem, msg, line=line, extra=extra or {})


def run(ds, cfg, preds=None):
    t = cfg["second_opinion"]
    conf_th = t["conf"]
    min_area_px = t["min_area_px"]
    fp_conf = t["fp_conf"]
    iou_missing = t["iou_missing"]
    iou_match = t["iou_match"]
    iou_loose = t["iou_loose"]

    if preds is None:
        preds = get_preds(ds, cfg, "checker")

    out = []
    for img_path in ds.image_files():
        stem = img_path.stem
        if stem not in preds:
            continue
        lines = ds.read_label_lines(stem)          # [(số dòng gốc, nội dung)] hoặc None
        if lines is None:
            continue                               # không có file nhãn: việc của MISSING_LABEL_FILE
        size = ds.image_size(stem)
        if size is None:
            continue                               # ảnh hỏng: check `images` sẽ báo
        W, H = size

        L = ds.valid_boxes(stem)                   # (cls, cx, cy, w, h), đã bỏ dòng hỏng
        pr = preds[stem]                           # [cls, conf, cx, cy, w, h]

        # P = dự đoán đủ tin cậy VÀ đủ lớn (px)
        P = [r for r in pr if r[1] >= conf_th and (r[4] * W) * (r[5] * H) >= min_area_px]

        # --- các cờ sinh ra từ dự đoán: tối đa MỘT issue mỗi dự đoán ---
        for r in P:
            rcls, rconf = int(r[0]), r[1]
            rbox = tuple(r[2:6])
            cands = []                             # (iou, lớp nhãn)
            for b in L:
                iou = iou_xywh(rbox, b[1:5])
                if iou >= iou_missing:
                    cands.append((iou, b[0]))

            if not cands:
                out.append(_issue("POSSIBLE_MISSING", stem,
                                  f"model thấy lớp {rcls} (conf {rconf:.2f}) mà nhãn không có box",
                                  extra={"conf": round(rconf, 4)}))
                continue

            same = [iou for iou, c in cands if c == rcls]
            if same:
                best = max(same)
                if best < iou_loose:
                    out.append(_issue("POSSIBLE_LOOSE_BOX", stem,
                                      f"box lớp {rcls} khớp vật nhưng lỏng/lệch (IoU {best:.2f} < {iou_loose})",
                                      extra={"conf": round(rconf, 4), "iou": round(best, 4)}))
                continue                           # đã có nhãn đúng lớp phủ vật này: không cờ

            best_other = max(iou for iou, c in cands if c != rcls)
            if best_other >= iou_match:
                out.append(_issue("POSSIBLE_WRONG_CLASS", stem,
                                  f"model thấy lớp {rcls} (conf {rconf:.2f}) nhưng nhãn gán lớp khác (IoU {best_other:.2f})",
                                  extra={"conf": round(rconf, 4), "iou": round(best_other, 4)}))
            # IoU trong [iou_missing, iou_match): mơ hồ, không cờ gì

        # --- FALSE_POS: box nhãn không khớp vật nào model thấy ---
        Pfp = [r for r in pr if r[1] >= fp_conf]   # KHÔNG lọc diện tích
        for ln, text in lines:
            try:
                cls, cx, cy, w, h = parse_yolo_line(text)
            except ValueError:
                continue
            if not 0 <= cls < len(ds.classes):
                continue                           # dòng hỏng: label_format đã báo
            best = max((iou_xywh(tuple(r[2:6]), (cx, cy, w, h)) for r in Pfp), default=0.0)
            if best < iou_missing:
                out.append(_issue("POSSIBLE_FALSE_POS", stem,
                                  f"box lớp {cls} không khớp vật nào model thấy (IoU tốt nhất {best:.2f})",
                                  line=ln, extra={"iou": round(best, 4)}))
    return out
