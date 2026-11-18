# Báo cáo Day 09 — Đinh Hữu Thái An, 2A202603283

> Xoá mọi `TODO(học viên)` và dòng gợi ý. Mỗi ý trả lời bằng **số lấy từ `submission/reports/`**, kèm mẫu số. Cả báo cáo ~1–2 trang.

## 1. Kết luận trước
Pipeline kết luận Hold. Lô có 224 ảnh, bước 1 loại 22 (hỏng, kích thước, trùng, rò rỉ test) còn 202 ảnh. Dataset xuất ra clean_v1 có 202 ảnh (bản nháp, chưa phát hành). Lý do Hold: không còn lỗi bắt buộc nhưng 38/202 ảnh (18,8%) còn cảnh báo, vượt ngưỡng 10%; hàng đợi có 125 ảnh mà ngân sách duyệt chỉ 80.
## 2. Lỗi và phân việc
Lấy từ `02_error_list_v1` (gom theo mã lỗi). Cột "ai sửa": **máy** (autofix) hay **người** (hàng đợi); chỉ ghi những mã có mặt trong lô của bạn.

| Mã lỗi | Số ảnh/dòng | Ai sửa (máy/người) | Vì sao |
|---|---:|---|---|
| BAD_SIZE | 3 ảnh | máy (loại ở bước 1) | cạnh ngắn < 250px, quy tắc cứng |
| CORRUPT_IMAGE | 2 ảnh | máy (loại ở bước 1) | không mở được ảnh thì không có gì để gán nhãn |
| DUP_IMAGE_EXACT | 8 ảnh | máy (loại ở bước 1) | trùng từng byte, chắc chắn |
| DUP_IMAGE_NEAR | 4 ảnh | máy (loại ở bước 1) | pHash ≤ 10 (ngưỡng đã calibrate) |
| LEAK_TEST | 5 ảnh | máy (loại ở bước 1) | trùng ảnh kho test/val, rò rỉ làm số đo sai |
| BAD_CLASS_ID | 6 dòng | máy (drop_line) | id ngoài 0..3, máy không đoán được lớp đúng nên bỏ dòng |
| BAD_FORMAT | 3 dòng | máy (drop_line) | giá trị nan/inf hoặc thiếu cột |
| BOX_ZERO | 4 dòng | máy (drop_line) | w hoặc h = 0 |
| DUP_BOX | 7 dòng | máy (drop_line) | cùng lớp, IoU 0.93–0.99, dòng thừa |
| BOX_OUT_OF_IMAGE | 8 dòng | máy (clip_box) | tràn 0.023–0.06, cắt về [0,1] |
| BAD_CLASS_NAME | 2 mục (1 lần sửa) | máy (rename_classes) | 'Person', 'Bus' viết hoa sai, ghi lại data.yaml |
| MISSING_LABEL_FILE | 6 ảnh | người | máy không bịa nhãn, chuyển người gán (needs_label) |
| POSSIBLE_* (model lớn) | 128 dòng | người | lỗi ngữ nghĩa, model cũng có thể sai nên chỉ cảnh báo |

Mỗi ảnh có điểm ưu tiên dựa trên trọng số theo mã lỗi trong review.weights của pipeline.yaml. NEEDS_LABEL nặng nhất (10), nên 6 ảnh thiếu nhãn đứng ngay sau img_0020 (11,52, có WRONG_CLASS trọng số 3 và MISSING, FIXED_DROP_LINE trọng số 2). Tiếp theo là các ảnh có POSSIBLE_WRONG_CLASS và POSSIBLE_MISSING. Điểm số thực tế còn cao hơn tổng trọng số thuần (img_0020: 7 so với 11,52), nên có thêm yếu tố khác, nhiều khả năng là conf của model
Với ngân sách 80 ảnh, người xem được 88/128 dòng POSSIBLE_* cảnh báo ngữ nghĩa; 45 ảnh cuối chưa ai xem

## 3. Calibration (2 ngưỡng)
| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | 10 | DUP_IMAGE_NEAR: 1.00/0.50 → 1.00/0.67; EXACT giữ 1.00/1.00 | nằm giữa cao nguyên 4–14 nên ổn định; 16 có F1 cao nhất nhưng sát điểm vỡ ở 20 |
| `second_opinion.conf` | 0.25 | 0.6 | MISSING: 0.50/1.00 → 0.57/1.00; WRONG_CLASS: 0.56/0.83 → 0.62/0.83; LOOSE_BOX: 0.29/0.40 → 0.20/0.20 | giữ recall 1.00 cho box thiếu nhãn, giảm cảnh báo thừa; đổi lại LOOSE_BOX kém đi |

precision của phash_dist luôn 1.00 vì calib chỉ có 39 ảnh, ít cặp ảnh khác nhau nhưng giống nhau để lộ false positive. Giá trị 16 có F1 cao nhất (0.91) nhưng sát điểm vỡ ở 20. Bằng chứng từ dữ liệu của bạn: số ảnh bị loại ở bước 1 tăng từ 19 lên 22 khi đổi 2 → 10.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
(a) Chạy python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb (không có người duyệt): còn 6 lỗi bắt buộc (ảnh thiếu file nhãn, máy không bịa nhãn) → kết luận Rework, mã thoát 20, dataset không được xuất. Khi người duyệt (ngân sách 80 ảnh) gán nhãn cho các ảnh này, lỗi bắt buộc về 0 và kết luận chuyển sang Hold. 
(b) **(b) Điểm mù (pipeline không bắt được), có bằng chứng:**
1. **Box lỏng nhẹ:** `POSSIBLE_LOOSE_BOX` chỉ cờ khi cùng lớp mà IoU < 0.5 (`iou_loose` cố định), nên box lỏng nhưng IoU ≥ 0.5 không bao giờ bị cờ. Trên calib, ở conf = 0.6 chỉ bắt 1/5 lỗi chèn (precision 0.20, recall 0.20); F1 cao nhất cũng chỉ 0.46 (conf 0.1).
2. **Ảnh gần trùng biến đổi mạnh:** ở `phash_dist` = 10, `DUP_IMAGE_NEAR` đạt recall 0.67 (bắt 4/6 trên calib); nâng lên 16 được 0.83 nhưng sát điểm vỡ ở 20. Lô raw không có đáp án, nên pipeline bắt 4 cặp (img_0076, 0143, 0186, 0223) và không biết còn cặp nào lọt.
3. **Tự sửa làm mất nhãn thật:** `drop_line` xoá dòng hỏng thay vì khôi phục, nên img_0050, img_0104, img_0194 chỉ còn file nhãn rỗng dù ảnh có vật. Máy chỉ để lại cờ `EMPTY_LABEL` trong hàng đợi, nhãn có được trả lại hay không phụ thuộc người duyệt.

## 5. Train: raw vs clean (tùy chọn, chấm riêng)
Chưa chạy

## 6. Quyết định và ghi chú AI
(a) Quyết định cuối: Hold, chưa phát hành. Để thành Release cần giảm số ảnh còn cảnh báo xuống tối đa 20/202 (dưới 10%): tăng ngân sách duyệt để xem hết hàng đợi 125 ảnh, và xử lý nguồn POSSIBLE_FALSE_POS (không chỉnh được bằng conf). (b) Dùng AI (Claude, chat web) viết cả 4 module: label_format, gate, autofix, second_opinion.
Prompt chính: dán file stub (docstring), file test và common.py, yêu cầu viết code cho test pass và không sửa test.
Cách kiểm: chạy pytest từng module rồi cả bộ (79 passed), sau đó chạy pipeline và đối chiếu kết quả với ảnh., 0 lần AI sai.
