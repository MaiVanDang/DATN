# Active Auth — Web Demo (Streamlit)

Trực quan hóa quá trình xác thực hành vi ở **mức phiên** cho một chủ máy đã đăng ký.
Hiển thị đồng thời 3 điểm và 2 kết luận để đối chiếu minh bạch:

- **Điểm quán tính (`cos_znorm`)** — *đường quyết định chính, khớp bản triển khai on-device*:
  mean cosine tới anchor → chuẩn hóa z-norm theo nhóm nền (cohort) → sigmoid.
- **Điểm touch (RF)** — Random Forest owner-vs-pool trên đặc trưng touch **33 chiều** (tap + scroll).
- **Điểm fusion** — trung bình có trọng số của điểm quán tính và điểm touch.

Kết luận theo **quán tính** (bản triển khai) và theo **fusion** hiển thị cạnh nhau; nếu fusion không
tốt hơn quán tính thì bảng sẽ cho thấy đúng như vậy (khớp kết luận "fusion không cải thiện tập mở").

Kiến trúc backbone: **CNN 1D** (kiến trúc đã chọn để triển khai). Ngữ cảnh: **walking** hoặc **all** (3 hoạt động).

## Cài đặt

```bash
cd web_demo
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows
pip install -r requirements.txt   # streamlit, torch, scikit-learn, numpy, pandas
```

## Cấu trúc cần có

```
web_demo/
├── app.py                  # giao diện Streamlit
├── verifier.py             # lõi chấm điểm: cos_znorm + RF touch + fusion
├── models.py               # định nghĩa backbone (CNN / ConvLSTM / ConvLSTM-Bi)
├── touch_subwindow.py      # đặc trưng touch 33-D (sub-window tap + scroll)
├── evaluate_variants.py    # (tùy chọn) benchmark 6 biến thể để chọn model
├── validate_threshold.py   # (tùy chọn) kiểm chứng ngưỡng per-owner vs cố định
├── requirements.txt
│
├── artifacts/
│   └── cnn/                                 ← CHỈ CNN cần cho demo
│       ├── models_walking/backbone.pt
│       ├── models_all/backbone.pt
│       ├── export_walking/                  ← pool + scaler cho mode 'walking'
│       │   ├── impostor_pool_inertial.npy
│       │   ├── impostor_pool_touch.npy
│       │   └── touch_scaler.json
│       └── export_all/                      ← (tương tự) cho mode 'all'
│   └── (convlstm/, convlstm_bi/ chỉ cần khi chạy evaluate_variants.py)
│
└── processed/                          ← dữ liệu người dùng (cohort + chủ máy để demo)
    ├── user1/
    │   ├── X_walking.npy  / y_walking.npy   ← cửa sổ đi bộ + nhãn phiên
    │   ├── X_inertial.npy / y_inertial.npy  ← cửa sổ cả 3 hoạt động
    │   ├── tap_gestures.csv                 ← sự kiện chạm (cho touch 33-D)
    │   └── scroll_gestures.csv              ← cử chỉ cuộn
    └── ... (≥ 2 user)
```

## Chạy demo

```bash
streamlit run app.py
```

Trình duyệt tự mở http://localhost:8501.

## Luồng demo

1. **Sidebar** → chọn **Thư mục dữ liệu** (mặc định `processed`) và **Ngữ cảnh** (all / walking).
2. Chọn **Chủ máy** + **Số session enroll** → bấm **🎯 Đăng ký**.
   Bước này tạo anchor, tính tham số z-norm cohort và ngưỡng per-owner; huấn luyện RF touch nếu có dữ liệu chạm.
3. (Nếu có touch) chỉnh slider **Fusion** để khám phá — mặc định là trọng số được tune tự động trên tập val.
4. Bấm **▶️ Chấm điểm tất cả** → bảng **Chủ máy vs Người lạ** với các cột:
   `Người · Session · Điểm quán tính · Điểm touch · Điểm fusion · Kết luận (quán tính) · Kết luận (fusion) · Đúng?`
   kèm 2 chỉ số **Độ chính xác** (quán tính vs fusion).

## Ngưỡng quyết định

Ngưỡng được **hiệu chuẩn riêng cho từng chủ máy** ngay lúc đăng ký (cân bằng FAR≈FRR trên phiên val,
ở đúng mức gộp cửa sổ mà quyết định sử dụng), fallback về hằng số `0.23` khi thiếu dữ liệu. Đây là
cùng cơ chế ngưỡng per-owner với bản triển khai trên thiết bị.

## Script phụ (không thuộc demo)

- `evaluate_variants.py` — so 6 biến thể (3 kiến trúc × 2 ngữ cảnh) bằng kiểm thử chéo leave-users-out
  (AUC/EER, kích thước, độ trễ) để làm căn cứ chọn CNN. Cần đủ 3 folder `cnn/`, `convlstm/`, `convlstm_bi/`.
- `validate_threshold.py` — đối chiếu ngưỡng per-owner với ngưỡng cố định, báo FRR (owner bị từ chối) và FAR (impostor lọt).

## Troubleshooting

- **Không thấy user**: kiểm tra `processed/<user>/` có `X_*.npy` + `y_*.npy`.
- **Nhánh touch bị bỏ qua**: user thiếu `tap_gestures.csv` / `scroll_gestures.csv` hoặc lệch chiều (≠ 33) → demo tự chạy thuần quán tính, cột touch/fusion để trống.
- **Backbone size mismatch**: đã xử lý trong `models.py` (bỏ classifier head khi `n_users` khác lúc train).
- **Thiếu `impostor_pool_*.npy` trong `export_<mode>/`**: demo vẫn chạy nhưng z-norm cohort suy biến về cosine thô → nên bảo đảm có đủ pool để điểm sát bản triển khai.
```
