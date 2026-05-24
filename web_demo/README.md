# Active Auth — Web Demo (Streamlit)

Demo so sánh **6 biến thể model** song song:

- **3 kiến trúc**: CNN, ConvLSTM, ConvLSTM-Bi
- **2 chế độ training**: walking (1 action) + all (3 actions)

Mỗi session test được score qua TẤT CẢ 6 model để so sánh trực tiếp.

## Cài đặt

```bash
cd demo
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

## Cấu trúc cần có

```
demo/
├── app.py
├── verifier.py
├── touch_features.py
├── models.py
├── build_pool.py
├── requirements.txt
├── README.md
│
├── artifacts/                            ← root chứa 3 model variants
│   ├── cnn_v2/
│   │   ├── models_walking/backbone.pt
│   │   ├── models_all/backbone.pt
│   │   ├── export_walking/               ← pool + scaler cho mode 'walking'
│   │   │   ├── impostor_pool_inertial.npy
│   │   │   ├── impostor_pool_touch.npy
│   │   │   └── touch_scaler.json
│   │   └── export_all/                   ← pool + scaler cho mode 'all'
│   │       └── ...
│   ├── convlstm_v2/
│   │   └── (cấu trúc tương tự)
│   └── convlstm_bi_v2/
│       └── (cấu trúc tương tự)
│
├── processed_data/                       ← cohort training data
│   ├── user1/
│   │   ├── X_walking.npy                 ← windows đi bộ
│   │   ├── y_walking.npy                 ← session ID per window
│   │   ├── X_inertial.npy                ← windows tất cả activity
│   │   ├── y_inertial.npy
│   │   └── touch_session_features.csv    ← 48-D touch features
│   └── ... (≥2 users)
│
└── newbie_data/                          ← (optional) users UNSEEN
    └── newbie1/
        └── (cấu trúc tương tự processed_data)
```

## Chạy demo

```bash
streamlit run app.py
```

Browser tự mở http://localhost:8501.

## Flow demo

### Bước 1: Sidebar — cấu hình + enrollment

1. Verify sidebar hiển thị `✓ Loaded 6/6 variants`
2. Chọn `Owner pool` (Cohort hoặc Newbie)
3. Chọn `Owner user`
4. Đặt `Số session để enroll` (mặc định 4)
5. Bấm **Enroll all 6 variants** → train 6 RF (~30 giây)

### Bước 2: Tab "Own data" — FRR

Bấm **Run own-data verification** → bảng FRR per-variant + chi tiết từng session × variant.

### Bước 3: Tab "Single impostor"

Pick impostor → **Run impostor verification** → so sánh FAR giữa 6 variants.

### Bước 4: Tab "Batch in-cohort"

**Run batch verification** → tính FAR tổng trên tất cả cohort impostors, có distribution chart 2×3 (mỗi subplot 1 variant).

### Bước 5: Tab "Newbie" — generalization

Pick newbie hoặc batch → **Run newbie test** → đánh giá khả năng generalize của từng variant trên user UNSEEN.

## Threshold

Sidebar có toggle:

- **Adaptive per-variant (mặc định)**: mỗi variant dùng EER threshold riêng tính từ val set lúc enroll
- **Manual**: 1 threshold thủ công áp dụng cho cả 6 variant (để so sánh fair tại cùng operating point)

## Bảng kết quả

Mỗi tab hiển thị 2 thứ:

1. **Metric summary** — 6 dòng (1 dòng/variant), cột FAR hoặc FRR, có highlight best (xanh) / worst (đỏ)
2. **Detail table** — long-form, cột: `model | train_mode | test_user | session | p_inertial | p_touch | fused | threshold | decision | n_windows`

## Troubleshooting

**`❌ Thiếu các file/folder sau`**: kiểm tra `artifacts/` có đủ 3 model × 2 mode = 12 sub-folder (6 `models_*` + 6 `export_*`).

**`touch_session_features.csv không tồn tại`**: chạy step2 của training pipeline để generate file này.

**Backbone size mismatch**: đã được handle trong `models.py` (drop classifier head khi load nếu n_users khác lúc train).

**Touch RF skipped**: user không có touch data hoặc < 2 session có touch. Demo tự fallback dùng pure inertial cho variant đó.

**Variant nào đó enroll thất bại**: app sẽ hiển thị warning và tiếp tục với các variant còn lại — không stop toàn bộ.
