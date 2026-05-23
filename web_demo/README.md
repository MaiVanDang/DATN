# Active Auth — Web Demo (Streamlit)

Demo khoa học cho thesis defense. **V5-synced**: dùng artifacts (backbone, impostor pool, scaler) đã save từ training pipeline V5, đảm bảo demo cho ra kết quả khớp với metric trong báo cáo.

## Cài đặt

```bash
cd demo
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

## Cấu trúc cần có

```
demo/
├── app.py
├── verifier.py
├── touch_features.py
├── models.py
├── requirements.txt
├── README.md
│
├── models/
│   └── backbone.pt                       ← từ training V5
│
├── export/                                ← TỪ TRAINING V5 — đảm bảo demo match báo cáo
│   ├── impostor_pool_inertial.npy        (100, 128) — pool inertial
│   ├── impostor_pool_touch.npy           (96, 47)   — pool touch đã scaled
│   ├── touch_scaler.json                 — scaler params (mean + scale)
│   └── backbone_metadata.json            — metadata users trained
│
└── processed_data/                        ← data của cohort training (24 user)
    ├── user1/
    │   ├── X.npy                         — inertial windows shape (N, 9, 100)
    │   ├── session.npy                   — session ID per window
    │   ├── y.npy                         — activity labels (không dùng)
    │   └── touch_session_features.csv    — 47-D touch features per session
    ├── user2/
    │   └── ...
    └── ... (24 users)
```

## (Optional) Newbie data — cho thesis defense

User CHƯA TỪNG xuất hiện trong training cohort:

```
demo/
└── newbie_data/
    ├── newbie1/
    │   ├── X.npy
    │   ├── session.npy
    │   └── touch_session_features.csv
    └── ...
```

## Chạy demo

```bash
streamlit run app.py
```

Browser tự mở http://localhost:8501

## Đảm bảo demo match báo cáo

Demo này khớp 100% với training pipeline V5 nhờ:

1. **Load pre-built impostor pool** từ `export/impostor_pool_*.npy` (không rebuild on-the-fly)
2. **Load pre-fit StandardScaler** từ `export/touch_scaler.json` (fit trên pool only, không phải pool+owner)
3. **Per-window fusion**: score touch (per-session) được broadcast lên từng window, fuse rồi mean
4. **Tune fusion_w**: grid search 51 bước, tie-break ưu tiên 0.5 (giống `fusion.search_weight()` trong V5)
5. **RF hyperparameters** giữ y nguyên: `n_estimators=200, max_features='sqrt', class_weight='balanced'`, `min_samples_leaf=2` (inertial) / `=1` (touch)

## Flow demo cho thesis defense

### Bước 1: Sidebar — Enrollment

1. Verify sidebar hiển thị "✓ V5 artifacts loaded" với shape đúng
2. Chọn `Owner user` = vd `user5`
3. Chọn `Số session để enroll` = 4 (giữ 2 sessions làm test own-data)
4. Bấm **Enroll** → train RF (~2-5 giây). Message sẽ show `fusion_w = X.XX` đã tune

### Bước 2: Tab "Own data" — FRR

1. Bấm **Run own-data verification**
2. Xem score 2 sessions còn lại — kỳ vọng TRUSTED, FRR = 0%

### Bước 3: Tab "Single impostor"

1. Chọn `Pick impostor user` lạ
2. Bấm **Run impostor verification** — kỳ vọng REJECTED, FAR = 0%

### Bước 4: Tab "Batch in-cohort"

1. Bấm **Run batch verification**
2. Xem FAR/FRR tổng, distribution chart owner vs impostor

### Bước 5: Tab "Newbie" — generalization

1. Pick newbie hoặc batch
2. Bấm **Run newbie test**
3. Kỳ vọng: Newbie cluster sát với in-cohort impostor (generalization tốt)

## Troubleshooting

**`Lỗi load artifacts`**: kiểm tra `export/` chứa đủ 3 file `impostor_pool_inertial.npy`, `impostor_pool_touch.npy`, `touch_scaler.json`.

**`touch_session_features.csv không tồn tại`**: chạy step2 của training pipeline để generate file này cho mỗi user, hoặc copy từ folder training của notebook.

**Backbone size mismatch**: đã được handle trong `models.py`, demo bỏ classifier head khi load. Nếu vẫn lỗi, xoá `__pycache__/` rồi restart.

**Touch RF skipped**: 1 số user không có `touch_session_features.csv` hoặc < 2 session có touch. Demo tự fallback dùng pure inertial.
