# ml_pipeline/export

Chuyển đổi checkpoint PyTorch sang TFLite và đồng bộ assets vào Android app.

## Script

| File | Mô tả |
|------|-------|
| `export_tflite.py` | Script thống nhất, hỗ trợ 3 model. **Dùng cái này.** |
| `export_tflite_cnn.py` | Script cũ, chỉ dành cho BackboneCNN. |

---

## Cách dùng

### 1. Chọn model

Mở `export_tflite.py`, sửa dòng 16:

```python
MODEL = "cnn"          # Conv1D x3 + GlobalAvgPool  → 308 KB
# MODEL = "convlstm"   # Conv1D x2 + LSTM           → 768 KB
# MODEL = "convlstm_bi"# Conv1D x2 + BiLSTM         → 808 KB
```

### 2. Chạy (từ thư mục gốc `f:/DATN`)

```powershell
python ml_pipeline/export/export_tflite.py
```

### 3. Rebuild Android app

Android Studio → **Build → Rebuild Project**

---

## Output

Script tự động sinh và copy 4 file vào `android_app/.../assets/`:

```
backbone.tflite            ← model chính
impostor_pool_inertial.npy ← pool impostors cho inertial scoring
impostor_pool_touch.npy    ← pool impostors cho touch scoring
touch_scaler.json          ← scaler cho touch features
```

> `scaler_params.json` và `export_manifest.json` không bị ghi đè — cập nhật thủ công nếu thay đổi ngưỡng.

---

## Nguồn artifacts theo model

| MODEL | backbone.pt | impostor pools |
|-------|-------------|----------------|
| `cnn` | `training/artifacts/backbonecnn/models/` | `training/artifacts/backbonecnn/export/` |
| `convlstm` | `training/artifacts/convlstm/models/` | `training/artifacts/convlstm/export/` |
| `convlstm_bi` | `training/artifacts/convlstm_bi/models/` | `training/artifacts/convlstm_bi/export/` |

---

## Kết quả đã kiểm tra

| Model | TFLite size | Keras↔TFLite diff | AUC |
|-------|-------------|-------------------|-----|
| BackboneCNN | 308 KB | 5.4e-07 | 0.9970 ± 0.0096 |
| ConvLSTM | 768 KB | 6.9e-07 | 0.9964 ± 0.0109 |
| ConvLSTM-Bi | 808 KB | 8.9e-07 | 0.9964 ± 0.0117 |

Input TFLite: `[1, 200, 9]` float32 (NWC) — Output: `[1, 128]` float32 (embedding)

---

## Yêu cầu

```
Python 3.10+
torch >= 2.1
tensorflow >= 2.13
numpy < 2   (nếu dùng torch < 2.3)
```
