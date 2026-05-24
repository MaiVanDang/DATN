# All-action assets

Đặt file của model **train trên tất cả hoạt động (all-action)** vào folder này.
Lấy từ pipeline export ứng với mode `all` (vd: `cnn_v2/export_all/`).

File cần có:
- `backbone.tflite`              — convert từ `models_all/backbone.pt` → ONNX → TFLite
- `scaler_params.json`           — copy từ `export_all/scaler_params.json`
- `export_manifest.json`         — copy / sửa `context_mode = "all"`
- `touch_scaler.json`            — `export_all/touch_scaler.json`
- `impostor_pool_inertial.npy`   — `export_all/impostor_pool_inertial.npy`
- `impostor_pool_touch.npy`      — `export_all/impostor_pool_touch.npy`

> Mặc định folder này được build kèm copy của model walking để app chạy được ngay
> ở mode All-action (chỉ là model thật chưa được swap). Khi swap xong, app sẽ
> tự dùng model all-action thực.
>
> Nếu xoá folder này (hoặc thiếu `backbone.tflite`), `ModeSelectActivity` sẽ
> tự disable nút "All-action" với hint cho user.
