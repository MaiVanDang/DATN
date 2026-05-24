# Walking assets

Model **train chỉ trên dữ liệu walking** (EER thấp nhất, chuyên cho người
chủ yếu dùng điện thoại lúc đi bộ).

File trong folder:
- `backbone.tflite`              — convert từ `cnn_v2/models_walking/backbone.pt`
- `scaler_params.json`           — `cnn_v2/export_walking/scaler_params.json`
- `export_manifest.json`         — `cnn_v2/export_walking/export_manifest.json`
- `touch_scaler.json`            — `cnn_v2/export_walking/touch_scaler.json`
- `impostor_pool_inertial.npy`   — `cnn_v2/export_walking/impostor_pool_inertial.npy`
- `impostor_pool_touch.npy`      — `cnn_v2/export_walking/impostor_pool_touch.npy`

Source checkpoint hiện tại: `cnn_v2 walking — AUC_fusion=0.9965, EER=1.20%`.
