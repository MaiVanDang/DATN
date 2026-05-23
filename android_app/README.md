# Android App — Authenticator

Ứng dụng xác thực hành vi liên tục chạy hoàn toàn on-device, không cần server.

## Cấu trúc

```
android_app/
└── B_authenticator_app/
    └── app/src/main/
        ├── assets/
        │   └── backbone.tflite        # CNN encoder (được copy từ ml_pipeline/export/)
        ├── java/com/datn/authenticator/
        │   ├── inference/
        │   │   ├── InferenceEngine.kt  # Scoring chính: cosine similarity + RF
        │   │   ├── OwnerProfile.kt     # Lưu anchors, RF models, EMA state
        │   │   └── TFLiteEncoder.kt    # Chạy TFLite → 128-D embedding
        │   ├── ui/
        │   │   ├── OwnerEnrollmentActivity.kt  # Đăng ký owner
        │   │   └── AuthDashboardActivity.kt    # Màn hình xác thực real-time
        │   └── sensors/
        │       └── SensorCollector.kt  # Thu thập IMU + touch data
        └── res/
```

## Cách hoạt động

1. **Enrollment**: Owner đi bộ và sử dụng điện thoại → thu thập 20 walking windows → lưu làm anchors (128-D embeddings)
2. **Authentication**: Liên tục thu 4s IMU window → TFLite → embedding → cosine similarity với anchors → score
3. **Touch scoring**: Gom gesture trong cửa sổ → 48-D feature vector → RandomForest on-device
4. **Fusion**: EMA (α=0.8) kết hợp inertial score + touch score → confidence

## Scoring (InferenceEngine.kt)

```
inertial_score = sigmoid(8 * (cosine_sim - 0.25))
# cosine_sim ≥ 0.38 → score > 0.50
# cosine_sim ≥ 0.57 → score > 0.75 (TRUSTED)
```

RF được train on-device nhưng chỉ dùng để log tham khảo, không ảnh hưởng score.

## Build & Run

1. Copy `backbone.tflite` vào `app/src/main/assets/` (hoặc chạy export script để tự động copy)
2. Mở project trong Android Studio
3. Build & deploy lên thiết bị Android (minSdk 26)
4. Enroll owner → chạy authentication

## Yêu cầu thiết bị

- Android 8.0+ (API 26)
- Accelerometer + Gyroscope + Magnetometer
- Màn hình cảm ứng
