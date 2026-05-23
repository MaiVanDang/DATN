# BioAuth Authenticator — Hướng dẫn nhanh

Project Android Studio này **đã được nhúng TFLite hoàn chỉnh**, chỉ cần mở và build.

## Yêu cầu
- Android Studio Hedgehog (2023.1.1) trở lên
- JDK 17+ (Android Studio đã có sẵn)
- Android phone với API 29+ (Android 10+), có sensor accelerometer + gyroscope + magnetometer

## Build và chạy (3 bước)

### 1. Mở project
```
File → Open → chọn thư mục B_authenticator_app/
```
Android Studio sẽ tự động sync Gradle (~2-3 phút lần đầu).

### 2. Cắm điện thoại
- Bật **Developer options** (Settings → About → Build number × 7)
- Bật **USB debugging**
- Cắm cáp, nhấn "Allow" trên điện thoại

### 3. Build & install
- Bấm **Run** (icon ▶ xanh) hoặc `Shift+F10`
- Đợi build (~30 giây) và install
- App tự mở trên điện thoại

## Flow sử dụng

```
┌─────────────────────────────────────────────────────┐
│  Mở app                                              │
│    ↓                                                 │
│  (lần đầu) Cấp quyền BODY_SENSORS + POST_NOTIF      │
│    ↓                                                 │
│  Bấm "Đăng ký mẫu lắc"                              │
│    → Dialog hỏi loại:                                │
│       • Đăng ký sinh trắc học (TFLite)  ← chọn ĐÂY  │
│       • Đăng ký mẫu lắc (fallback)                  │
│    ↓                                                 │
│  Bấm "Bắt đầu đăng ký"                              │
│  Cầm điện thoại tự nhiên ~15 giây                   │
│  Progress bar đếm 1/6 → 6/6                         │
│    ↓                                                 │
│  Quay về Main, bấm "Đăng ký mẫu lắc" lần nữa        │
│  Chọn "Đăng ký mẫu lắc (fallback)"                  │
│  Đăng ký pattern lắc số bí mật (vd: lắc 5 lần)      │
│    ↓                                                 │
│  Quay về Main → bấm "Khởi động dịch vụ"             │
│                                                      │
│  Service chạy ngầm — mỗi 2s tính trust score:        │
│  • TRUSTED (xanh)  — score > 75%                    │
│  • WARNING (vàng)  — score 45-75%                   │
│  • UNKNOWN (đỏ)    — score < 45%                    │
│                                                      │
│  Khi UNKNOWN → app yêu cầu lắc theo pattern bí mật  │
└─────────────────────────────────────────────────────┘
```

## Đã nhúng những gì

```
app/src/main/assets/
├── backbone.tflite        ← 317 KB, model CNN encoder em train V5
├── scaler_params.json     ← config z-score normalization
├── export_manifest.json   ← contract metadata
├── impostor_pool_*.npy    ← optional, để dành V2 fusion
├── touch_scaler.json      ← optional
└── model_card.json        ← optional, debug
```

Model spec:
- Input: `[1, 9, 100]` (9 sensors × 100 timesteps @ 50Hz = 2s window)
- Output: 128-D embedding
- Latency: ~2-5ms trên Android mid-range (CPU 4-thread)
- **GPU delegate đã bỏ** trong V2 vì model quá nhỏ (CPU đã thừa nhanh)

## Cấu trúc code

| File | Chức năng |
|---|---|
| `inference/InferenceEngine.kt` | Load TFLite, extract embedding, compute cosine sim → trust score |
| `inference/OwnerProfile.kt`    | Lưu 6 anchor embeddings vào internal storage (3KB binary) |
| `inference/SensorWindowCollector.kt` | Đọc 9-channel sensors @ 50Hz |
| `inference/ScoreAggregator.kt` | EMA smooth trust score qua 5 windows gần nhất |
| `service/AuthenticationService.kt` | Foreground service, trigger by SCREEN_ON + motion |
| `ui/MainActivity.kt` | Realtime status + score display |
| `ui/OwnerEnrollmentActivity.kt` | UI thu 6 anchors |
| `ui/EnrollmentActivity.kt` | UI đăng ký pattern lắc (fallback) |
| `fallback/FallbackActivity.kt` | Overlay khi UNKNOWN — yêu cầu lắc bí mật |
| `fallback/ShakeDetector.kt` | Đếm số lần lắc |

## Troubleshooting

### App vào MOCK mode
Log: `W InferenceEngine: backbone.tflite not found. Running in MOCK mode.`
→ File assets bị nén. Kiểm tra `app/build.gradle.kts` có `noCompress.addAll(listOf("tflite", "json", "npy"))`.

### Trust score luôn 0.5
→ Em chưa enroll sinh trắc học. Vào Main → "Đăng ký mẫu lắc" → chọn "Đăng ký sinh trắc học".

### Trust score luôn cao kể cả người khác cầm
→ Tăng `SCORE_BIAS` trong `InferenceEngine.kt` (line ~178) từ 0.3 lên 0.4.

### Trust score luôn thấp kể cả em cầm
→ Giảm `SCORE_BIAS` xuống 0.2.

### Service không start
→ Cấp quyền `BODY_SENSORS` và `POST_NOTIFICATIONS` (Settings → Apps → BioAuth → Permissions).

## Tuning hai hằng số quan trọng

```kotlin
// File: InferenceEngine.kt, near line 175
private const val SCORE_SCALE = 8f      // độ "dốc" sigmoid (cao → quyết liệt hơn)
private const val SCORE_BIAS  = 0.3f    // điểm trung tâm sigmoid
```

| Triệu chứng | Sửa | Hiệu quả |
|---|---|---|
| FAR cao (kẻ lạ được trust) | tăng `SCORE_BIAS` 0.3 → 0.4 | strict hơn |
| FRR cao (em bị reject) | giảm `SCORE_BIAS` 0.3 → 0.2 | dễ dãi hơn |
| Score nhảy quanh 0.5 | tăng `SCORE_SCALE` 8 → 12 | decisive hơn |
| Score binary 0 hoặc 1 | giảm `SCORE_SCALE` 8 → 5 | smooth hơn |

## Thư mục project

```
B_authenticator_app/
├── app/                          ← app module
│   ├── src/main/
│   │   ├── assets/               ← model + config files
│   │   ├── java/com/datn/authenticator/
│   │   │   ├── inference/        ← TFLite + scoring logic
│   │   │   ├── service/          ← background service
│   │   │   ├── ui/               ← MainActivity + enrollment screens
│   │   │   ├── fallback/         ← shake pattern fallback
│   │   │   └── util/             ← helpers
│   │   ├── res/                  ← layouts + strings
│   │   └── AndroidManifest.xml
│   └── build.gradle.kts
├── build.gradle.kts              ← top-level config
├── gradle.properties
├── settings.gradle.kts
├── gradle/                       ← gradle wrapper
└── README.md
```
