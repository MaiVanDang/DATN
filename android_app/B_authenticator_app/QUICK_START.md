# BioAuth Authenticator — Hướng dẫn nhanh

Project này đã nhúng sẵn TFLite + asset hoàn chỉnh trong `app/src/main/assets/`, mở Android Studio và build là chạy được.

## Yêu cầu
- Android Studio Hedgehog (2023.1.1) trở lên
- JDK 17+ (Android Studio đã có sẵn)
- Thiết bị Android API 29+ (Android 10+), có accelerometer + gyroscope (magnetometer khuyến nghị)

## Build và chạy (3 bước)

### 1. Mở project
```
File → Open → chọn thư mục B_authenticator_app/
```
Android Studio sẽ tự sync Gradle (~2-3 phút lần đầu).

### 2. Cắm điện thoại
- Bật **Developer options** (Settings → About → Build number × 7)
- Bật **USB debugging**
- Cắm cáp, nhấn "Allow" trên điện thoại

### 3. Build & install
- Bấm **Run** (▶ xanh) hoặc `Shift+F10`
- Đợi build (~30 giây) và install
- App tự mở trên điện thoại

## Flow sử dụng

```
┌─────────────────────────────────────────────────────────┐
│  Mở app  →  ModeSelectActivity                          │
│            Chọn Walking hoặc All-action                  │
│                                                          │
│  Cấp quyền BODY_SENSORS + POST_NOTIFICATIONS             │
│                                                          │
│            OwnerEnrollmentActivity                       │
│            Cầm máy tự nhiên ~80 giây → 20 anchor IMU    │
│                                                          │
│            TouchEnrollActivity                           │
│            Tap 15 lần → Scroll 8 lần → Gõ 60 ký tự     │
│            → app tự train RF_inertial + RF_touch         │
│                                                          │
│            FallbackEnrollActivity                        │
│            Lắc 3 trial theo nhịp tự nhiên → lưu median   │
│                                                          │
│            QuizActivity                                  │
│            Màn quiz (đọc + tap + gõ) — service nền      │
│            tính trust score mỗi 4 giây                  │
│                                                          │
│  Trạng thái hiển thị qua notification + Quiz UI:        │
│    • TRUSTED (xanh)  — score ≥ 0.75                     │
│    • WARNING (vàng)  — score 0.45 – 0.75                │
│    • UNKNOWN (đỏ)    — score < 0.45                     │
│                                                          │
│  WARNING 15s liên tục, hoặc UNKNOWN tức thì:            │
│    → FallbackActivity bung lên, yêu cầu lắc đúng pattern│
└─────────────────────────────────────────────────────────┘
```

> Cần cấp `SYSTEM_ALERT_WINDOW` thủ công ở Settings → Apps → BioAuth → "Display over other apps" để FallbackActivity overlay được lên app khác.

## Đã nhúng những gì

```
app/src/main/assets/
├── walking/                        # mode: chỉ hoạt động đi bộ
│   ├── backbone.tflite             # ~317 KB, CNN encoder, output 128-D
│   ├── scaler_params.json          # config z-score normalization
│   ├── export_manifest.json        # metadata + ngưỡng decision
│   ├── touch_scaler.json           # scaler cho touch RF
│   ├── impostor_pool_inertial.npy
│   └── impostor_pool_touch.npy
└── all/                            # mode: mọi hoạt động
    └── (cùng 6 file)
```

Model spec:
- Input: `[1, 200, 9]` (200 timesteps × 9 sensor channels @ 50 Hz = 4 s window)
- Output: 128-D embedding
- Backend: TFLite CPU 4-thread (GPU delegate đã được bỏ — model nhỏ, CPU đủ nhanh)
- Latency: ~2-5 ms trên Android mid-range

## Cấu trúc code (chính)

| File | Chức năng |
|---|---|
| `inference/InferenceEngine.kt`      | Load TFLite, extract embedding, cosine sim → trust score |
| `inference/OwnerProfile.kt`         | Persist anchors + RF + fusion weight |
| `inference/AdaptiveAnchorBuffer.kt` | Bổ sung anchor khi score cao ổn định |
| `inference/SensorWindowCollector.kt`| Đọc 9-channel IMU @ 50 Hz, build window |
| `inference/TouchCollector.kt`       | Tap/scroll/keystroke → 48-D vector |
| `inference/RandomForestClassifier.kt`| RF on-device cho inertial + touch |
| `inference/FusionEngine.kt`         | Trộn inertial + touch score, tune w |
| `inference/ScoreAggregator.kt`      | EMA smoothing → AuthState |
| `service/AuthenticationService.kt`  | Foreground service, capture loop, fallback orchestration |
| `ui/ModeSelectActivity.kt`          | Entry point, chọn mode |
| `ui/OwnerEnrollmentActivity.kt`     | Thu 20 anchor IMU |
| `ui/TouchEnrollActivity.kt`         | Guided tap/scroll/typing + train RF |
| `ui/FallbackEnrollActivity.kt`      | Đăng ký mẫu lắc bí mật |
| `ui/QuizActivity.kt`                | Màn dùng app + hiển thị status realtime |
| `fallback/FallbackActivity.kt`      | Overlay yêu cầu lắc đúng pattern |
| `fallback/ShakeDetector.kt`         | Đếm số lần lắc |

## Troubleshooting

### App vào MOCK mode
Log: `W InferenceEngine: walking/backbone.tflite not found. Running in MOCK mode.`
→ Kiểm tra file có tồn tại trong `assets/walking/` chưa. Cũng kiểm tra `app/build.gradle.kts` có `noCompress.addAll(listOf("tflite", "json", "npy"))` (đã có sẵn).

### Trust score luôn 0.5
→ Chưa enroll IMU xong. Vào ModeSelect → OwnerEnrollment → hoàn tất 20 anchor.

### Trust score luôn cao kể cả người khác cầm
→ Tăng `SCORE_BIAS` trong `InferenceEngine.kt` (gần dòng 175) từ 0.25 lên 0.35.

### Trust score luôn thấp kể cả owner cầm
→ Giảm `SCORE_BIAS` xuống 0.15.

### Service không start
→ Cấp quyền `BODY_SENSORS` và `POST_NOTIFICATIONS` (Settings → Apps → BioAuth → Permissions).

### FallbackActivity không bung lên
→ Cần cấp `SYSTEM_ALERT_WINDOW` thủ công (Settings → Apps → BioAuth → "Display over other apps").

## Tuning hai hằng số quan trọng

```kotlin
// File: inference/InferenceEngine.kt, trong companion object
private const val SCORE_SCALE = 8f      // độ "dốc" sigmoid
private const val SCORE_BIAS  = 0.25f   // điểm trung tâm sigmoid
```

| Triệu chứng | Sửa | Hiệu quả |
|---|---|---|
| FAR cao (kẻ lạ vẫn TRUSTED) | tăng `SCORE_BIAS` 0.25 → 0.35 | Strict hơn |
| FRR cao (owner bị reject)    | giảm `SCORE_BIAS` 0.25 → 0.15 | Lỏng hơn |
| Score nhảy quanh 0.5          | tăng `SCORE_SCALE` 8 → 12     | Quyết đoán hơn |
| Score gần như 0 hoặc 1        | giảm `SCORE_SCALE` 8 → 5      | Mượt hơn |

## Thư mục project

```
B_authenticator_app/
├── README.md                  ← build + troubleshooting chi tiết
├── QUICK_START.md             ← bạn đang đọc
├── CHANGES.md                 ← lịch sử thay đổi
├── app/
│   ├── src/main/
│   │   ├── assets/walking/    ← model + scaler cho mode walking
│   │   ├── assets/all/        ← model + scaler cho mode all-action
│   │   ├── java/com/datn/authenticator/
│   │   │   ├── inference/     ← TFLite + scoring + RF
│   │   │   ├── service/       ← background service
│   │   │   ├── ui/            ← 5 activity của enrollment + quiz
│   │   │   ├── fallback/      ← shake-pattern fallback
│   │   │   ├── model/         ← data class shared
│   │   │   └── util/          ← helpers, BootReceiver
│   │   ├── res/               ← layouts + strings + themes
│   │   └── AndroidManifest.xml
│   ├── build.gradle.kts
│   └── proguard-rules.pro
├── build.gradle.kts           ← top-level
├── gradle.properties
├── settings.gradle.kts
└── gradle/                    ← gradle wrapper
```
