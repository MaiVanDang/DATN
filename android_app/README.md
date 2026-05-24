# Android App — Behavioural Authenticator

Ứng dụng xác thực hành vi (behavioural authentication) liên tục, chạy **hoàn toàn on-device**, không cần server, không gửi dữ liệu ra ngoài.

App kết hợp ba tín hiệu để tính một điểm tin cậy (trust score) duy nhất:

1. **Inertial (IMU)** — CNN 1D encoder (TFLite) biến cửa sổ 4 giây dữ liệu acc + gyro + mag thành embedding 128-D, so cosine với anchors của owner.
2. **Touch** — đặc trưng cử chỉ (tap / scroll / keystroke) được vector hoá 48-D rồi cho qua Random Forest (huấn luyện trên thiết bị).
3. **Fallback** — khi điểm tin cậy quá thấp, hiện overlay yêu cầu lắc điện thoại theo "mật khẩu" đã đăng ký (số 0–9 mã hoá thành số lần lắc).

Điểm cuối được smooth bằng EMA (α = 0.8, cửa sổ 5 lần), map sang ba trạng thái: **TRUSTED / WARNING / UNKNOWN**.

---

## Cấu trúc thư mục

```
android_app/
└── B_authenticator_app/                       # Android Studio project
    ├── README.md                              # Build / run / troubleshooting
    ├── QUICK_START.md                         # Hướng dẫn nhanh
    ├── CHANGES.md                             # Lịch sử thay đổi (mode walking/all)
    ├── build.gradle.kts                       # Top-level Gradle
    ├── settings.gradle.kts
    ├── gradle.properties
    └── app/
        ├── build.gradle.kts                   # App module: deps + AGP config
        ├── proguard-rules.pro
        └── src/main/
            ├── AndroidManifest.xml
            ├── assets/
            │   ├── walking/                   # Model train chỉ với hoạt động đi bộ
            │   │   ├── backbone.tflite        # CNN encoder, output 128-D
            │   │   ├── scaler_params.json     # Z-score normalization config
            │   │   ├── export_manifest.json   # Metadata + ngưỡng decision
            │   │   ├── touch_scaler.json      # Scaler cho touch RF
            │   │   ├── impostor_pool_inertial.npy
            │   │   └── impostor_pool_touch.npy
            │   └── all/                       # Model train trên mọi hoạt động
            │       └── (cùng 6 file như walking)
            ├── java/com/datn/authenticator/
            │   ├── AuthenticatorApp.kt        # Application — khởi tạo notification channel
            │   ├── model/
            │   │   ├── AuthState.kt           # enum TRUSTED / WARNING / UNKNOWN + ngưỡng
            │   │   ├── SensorWindow.kt        # (200, 9) tensor + channel index
            │   │   ├── ScalerParams.kt        # Z-score normalization (per-window / fitted)
            │   │   └── ExportManifest.kt      # Schema cho export_manifest.json
            │   ├── inference/
            │   │   ├── InferenceEngine.kt     # TFLite wrapper + cosine → trust score
            │   │   ├── SensorWindowCollector.kt # Thu 50 Hz IMU, build (200, 9) window
            │   │   ├── TouchCollector.kt      # Tap/scroll/keystroke → 48-D vector
            │   │   ├── OwnerProfile.kt        # Persist anchors + RF + fusion weight
            │   │   ├── AdaptiveAnchorBuffer.kt # Bổ sung anchor khi user TRUSTED ổn định
            │   │   ├── RandomForestClassifier.kt # Cây quyết định + RF on-device
            │   │   ├── ScoreAggregator.kt     # EMA smoothing → AuthState
            │   │   ├── FusionEngine.kt        # Trộn inertial + touch score, tune w
            │   │   └── NpyReader.kt           # Đọc .npy float32 trong assets
            │   ├── service/
            │   │   └── AuthenticationService.kt # Foreground service, capture loop
            │   ├── fallback/
            │   │   ├── ShakeDetector.kt       # Peak-count trên trục x của acc
            │   │   ├── PatternStorage.kt      # EncryptedSharedPreferences
            │   │   └── FallbackActivity.kt    # Overlay UI xác thực bằng lắc
            │   ├── ui/
            │   │   ├── ModeSelectActivity.kt    # Entry point — chọn Walking / All-action
            │   │   ├── OwnerEnrollmentActivity.kt # Thu 20 anchor IMU
            │   │   ├── TouchEnrollActivity.kt    # Tap + scroll + keystroke + train RF
            │   │   ├── FallbackEnrollActivity.kt # Đăng ký mẫu lắc bí mật
            │   │   └── QuizActivity.kt           # Màn dùng app + hiển thị trust score
            │   └── util/
            │       ├── ContextMode.kt         # enum WALKING / ALL + persistence
            │       ├── NotificationHelper.kt  # Notification channel
            │       └── BootReceiver.kt        # Auto-start service sau reboot
            └── res/
                ├── layout/  drawable/  values/  xml/  mipmap-anydpi-v26/
```

---

## Luồng sử dụng

```
ModeSelectActivity          → user chọn Walking hoặc All-action
       ↓
OwnerEnrollmentActivity     → thu 20 cửa sổ IMU 4 s = 20 anchor embeddings
       ↓
TouchEnrollActivity         → tap 15, scroll 8, keystroke 60 ký tự
                              → train RF_inertial + RF_touch + tune fusion weight
       ↓
FallbackEnrollActivity      → lắc 3 trial tự nhiên, app lấy median làm chữ ký
                              (không có "số bí mật" — nhịp lắc của bạn là chữ ký)
       ↓
QuizActivity                → màn dùng app bình thường (đọc câu hỏi, gõ note)
                              AuthenticationService chạy nền, cập nhật score mỗi 4 s
                              UNKNOWN → tự bung FallbackActivity yêu cầu lắc
```

Sau lần enroll đầu, mở app lại sẽ skip thẳng vào `QuizActivity` (state machine ở `ModeSelectActivity.onCreate`). Muốn enroll lại: nút **Đăng ký lại** ở Quiz, hoặc **Đổi mode** ở Owner enrollment.

---

## Scoring pipeline

```
SensorWindowCollector  ──► SensorWindow (200 × 9, 50 Hz × 4 s)
        │
        ▼
ScalerParams.normalize  (per-window z-score)
        │
        ▼
TFLite Interpreter (backbone.tflite)  ──► 128-D embedding
        │
        ▼
mean cosine sim  vs  20 core anchors  + N adaptive anchors
        │
        ▼
sigmoid(8 · (cosine − 0.25))           ──► p_inertial ∈ [0, 1]

TouchCollector.buildFeatureVector()    ──► 48-D vector
        │
        ▼
touch_scaler.json (z-score)
        │
        ▼
RF_touch.predictProba                  ──► p_touch ∈ [0, 1]

FusionEngine.fuse(p_inertial, p_touch, w)  ──► fused
        │
        ▼
ScoreAggregator (EMA α=0.8, window=5)  ──► S_t
        │
        ▼
AuthState.fromScore                    ──► TRUSTED / WARNING / UNKNOWN
```

Ngưỡng mặc định ở `AuthState`: TRUSTED ≥ 0.75, WARNING ≥ 0.45, còn lại UNKNOWN.

`SCORE_SCALE` và `SCORE_BIAS` (sigmoid curve) là hằng số trong `InferenceEngine.kt`:

| Triệu chứng | Điều chỉnh | Ảnh hưởng |
|---|---|---|
| Người lạ vẫn được TRUSTED (FAR cao) | `SCORE_BIAS` 0.25 → 0.35 | Strict hơn |
| Owner bị WARNING/UNKNOWN (FRR cao) | `SCORE_BIAS` 0.25 → 0.15 | Lỏng hơn |
| Score nhảy quanh 0.5 | `SCORE_SCALE` 8 → 12 | Quyết đoán hơn |
| Score gần như binary 0 hoặc 1 | `SCORE_SCALE` 8 → 5 | Mượt hơn |

---

## State machine của AuthenticationService

```
TRUSTED ───score < 0.75──► WARNING ───15 s không lên lại──► FallbackActivity
   ▲                          │
   │                          └──score < 0.45──► UNKNOWN ──► FallbackActivity (ngay)
   │
   └── pass fallback / score recovers
```

- Capture được trigger bởi 3 nguồn: `Sensor.TYPE_SIGNIFICANT_MOTION`, `Intent.ACTION_SCREEN_ON`, và một loop định kỳ 4 giây.
- Cooldown 5 giây giữa các capture để tránh đè quá nhiều.
- Sau khi fallback verified, score được reset lên 1.0, có 30 giây grace period trước khi có thể bung fallback lần nữa.
- Sau 3 lần fallback fail → `fallbackBlocked = true`, không bung fallback nữa cho tới khi user mở app và làm gì đó.

---

## Build & Run

**Yêu cầu:**
- Android Studio Hedgehog (2023.1.1) trở lên, JDK 17+
- Thiết bị Android API 29+ (Android 10+) có accelerometer + gyroscope (magnetometer optional)
- `compileSdk = 36`, `minSdk = 29`, Kotlin/JVM target 11

**Các bước:**

```bash
# 1. Mở project
cd android_app/B_authenticator_app
# File → Open → trỏ vào thư mục B_authenticator_app/

# 2. Build APK debug từ command line (tuỳ chọn)
./gradlew assembleDebug
# Output: app/build/outputs/apk/debug/app-debug.apk

# 3. Cài qua adb
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Khi cài lần đầu, app sẽ xin các quyền:
- `BODY_SENSORS` (auto) — đọc IMU
- `POST_NOTIFICATIONS` (Android 13+) — notification persistent của service
- `SYSTEM_ALERT_WINDOW` — **cần cấp thủ công** ở Settings → Apps → BioAuth → "Display over other apps" để fallback overlay hoạt động

---

## MOCK mode

Nếu `assets/<mode>/backbone.tflite` không có (hoặc bị compress), `InferenceEngine.load()` rơi xuống MOCK mode: embedding được sinh random theo seed timestamp. Lúc đó score sẽ dao động không có ý nghĩa — chỉ dùng để demo flow UI.

Để force MOCK: xoá `app/src/main/assets/walking/backbone.tflite`, build lại.

---

## Thay model mới

Để cập nhật backbone:

```bash
# Ví dụ với mode walking:
cp /path/to/new/backbone.tflite       app/src/main/assets/walking/
cp /path/to/new/scaler_params.json    app/src/main/assets/walking/
cp /path/to/new/export_manifest.json  app/src/main/assets/walking/
```

Lưu ý: embedding của model mới khác model cũ → mọi anchor đã enroll trở nên vô nghĩa. App phát hiện điều này qua hash file (chưa implement) — tạm thời, user phải **enroll lại** sau khi thay model. Cách đơn giản nhất: bấm "Đăng ký lại" trong Quiz.

Trong `build.gradle.kts` đã có `noCompress.addAll(listOf("tflite", "json", "npy"))` để AGP không nén các file này khi đóng gói APK.

---

## Yêu cầu thiết bị

- **Android 10+** (API 29) — cần `FOREGROUND_SERVICE_DATA_SYNC` được handle đúng
- **Accelerometer + Gyroscope** (bắt buộc)
- **Magnetometer** (khuyến nghị, có thể fallback nếu thiếu)
- **Significant motion sensor** (khuyến nghị) — nếu thiếu, service chỉ dựa vào screen-on + periodic loop

---

## Tài liệu kèm theo

- `B_authenticator_app/README.md` — chi tiết build, troubleshooting, permissions
- `B_authenticator_app/QUICK_START.md` — hướng dẫn nhanh + tuning hai hằng số sigmoid
- `B_authenticator_app/CHANGES.md` — diff khi thêm support 2 mode (walking / all)
