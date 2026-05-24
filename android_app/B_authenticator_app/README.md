# B_authenticator_app — Build & Run

Project Android Studio cho app **Behavioural Authenticator**. Tất cả model TFLite + asset cần thiết đã được nhúng sẵn trong `app/src/main/assets/`, mở project là build được ngay.

> Mô tả pipeline, scoring, và state machine: xem [`../README.md`](../README.md).
> Hướng dẫn nhanh: xem [`QUICK_START.md`](QUICK_START.md).
> Lịch sử thay đổi (mode walking / all-action): xem [`CHANGES.md`](CHANGES.md).

---

## Yêu cầu

- **Android Studio Hedgehog (2023.1.1)** trở lên
- **JDK 17+** (đi kèm Android Studio)
- Thiết bị Android **API 29+** (Android 10+) có accelerometer + gyroscope, khuyến nghị có magnetometer

Cấu hình build (xem `app/build.gradle.kts`):

| | |
|---|---|
| `compileSdk` | 36 |
| `targetSdk`  | 36 |
| `minSdk`     | 29 |
| JVM target   | 11 |
| TFLite       | `org.tensorflow:tensorflow-lite:2.14.0` (CPU only) |

---

## Mở project trong Android Studio

1. `File → Open` → trỏ vào thư mục `B_authenticator_app/`
2. Bấm **Trust project** khi được hỏi
3. Đợi Gradle sync (~2 phút lần đầu — sẽ tải `gradle-wrapper.jar` về tự động)
4. `Run → Run 'app'`

> **Lưu ý:** thư mục `gradle/wrapper/` chỉ chứa `gradle-wrapper.properties`. `gradle-wrapper.jar` sẽ được Android Studio tự tải xuống lần sync đầu tiên. Cần kết nối Internet.

## Build từ command line

```bash
cd B_authenticator_app
./gradlew assembleDebug
# Output: app/build/outputs/apk/debug/app-debug.apk

# Cài qua adb:
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

(Trên Linux/macOS: `chmod +x gradlew` nếu cần.)

---

## Khi chạy lần đầu

App sẽ:

1. Hiện **ModeSelectActivity** — chọn `Walking` hoặc `All-action`
2. Yêu cầu cấp `BODY_SENSORS` và `POST_NOTIFICATIONS` (Android 13+)
3. Vào **OwnerEnrollmentActivity** — bấm "Bắt đầu" rồi cầm máy tự nhiên ~80 giây để thu 20 cửa sổ IMU 4 giây
4. Vào **TouchEnrollActivity** — tap 15 lần, scroll 8 lần, gõ 60 ký tự → app tự train RF
5. Vào **FallbackEnrollActivity** — lắc điện thoại 3 trial theo nhịp tự nhiên của bạn → app lấy median số lần lắc làm "chữ ký", dùng để xác thực dự phòng khi score xuống thấp
6. Vào **QuizActivity** — màn dùng app bình thường. `AuthenticationService` chạy nền, hiện notification persistent với điểm tin cậy hiện tại

Sau enroll, mỗi lần mở app sẽ vào thẳng QuizActivity (skip enroll). Muốn xoá profile và làm lại: bấm **"Đăng ký lại"** trong Quiz, hoặc **"Đổi mode"** ở Owner enrollment.

`SYSTEM_ALERT_WINDOW` cần cấp **thủ công** ở Settings → Apps → BioAuth → "Display over other apps" để fallback overlay có thể bung lên khi đang ở app khác.

---

## Permissions

| Permission | Khi nào cần | Ai cấp |
|---|---|---|
| `BODY_SENSORS` | Đọc accelerometer / gyroscope / magnetometer | runtime, ở enrollment |
| `FOREGROUND_SERVICE` + `FOREGROUND_SERVICE_DATA_SYNC` | Service chạy nền | install-time |
| `POST_NOTIFICATIONS` | Notification persistent (Android 13+) | runtime |
| `SYSTEM_ALERT_WINDOW` | `FallbackActivity` overlay lên app khác | **thủ công, Settings** |
| `HIGH_SAMPLING_RATE_SENSORS` | 50 Hz trên Android 12+ | install-time |
| `RECEIVE_BOOT_COMPLETED` | Auto-start sau reboot (qua `BootReceiver`) | install-time |

---

## MOCK mode

Nếu `assets/<mode>/backbone.tflite` không có hoặc bị compress, `InferenceEngine.load()` rơi xuống MOCK mode: embedding random theo seed timestamp. Lúc đó:

- Log: `W InferenceEngine: walking/backbone.tflite not found. Running in MOCK mode.`
- Score sẽ dao động không có ý nghĩa — chỉ demo flow

Nguyên nhân hay gặp: AGP nén file `.tflite` khi đóng gói. Trong project này đã có:

```kotlin
// app/build.gradle.kts
androidResources {
    noCompress.addAll(listOf("tflite", "json", "npy"))
}
```

nên không bị nén. Nếu vẫn MOCK: kiểm tra file thực sự tồn tại trong `assets/walking/` hoặc `assets/all/`.

---

## Thay model thực

```bash
# Mode walking:
cp /path/to/new/backbone.tflite       app/src/main/assets/walking/
cp /path/to/new/scaler_params.json    app/src/main/assets/walking/
cp /path/to/new/export_manifest.json  app/src/main/assets/walking/
cp /path/to/new/touch_scaler.json     app/src/main/assets/walking/
cp /path/to/new/impostor_pool_*.npy   app/src/main/assets/walking/

# Hoặc mode all (cấu trúc tương tự, thư mục assets/all/)
```

Build lại — `InferenceEngine` sẽ tự phát hiện và rời MOCK mode.

> **Quan trọng:** thay model = embedding khác = anchors cũ vô nghĩa. Cần **enroll lại** sau khi thay. Bấm "Đăng ký lại" trong Quiz hoặc gỡ và cài lại app.

---

## Cấu trúc thư mục (rút gọn)

```
B_authenticator_app/
├── README.md                  ← bạn đang đọc
├── QUICK_START.md
├── CHANGES.md
├── build.gradle.kts           ← top-level Gradle
├── settings.gradle.kts
├── gradle.properties
├── gradle/wrapper/            ← chỉ có .properties, .jar tự tải
├── local.properties           ← SDK path (gitignored)
└── app/
    ├── build.gradle.kts       ← app module
    ├── proguard-rules.pro
    └── src/main/
        ├── AndroidManifest.xml
        ├── assets/walking/    ← model + scaler + impostor pool cho mode walking
        ├── assets/all/        ← tương tự cho mode all-action
        ├── java/com/datn/authenticator/
        │   ├── AuthenticatorApp.kt
        │   ├── model/         ← AuthState, SensorWindow, ScalerParams, ExportManifest
        │   ├── inference/     ← InferenceEngine, OwnerProfile, RF, ScoreAggregator, …
        │   ├── service/       ← AuthenticationService (foreground)
        │   ├── fallback/      ← ShakeDetector, PatternStorage, FallbackActivity
        │   ├── ui/            ← ModeSelect → OwnerEnroll → TouchEnroll → FallbackEnroll → Quiz
        │   └── util/          ← ContextMode, NotificationHelper, BootReceiver
        └── res/               ← layout, drawable, values, xml, mipmap-anydpi-v26
```

Mô tả từng file: xem [`../README.md`](../README.md).

---

## Troubleshooting

**Gradle sync fails: "Could not download gradle-wrapper.jar"**
Bật mạng và Sync lại; Android Studio sẽ download `gradle-wrapper.jar` tự động lần đầu.

**`Unresolved reference: BuildConfig` đỏ trong editor**
`Build → Clean Project`, rồi `Build → Rebuild Project`. `BuildConfig` được generate khi build, không phải khi sync.

**Service không hiện notification**
Android 13+ yêu cầu cấp `POST_NOTIFICATIONS` runtime. Vào Settings → Apps → BioAuth → Notifications → Allow.

**FallbackActivity không bung lên dù score thấp**
Cần cấp `SYSTEM_ALERT_WINDOW` thủ công ở Settings → Apps → BioAuth → "Display over other apps".

**TFLite crash khi load model**
Kiểm tra `app/src/main/assets/<mode>/backbone.tflite` có tồn tại. `noCompress.add("tflite")` trong `build.gradle.kts` đảm bảo file không bị nén.

**Inference rất chậm**
Project này dùng CPU 4-thread, model chỉ ~317 KB nên latency CPU thường 2–5 ms. Nếu chậm hơn nhiều: kiểm tra log có rơi vào MOCK mode không (mock cũng nhanh, nhưng không có ý nghĩa). Có thể chuyển sang 1-thread bằng `InferenceEngine.load(this, forceCpuOneThread = true)` nếu nghi do contention.

**Đổi mode (walking ↔ all) nhưng score không đổi**
Khi đổi mode, app sẽ xoá `OwnerProfile` + `PatternStorage` (có dialog xác nhận) và bắt enroll lại. Nếu chưa enroll lại thì pipeline vẫn dùng anchors cũ + model mới → vô nghĩa. Hoàn tất enroll mới rồi mới đánh giá.
