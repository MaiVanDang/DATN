# BioAuth Authenticator — TFLite Edition

> 📖 **Bắt đầu nhanh**: xem [QUICK_START.md](QUICK_START.md) — chỉ cần Android Studio + cáp USB.

> 🎯 Project này đã được **nhúng TFLite hoàn chỉnh** từ model train V5 (19 users, EER median 1.7e-3).
> Toàn bộ pipeline `assets/backbone.tflite` + on-device biometric enrollment đã sẵn sàng.

---

# Mảng B — App Authenticator (Android)

App xác thực hành vi liên tục dùng CNN 1D + cử chỉ lắc (fallback). Mọi component map đến báo cáo (ghi chú trong từng file `.kt`).

## Mở project

1. **Android Studio Hedgehog (2023.1.1) hoặc mới hơn**.
2. File → Open → trỏ vào folder `B_authenticator_app/`.
3. Khi Android Studio hỏi sync Gradle, bấm "Trust project" → OK.
4. Đợi Gradle sync (~2 phút lần đầu).

> **Quan trọng**: Folder `gradle/wrapper/` chỉ có `gradle-wrapper.properties` — thiếu `gradle-wrapper.jar`. Android Studio sẽ tự tải xuống khi sync. Nếu lỗi network, vào File → Settings → Build → Gradle → "Use Gradle from: gradle-wrapper.properties" và bấm "Sync".

## Chạy ngay (MOCK MODE — không cần model)

App sẽ tự chạy ở chế độ mock nếu không tìm thấy `model.tflite` trong `app/src/main/assets/`. Mock predictor dao động tự nhiên giữa TRUSTED ↔ WARNING ↔ UNKNOWN mỗi ~10 giây để bạn demo flow xác thực + fallback ngay khi chưa train xong.

1. Cắm điện thoại Android (SDK 29+, API ≥ Android 10).
2. Run → Run 'app'.
3. Cấp permissions: BODY_SENSORS + POST_NOTIFICATIONS.
4. Bấm "Đăng ký mẫu lắc" → chọn digit → lắc 3 lần × 10 giây → Lưu.
5. Bấm "Khởi động dịch vụ" → notification persistent xuất hiện.
6. Đợi ~30s — score sẽ dao động, một lúc nó sẽ rớt < 0.45 → FallbackActivity bật lên.

## Chạy với model thực

Sau khi mảng A export thành công cho owner X (vd `mvdang`):

```bash
cp ../A_export_pipeline/export/mvdang/model.tflite          app/src/main/assets/
cp ../A_export_pipeline/export/mvdang/scaler_params.json    app/src/main/assets/
cp ../A_export_pipeline/export/mvdang/export_manifest.json  app/src/main/assets/
```

Build lại — InferenceEngine sẽ tự phát hiện và disable mock mode.

## Cấu trúc code (map sang báo cáo)

```
app/src/main/java/com/datn/authenticator/
├── AuthenticatorApp.kt              # Application class
├── model/
│   ├── AuthState.kt                 # Mục 5.3.4 — TRUSTED/WARNING/UNKNOWN
│   ├── SensorWindow.kt              # Mục 3.3.3 — (100,9) tensor
│   ├── ScalerParams.kt              # Mục 4.4.5 — Z-score normalization
│   └── ExportManifest.kt            # Đọc metadata từ A_export_pipeline
├── inference/
│   ├── InferenceEngine.kt           # Mục 5.3.1 + 5.5 — TFLite wrapper, GPU/CPU
│   ├── ScoreAggregator.kt           # Mục 5.3.3 — EMA score S_t
│   └── SensorWindowCollector.kt     # Mục 5.3.2 — capture + bin sensors
├── service/
│   └── AuthenticationService.kt     # Mục 5.3.1 — foreground service
├── fallback/
│   ├── ShakeDetector.kt             # Mục 3.6 + 5.4.1 — peak detection
│   ├── PatternStorage.kt            # Mục 5.4.1 — EncryptedSharedPreferences
│   └── FallbackActivity.kt          # Mục 5.4 — overlay UI
├── ui/
│   ├── MainActivity.kt              # Status screen
│   ├── EnrollmentActivity.kt        # Đăng ký mẫu lắc
│   ├── SettingsActivity.kt          # Ngưỡng, auto-start
│   └── BenchmarkActivity.kt         # Mảng C — Bảng 5.5
└── util/
    ├── NotificationHelper.kt
    └── BootReceiver.kt              # Auto-start sau reboot
```

## Permissions (Bảng 5.1)

| Permission | Khi nào cần |
|---|---|
| `BODY_SENSORS` | Đọc accelerometer/gyroscope ở mọi lúc |
| `FOREGROUND_SERVICE` + `FOREGROUND_SERVICE_DATA_SYNC` | Service chạy nền |
| `POST_NOTIFICATIONS` | Notification persistent (Android 13+) |
| `SYSTEM_ALERT_WINDOW` | FallbackActivity overlay (cấp thủ công ở Settings) |
| `HIGH_SAMPLING_RATE_SENSORS` | 50 Hz trên Android 12+ |
| `RECEIVE_BOOT_COMPLETED` | Auto-start sau reboot |

`BODY_SENSORS` và `POST_NOTIFICATIONS` xin trong runtime ở MainActivity. `SYSTEM_ALERT_WINDOW` cần user cấp thủ công ở Settings → Apps → BioAuth → "Display over other apps".

## Demo cho hội đồng (kịch bản chi tiết: xem `D_demo/demo_script.md`)

## Troubleshooting

- **Gradle sync fails: "Could not download gradle-wrapper.jar"** — Bật mạng và Sync lại; Android Studio sẽ download wrapper jar tự động.

- **`Unresolved reference: BuildConfig` lỗi đỏ** — Build → Clean Project, rồi Build → Rebuild Project. `BuildConfig` được generate khi build (không phải khi sync).

- **Service không hiện notification** — Android 13+ yêu cầu user cấp `POST_NOTIFICATIONS`. Vào Settings → Apps → BioAuth → Notifications → Allow.

- **Không thấy FallbackActivity bật lên dù score thấp** — Cần cấp `SYSTEM_ALERT_WINDOW` thủ công.

- **TFLite crash khi load model** — Kiểm tra file `model.tflite` có nằm trong `app/src/main/assets/` không. AGP 8.x compress mặc định, nhưng `noCompress.add("tflite")` trong `build.gradle.kts` đã giữ nó nguyên.

- **Inference rất chậm trên GPU** — Một số máy GPU delegate khởi tạo lâu (cold start ~200ms) nhưng warm latency vẫn nhỏ. Xem BenchmarkActivity để xác minh.

## Build APK

```bash
cd B_authenticator_app
./gradlew assembleDebug
# Output: app/build/outputs/apk/debug/app-debug.apk
```

(Trên Linux/macOS chmod +x gradlew nếu cần.)
