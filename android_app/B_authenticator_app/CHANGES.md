# Thay đổi: hỗ trợ chọn mode train (Walking / All-action)

## Tổng quan flow

```
Mở app
  │
  ▼
┌────────────────────────────┐
│ ModeSelectActivity (MỚI)   │   ← chỉ hiện lần đầu, hoặc khi user bấm "Đổi mode"
│  • Card "Walking"          │
│  • Card "All-action"       │
└────────────┬───────────────┘
             │ lưu vào SharedPreferences: bioauth_prefs.context_mode
             ▼
┌────────────────────────────┐
│ OwnerEnrollmentActivity    │   ← thêm thanh "Chế độ: X | [Đổi]"
└────────────┬───────────────┘
             ▼  (flow cũ giữ nguyên)
   TouchEnroll → FallbackEnroll → Quiz
```

Lần mở thứ 2 trở đi: `ModeSelectActivity` detect đã có mode + đã enroll
→ skip thẳng vào QuizActivity / FallbackEnroll / OwnerEnrollment tuỳ tiến độ.

---

## File mới

| File | Mục đích |
|---|---|
| `util/ContextMode.kt` | Enum `WALKING`/`ALL` + SharedPreferences helper + `assetPath()` + `isAvailable()` |
| `ui/ModeSelectActivity.kt` | Entry point mới, 2 card cho user chọn |
| `res/layout/activity_mode_select.xml` | Layout 2 MaterialCardView |
| `assets/walking/README.md` | Doc cho folder walking |
| `assets/all/README.md` | Doc cho folder all + hướng dẫn thay model thực |

## File modified

| File | Thay đổi |
|---|---|
| `AndroidManifest.xml` | LAUNCHER chuyển sang `ModeSelectActivity`; thêm declaration cho activity mới |
| `inference/InferenceEngine.kt` | `load()` và `loadTouchScaler()` nhận tham số `mode: ContextMode = ContextMode.loadOrDefault(context)`. Tất cả asset path đều prefix bằng `walking/` hoặc `all/` |
| `ui/TouchEnrollActivity.kt` | `trainModels()` đọc impostor pool từ subfolder của mode đang chọn |
| `ui/OwnerEnrollmentActivity.kt` | Thêm thanh hiển thị mode + nút "Đổi mode" (sẽ xoá profile + quay lại ModeSelectActivity). Bảo vệ thêm: nếu mode chưa được chọn (`loadSaved == null`), tự redirect về ModeSelectActivity |
| `res/layout/activity_owner_enrollment.xml` | Thêm `LinearLayout` chứa `modeIndicator` (TextView) + `btnChangeMode` (Button) ngay dưới title |
| `res/values/strings.xml` | Thêm 14 chuỗi mới cho mode select + change mode dialog |

## Reorganize `assets/`

**Trước:**
```
assets/
├── backbone.tflite
├── scaler_params.json
├── export_manifest.json
├── touch_scaler.json
├── impostor_pool_inertial.npy
└── impostor_pool_touch.npy
```

**Sau:**
```
assets/
├── walking/
│   ├── backbone.tflite              ← model walking-only (giữ nguyên file gốc)
│   ├── scaler_params.json
│   ├── export_manifest.json         ← thêm "context_mode": "walking"
│   ├── touch_scaler.json
│   ├── impostor_pool_inertial.npy
│   └── impostor_pool_touch.npy
└── all/
    ├── backbone.tflite              ← HIỆN ĐANG LÀ COPY CỦA WALKING (xem ghi chú)
    ├── scaler_params.json
    ├── export_manifest.json         ← "context_mode": "all"
    ├── touch_scaler.json
    ├── impostor_pool_inertial.npy
    └── impostor_pool_touch.npy
```

> **Quan trọng**: vì pipeline gốc chỉ cung cấp model `walking` được convert
> sang TFLite, folder `assets/all/` hiện chứa **bản copy y hệt** của walking
> để app build và chạy được ngay không lỗi MOCK mode. Khi bạn convert được
> `cnn_v2/models_all/backbone.pt` → ONNX → TFLite, hãy **replace 6 file** trong
> `assets/all/` bằng artifact thực từ `cnn_v2/export_all/` + model mới.

Nếu `assets/all/backbone.tflite` bị xoá, app sẽ:
- `ContextMode.isAvailable(this, ALL) == false`
- Card "All-action" hiển thị disabled + cảnh báo "⚠️ Thiếu file model"

---

## Behavior xoá profile khi đổi mode

Mode đổi → embedding của model khác → anchor cosine cũ vô nghĩa.  
Vì vậy mọi action đổi mode đều **xoá**:
- `OwnerProfile` (anchors, RF_inertial, RF_touch, fusion_w)
- `PatternStorage` (shake pattern)

Action xoá nằm ở 2 chỗ:
- `ModeSelectActivity.confirmAndContinue()` — khi user pick mode khác lúc đã có profile
- `OwnerEnrollmentActivity.confirmChangeMode()` — khi user bấm nút "Đổi" ở thanh mode indicator

Cả 2 đều có AlertDialog xác nhận trước khi xoá.

---

## Hyperparameter của touch RF không đổi

`touch_train.py` (mảng A) đã có sẵn flag `--context-mode {walking,all}`. Mảng B
chỉ cần load đúng impostor pool tương ứng — đã được handle qua
`ContextMode.assetPath()`.

## Service không cần đổi

`AuthenticationService` gọi `InferenceEngine.load(this, useGpu = true)` với
default tham số → tự lấy mode user đã chọn từ SharedPreferences. Không phá flow
service auto-start sau boot.

---

## Test checklist

- [ ] Build sạch lần đầu: app mở vào ModeSelectActivity, 2 card hiển thị OK
- [ ] Chọn Walking → vào OwnerEnrollment, indicator hiển thị "Chế độ: Walking"
- [ ] Enroll xong → Quiz hoạt động bình thường
- [ ] Đóng/mở lại app: skip thẳng vào Quiz (không hỏi mode lại)
- [ ] Quiz → bấm "Re-enroll" → vào OwnerEnrollment, mode vẫn giữ
- [ ] OwnerEnrollment → bấm "Đổi" → dialog confirm → quay về ModeSelect
- [ ] ModeSelect → chọn All-action → toast/redirect OK
- [ ] (Nếu xoá `assets/all/backbone.tflite`) card All-action disabled
- [ ] Logs: `Loading model for mode=walking: walking/backbone.tflite` xuất hiện
