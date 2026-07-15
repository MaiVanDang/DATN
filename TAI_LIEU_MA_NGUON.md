# Tài liệu mã nguồn — Hệ thống xác thực sinh trắc học hành vi liên tục trên Android

Tài liệu mô tả **chức năng từng file mã nguồn**, **các hàm chính** kèm **nhiệm vụ** và **ý nghĩa các tham số quan trọng** đóng góp cho đồ án.

Hệ thống gồm 4 thành phần:

| Thư mục | Vai trò | Ngôn ngữ |
|---------|---------|----------|
| `ml_pipeline/` | Tiền xử lý dữ liệu, thống kê, vẽ biểu đồ, xuất mô hình TFLite | Python |
| `web_demo/` | Lõi xác thực (cosine + RF + fusion) và web demo trực quan | Python / Streamlit |
| `android_app/B_authenticator_app/` | Ứng dụng xác thực hành vi liên tục (sản phẩm chính) | Kotlin |
| `data_collectiom/DataCollectV2/` | Ứng dụng thu thập dữ liệu để xây dựng dataset | Kotlin |

Tham số dùng chung toàn hệ thống: **tần số lấy mẫu 50 Hz**, **cửa sổ 4 giây = 200 mẫu × 9 kênh** (acc, gyro, mag mỗi loại 3 trục), **embedding 128 chiều**, **đặc trưng touch 33 chiều**.

---

# 1. `ml_pipeline/` — Pipeline xử lý dữ liệu (Python)

## 1.1. `step1_quality_check.py` — Kiểm tra chất lượng dữ liệu thô
Quét toàn bộ `data/<user>/<session>/` để kiểm xem mỗi người dùng đã thu **đủ** dữ liệu quán tính (thời lượng) và touch (số sự kiện) theo chỉ tiêu chưa.

| Hàm | Nhiệm vụ | Tham số quan trọng |
|-----|----------|--------------------|
| `fmt_time(sec)` | Định dạng giây → chuỗi `MpSSs` (vd 19,5s → `0p19s`) | `sec`: số giây |
| `check_status(value, target, unit)` | So sánh giá trị thực với chỉ tiêu, trả "✓ ĐỦ"/"✗ THIẾU" kèm chênh lệch | `unit="time"` để hiển thị dạng thời gian |
| `get_duration(df, filename)` | Thời lượng thu **thực tế** (giây) sau khi loại các khoảng trống và bỏ đoạn quá ngắn | `df`: DataFrame một file CSV |
| `collect_inertial(session_dir)` | Tổng thời lượng 3 hoạt động (walking/standing/sitting) trong 1 phiên | |
| `collect_touch_counts(session_dir, session_id)` | Đếm số sự kiện tap/scroll trong phiên (gọi `process_tap`, `process_scroll`) | |
| `check_session` / `check_user` | In báo cáo theo phiên / theo người; xác minh đạt chỉ tiêu `INERTIAL_TARGET_MIN` (18 phút/hoạt động) và `TOUCH_TARGET` (600 tap, 600 scroll) | |

`MAX_GAP_SEC = 5/HZ`: ngưỡng coi là **gián đoạn tín hiệu** (5 chu kỳ lấy mẫu).

## 1.2. `step2_preprocess.py` — Tiền xử lý (file lõi của pipeline)
Chuyển dữ liệu CSV thô thành các cửa sổ chuẩn hóa `.npy` (quán tính) và bảng đặc trưng `.csv` (touch). **Mọi script khác đều dùng lại các hàm của file này** để đảm bảo nhất quán.

**Hằng số then chốt:** `HZ=50`, `WINDOW_SIZE=200` (4 giây), `STRIDE=20` (bước trượt → cửa sổ chồng lấn nhiều), `MAX_GAP_SEC=5/HZ`, `SENSOR_COLS` (9 kênh theo đúng thứ tự acc→gyro→mag).

| Hàm | Nhiệm vụ | Tham số quan trọng |
|-----|----------|--------------------|
| `_ts_info(df)` | Xác định cột thời gian (`timestamp_ms`/`_ns`) và hệ số quy đổi ra giây | |
| `split_segments(df)` | Cắt chuỗi tại các khoảng trống > `MAX_GAP_SEC`; chỉ giữ đoạn ≥ 1 cửa sổ (200 mẫu) | trả danh sách đoạn liên tục |
| `zscore_windows(X, eps)` | Chuẩn hóa Z-score **theo từng cửa sổ, từng kênh** (trừ trung bình, chia độ lệch chuẩn) | `eps`: tránh chia 0 |
| `make_windows(df, label)` | Trượt cửa sổ 200 mẫu/bước 20 trên từng đoạn → mảng `(N,200,9)` đã z-score | `label`: nhãn `<user>_<session>` gán cho mọi cửa sổ |
| `process_inertial(sess_dir, label)` | Sinh cửa sổ cho 2 cấu hình: **walking** (chỉ đi bộ) và **all** (đi+đứng+ngồi) | trả `(Xw,yw),(Xi,yi)` |
| `process_tap(sess_dir, session_id)` | Ghép cặp DOWN→UP thành sự kiện chạm, tính `hold_ms` (thời gian giữ) và `displacement` (độ dịch ngón) | lọc hold hợp lệ 0–500 ms |
| `_gesture_features(g)` | Trích **12 đặc trưng động học** của một cử chỉ cuộn: thời lượng, quãng đường, vận tốc trung bình/đỉnh/đầu/cuối, gia tốc đầu, độ thẳng, độ tập trung hướng (mrl)… | `g`: các điểm (t,x,y) của 1 cử chỉ |
| `process_scroll(sess_dir, session_id)` | Tách cử chỉ cuộn theo `pointer_id`, lọc thời lượng 10–5000 ms, sinh bảng đặc trưng | |
| `process_user(user_dir)` | Xử lý toàn bộ phiên của 1 người → lưu `X_walking.npy`, `X_inertial.npy`, `y_*.npy`, `tap_gestures.csv`, `scroll_gestures.csv` | |

## 1.3. `dataset_statistics.py` & `report_dataset_stats.py` — Thống kê dataset
- **`dataset_statistics.py`**: đọc `processed/` in bảng số cửa sổ IMU và số tap/scroll theo từng người và từng phiên, tổng kết toàn bộ (số window, số định danh). Sinh số liệu kiểm kê dữ liệu cho báo cáo.
- **`report_dataset_stats.py`**: sinh 2 bảng tái lập được cho mục "Lý do chọn mô hình":
  - `dataset_table(data_dir)`: kiểm kê window/session mỗi user.
  - `model_table()`: **số tham số, kích thước (MB), độ trễ CPU** của 3 kiến trúc (cnn/convlstm/convlstm_bi) — nguồn của **Bảng 4.1**. Latency đo bằng 20 lần forward, batch 10 cửa sổ.

## 1.4. `plot_charts.py` — Sinh toàn bộ biểu đồ minh họa (mục 3.4.1)
Gom mọi biểu đồ vào một script, **dùng chung hàm với `step2_preprocess`** (đúng ngưỡng, đúng logic phân đoạn). Cờ `SAVE`/`SHOW` cho phép chạy CLI (lưu file) hoặc notebook (xem inline).

| Hàm | Nhiệm vụ |
|-----|----------|
| `_load_processed(proc_dir)` | Đọc `.npy`, **chỉ lấy user dạng `user<số>`** (bỏ probe userA..E) để khớp số người trong báo cáo |
| `plot_phanbo(res, out)` | Biểu đồ phân bố số cửa sổ **theo người** và **theo cấu hình** (walking vs all); số nghìn dùng dấu chấm kiểu Việt Nam |
| `plot_tin_hieu_tho(...)` | Vẽ gia tốc kế thô khi **đi bộ (động)** vs **ngồi (tĩnh)** |
| `plot_zscore_users(...)` | Minh họa Z-score giúp **xóa khác biệt offset giữa người dùng** |
| `plot_scale_kenh(...)` | Boxplot 9 kênh trước/sau Z-score (đưa các kênh về cùng thang) |
| `plot_cua_so_truot(...)` | Minh họa **cửa sổ trượt** 200 mẫu, bước 20 mẫu (chồng lấn) |
| `plot_window_zscore(...)` | Một cửa sổ 4 giây trước/sau chuẩn hóa |
| `find_gaps(df)` / `report_gaps(...)` | Phát hiện điểm gián đoạn **giống hệt** `split_segments`, in cách tính số mẫu/thời lượng/số khoảng trống (giải trình con số trong báo cáo) |
| `plot_phan_doan(...)` | **Hình 3.4.1**: vẽ tín hiệu thật, tô đoạn giữ/loại/khoảng trống, phóng to một khoảng trống. Tham số `--pd_user user4 --pd_session 6 --pd_activity sitting` chỉ định nguồn vẽ |

*(`plot_charts.ipynb` là phiên bản notebook của file này để xem/lưu từng biểu đồ linh hoạt.)*

## 1.5. `export/export_tflite.py` & `export/export_tflite_cnn.py` — Xuất mô hình sang Android
Chuyển checkpoint PyTorch (`backbone.pt`) → mô hình **TFLite** nhúng vào app.
- **`export_tflite.py`** (hợp nhất): hỗ trợ cả `cnn`/`convlstm`/`convlstm_bi` (chọn ở biến `MODEL`). `export_tflite_cnn.py` là bản tiền nhiệm chỉ cho CNN.

| Bước | Nhiệm vụ |
|------|----------|
| Dựng mô hình Keras tương đương | `_conv_block_2layer(inp)` tạo 2 khối Conv1D dùng chung |
| `set_conv` / `set_bn` | **Chuyển trọng số** từ PyTorch sang Keras (chuyển vị tensor conv `(2,1,0)`, ghép bias LSTM `ih+hh`) |
| Smoke test | Kiểm tra đầu ra Keras vs TFLite lệch < `1e-4` (đảm bảo chuyển đổi không sai số) |
| Sao chép assets | Copy `backbone.tflite` + `impostor_pool_inertial.npy` + `touch_scaler.json` + cập nhật `export_manifest.json` vào `assets/<mode>/` |

---

# 2. `web_demo/` — Lõi xác thực & web demo (Python)

## 2.1. `models.py` — Kiến trúc mạng nơ-ron (3 backbone)
Định nghĩa 3 kiến trúc **cùng interface** (input `(batch, 9, T)` → embedding 128 chiều).

| Lớp/Hàm | Nhiệm vụ | Tham số quan trọng |
|---------|----------|--------------------|
| `BackboneCNN` | CNN 1D: 3 khối Conv (9→64→128→128) + AdaptiveAvgPool; `forward` trả `(logits, embedding)` | `n_users`: số lớp đầu ra; `dropout=0.4` |
| `_ConvLSTMEncoder` | 2 khối Conv giảm chuỗi + LSTM học phụ thuộc thời gian dài | `bidirectional`: BiLSTM thì hidden = embed_dim/2 mỗi chiều |
| `BackboneConvLSTM` | Bao `_ConvLSTMEncoder` + classifier | |
| `build_backbone(arch, n_users)` | Khởi tạo backbone theo tên `'cnn'/'convlstm'/'convlstm_bi'` | |
| `load_encoder(checkpoint_path, n_users, arch)` | Load checkpoint, **bỏ classifier head** chỉ giữ encoder; tự bỏ classifier nếu `n_users` khác lúc train | dùng cho suy luận embedding |

## 2.2. `verifier.py` — Lõi xác thực đa phương thức (file quan trọng nhất)
Triển khai **đúng đường quyết định on-device**: điểm quán tính `cos_znorm`, điểm touch (RF), và điểm fusion. `verify_session` trả **cả hai kết luận** (quán tính & fusion) để minh bạch so sánh. Chống rò rỉ: cohort/impostor loại trừ chính owner; enroll và test tách phiên.

**Hằng số quyết định:** `SCORE_SCALE=3.0`, `SCORE_BIAS=2.0` (tham số sigmoid sau z-norm), `FIXED_THRESHOLD=0.23` (ngưỡng dự phòng tại điểm EER), `THR_AGG_WINDOW=5` (gộp 5 cửa sổ trước khi đặt ngưỡng — khớp EWMA lúc chạy thật).

| Hàm | Nhiệm vụ | Tham số quan trọng |
|-----|----------|--------------------|
| `Artifacts` / `load_artifacts(export_dir)` | Nạp **impostor pool** quán tính (128-D), pool touch (33-D), scaler touch; tự tắt nhánh touch nếu lệch chiều | |
| `load_user_inertial(user_id, data_dir, mode)` | Đọc cửa sổ `.npy` của một người, tách **theo từng phiên** | `mode='walking'/'all'` chọn file nguồn |
| `extract_embeddings(encoder, windows)` | Z-score + đưa qua encoder → embedding 128-D | |
| `mean_cosine(e, a)` | Cosine trung bình giữa embedding và tập **anchor** | |
| `fit_cohort(anchors, pool)` | Tính `(mean, std)` điểm cosine mà anchor tạo ra trên cohort → tham số **z-norm** | gọi 1 lần lúc enroll |
| `score_inertial(embeds, anchors, cohort_mean, cohort_std)` | **Điểm quán tính**: mean cosine → z-norm cohort → sigmoid | đây là điểm dùng trên thiết bị |
| `_aggregate_scores(scores, w)` | Gộp điểm theo nhóm `w` cửa sổ rồi lấy trung bình (vì ở mức cửa sổ điểm genuine/impostor chồng lấn) | |
| `calibrate_owner_threshold(genuine, impostor)` | **Ngưỡng per-owner** cân bằng FAR≈FRR ở mức điểm đã gộp; thiếu impostor thì lùi `K_STD` độ lệch chuẩn dưới trung bình genuine | |
| `enroll(owner_id, n_enroll_sessions, ...)` | **Đăng ký chủ máy**: tách anchor/val, tính anchor + cohort z-norm, huấn luyện RF touch (nếu có), tune `fusion_w`, hiệu chuẩn ngưỡng per-owner | `n_enroll_sessions`: số phiên để enroll |
| `_build_inertial_pool` / `_build_touch_pool` | Dựng pool người lạ (loại owner), giới hạn kích thước (~100–nhiều) | `seed`: tái lập |
| `_tune(...)` | Grid-search **trọng số fusion** trên tập val theo AUC (ngưỡng vẫn cố định, độc lập số phiên enroll) | |
| `verify_session(enrollment, test_user, test_session, ...)` | Chấm 1 phiên test → trả `p_inertial`, `p_touch`, `p_fusion` và 2 kết luận TRUSTED/REJECTED | `fusion_w_override`: ép trọng số để khám phá |

## 2.3. `touch_subwindow.py` — Đặc trưng touch mức sub-window (33-D)
Dùng **đúng** sơ đồ đặc trưng mà pipeline đánh giá Chương 5 đã dùng, để web demo trình diễn nhánh touch trên đúng biểu diễn đã đánh giá.

**Cấu trúc 33 chiều:** 3 thống kê × 3 đại lượng tap (hold/displacement/inter-tap) = 9, + 1 (số tap), + 11 đặc trưng cuộn × 2 thống kê (mean/std) = 22, + 1 (số scroll) = **33**.

| Hàm | Nhiệm vụ | Tham số quan trọng |
|-----|----------|--------------------|
| `all_sessions(user_dir)` | Tập `session_id` trong `tap_gestures.csv` | |
| `_chunk_feats(tp_c, sc_c)` | Tính vector 33-D cho **một cụm** (20 tap + 10 scroll) | |
| `touch_subwindows(user_dir, sessions)` | Cắt cụm theo `TAP_CHUNK=20`/`SCR_CHUNK=10`, mỗi cụm cần ≥ `MIN_TAP_IN_CHUNK=5` tap → mảng `(n_subwindow, 33)` | |

## 2.4. `app.py` — Web demo (Streamlit)
Giao diện cho hội đồng tự so sánh: hiển thị **3 điểm** (quán tính/touch/fusion) + **2 kết luận**. Sidebar để chọn ngữ cảnh, chọn chủ máy, số phiên enroll và bấm **Đăng ký**; nút "Chấm điểm tất cả" chạy `verify_session` trên chủ máy + mọi người lạ, hiện bảng kết quả và độ chính xác. Nếu fusion không vượt quán tính, bảng sẽ cho thấy đúng vậy.

## 2.5. `validate_threshold.py` — Kiểm chứng ngưỡng per-owner
So sánh **FRR/FAR** giữa ngưỡng cố định (0,23) và ngưỡng per-owner trên cùng điểm `p_inertial`. `rate(decisions, accept_is_correct)` tính FRR (owner bị từ chối) hoặc FAR (impostor lọt). Bằng chứng cho việc ngưỡng per-owner ổn định owner mà vẫn an toàn.

## 2.6. `evaluate_variants.py` — So sánh 6 biến thể chọn mô hình nhúng
Đánh giá 6 tổ hợp {cnn, convlstm, convlstm_bi} × {walking, all} bằng **leave-users-out**, sinh `evaluation_results.json`.

| Hàm | Nhiệm vụ | Tham số quan trọng |
|-----|----------|--------------------|
| `find_eer(y_true, scores)` | Tính EER (điểm FPR≈FNR trên đường ROC) | |
| `cache_all_embeddings(encoder, users, mode, rng)` | Trích sẵn embedding 1 lần/user (tối ưu tốc độ), giới hạn `MAX_WIN_SESS=50` cửa sổ/phiên | |
| `evaluate_variant(variant, users)` | Với mỗi owner: dựng pool người lạ (`MAX_IMP_EMB=200`), train RF owner-vs-pool, đo **AUC/EER**, kích thước (MB), độ trễ (ms) | `N_ENROLL=4` phiên enroll |

---

# 3. `android_app/B_authenticator_app/` — App xác thực (Kotlin)

Luồng: `ModeSelect` → `OwnerEnrollment` (thu IMU + RF + cohort + ngưỡng) → `FallbackEnroll` (mật khẩu lắc) → `Quiz` (màn chính, dịch vụ chạy khi app hoạt động). Khi nghi ngờ → `Fallback`.

## 3.1. Gói `inference/` — Lõi suy luận

### `InferenceEngine.kt` — Bộ máy suy luận TFLite + chấm điểm
| Hàm | Nhiệm vụ | Tham số quan trọng |
|-----|----------|--------------------|
| `extractEmbedding(window)` | Chuẩn hóa cửa sổ rồi chạy TFLite → embedding 128-D (có chế độ **MOCK** khi thiếu mô hình) | `window`: SensorWindow 200×9 |
| `predict(window)` | **Điểm quán tính `cos_znorm`**: mean cosine tới anchor (lõi + thích nghi) → z-norm cohort → sigmoid | dùng `SCORE_SCALE=3`, `SCORE_BIAS=2` |
| `predictFused(window)` | Trả kết quả quán tính (bản triển khai đơn modal) | |
| `fitCohort(anchors, impostorPool)` (companion) | Tính `(mean,std)` cohort lúc enroll | trả `(0,1)` nếu pool rỗng |
| `scoreAgainstAnchors(embed, anchors, cohortMean, cohortStd)` | Chấm 1 embedding **giống hệt** `predict()` — dùng khi hiệu chuẩn ngưỡng | đảm bảo cùng thang điểm |
| `load(context, numThreads, mode)` (companion) | Nạp mô hình theo `assets/<mode>/`, scaler, OwnerProfile; kiểm tra shape (1,200,9)→(1,128) | `mode`: walking/all |

### `OwnerProfile.kt` — Lưu hồ sơ chủ máy (nhị phân, có version)
Lưu/đọc **anchor 128-D**, RF quán tính/touch, `fusionW`, **tham số cohort (mean,std)**, **ngưỡng per-owner**. Hỗ trợ 4 phiên bản định dạng (MAGIC_V1..V4) để tương thích ngược.
- `save(anchors, rfInertial, rfTouch, fusionW, cohortMean, cohortStd, thrInertial)`: ghi đầy đủ hồ sơ (V4).
- `getAnchors() / getCohortMean() / getThrInertial() / ...`: đọc có cache.
- `clear()`: xóa hồ sơ (dùng khi đổi mode/đăng ký lại).

### `ThresholdCalibrator.kt` — Hiệu chuẩn ngưỡng riêng từng owner
`calibrate(anchors, impostorPool, cohortMean, cohortStd, fallback)`: tính ngưỡng theo công thức `thr = impMedian + LENIENCY·(genMedian − impMedian)`, kẹp `[FLOOR=0.10, CEIL=0.45]`.
- `LENIENCY=0.30`: 0 = sát impostor (lỏng) … 1 = sát genuine (chặt).
- `genuine`: điểm leave-one-out giữa các anchor; `impostor`: điểm pool người lạ; gộp `AGG_WINDOW=5` bằng bootstrap để khớp EWMA.

### `ScoreAggregator.kt` — Làm mượt điểm theo thời gian (EWMA)
Gộp điểm các cửa sổ gần nhất bằng **trung bình mũ** (trọng số `alpha^k`) rồi suy ra trạng thái.
- `push(p)`: thêm điểm mới, trả điểm tổng hợp.
- `currentState()`: ánh xạ điểm → TRUSTED/WARNING/UNKNOWN theo `trustedThreshold`/`warningThreshold`.
- `windowSize=5`, `alpha=0.8`: số cửa sổ nhớ & độ ưu tiên điểm mới.

### `SensorWindowCollector.kt` — Thu một cửa sổ cảm biến
`collectOneWindow()` (suspend): bật acc/gyro/mag ở `SENSOR_DELAY_GAME` trong `WINDOW_SECONDS=4`, lấy **200 mẫu acc gần nhất**, **căn đồng bộ** gyro/mag về dấu thời gian từng mẫu acc bằng nội suy tuyến tính (`alignAt`). `RawChannelBuffer`: bộ đệm vòng cho từng kênh.

### `AdaptiveAnchorBuffer.kt` — Anchor thích nghi (cập nhật theo thời gian)
Thêm anchor mới khi người dùng **liên tục đạt điểm rất cao** để bám theo thay đổi hành vi nhẹ.
- `maybeAdd(embed, fusedScore)`: chỉ thêm khi `fusedScore ≥ adaptThreshold=0.92` đủ `requiredStreak=5` lần liên tiếp, tôn trọng `rateLimitMs` và `cooldownMs` (5 phút sau fallback).
- `onFallbackTriggered()`: kích cooldown (không học nhầm khi vừa nghi ngờ).
- Giới hạn `maxAdaptive=20`, lưu xuống đĩa.

### `RandomForestClassifier.kt` — Rừng ngẫu nhiên thuần Kotlin
Cài đặt RF chạy on-device (không cần thư viện ngoài), có thể serialize vào hồ sơ.
- `fit(X, y)`: dựng `nEstimators=200` cây, bootstrap mẫu, cân bằng lớp bằng trọng số, `maxFeatures=√(n)`.
- `predictProba(x)`: trung bình xác suất các cây.
- Lớp trong `DecisionTree`: chia nút theo **Gini** có trọng số, `MAX_DEPTH=20`, `minSamplesLeaf`.

### `FusionEngine.kt` — Dung hợp điểm quán tính + touch
- `fuse(pInertial, pTouch, w)`: `w·quán_tính + (1−w)·touch`.
- `tuneWeight(...)`: grid-search 51 mức `w` tối đa hóa AUC (tie-break về 0,5). `rocAuc(...)` tính AUC bằng quy tắc hình thang.

### `NpyReader.kt` — Đọc file `.npy` trên Android
`readFloat32_2D(context, assetPath)`: phân tích header `.npy`, chỉ nhận float32 little-endian 2 chiều (`<f4`), trả `Array<FloatArray>`. Dùng để nạp impostor pool từ assets.

## 3.2. Gói `model/` — Cấu trúc dữ liệu
| File | Vai trò |
|------|---------|
| `SensorWindow.kt` | Cửa sổ 200×9 dạng mảng phẳng; hằng chỉ số kênh `CH_ACC_X..CH_MAG_Z`; `toMatrix()` chuyển về ma trận |
| `ScalerParams.kt` | Tham số chuẩn hóa từ `scaler_params.json`: `normalize()` theo `per_window_zscore` (mặc định) hoặc `fitted_standard_scaler`; `isLikelyDriftedFromTraining()` cảnh báo lệch phân bố |
| `AuthState.kt` | Enum 3 trạng thái TRUSTED/WARNING/UNKNOWN; `fromScore(score, trusted, warning)` ánh xạ điểm → trạng thái |
| `ExportManifest.kt` | Đọc `export_manifest.json`: ngưỡng quyết định, tham số aggregator (alpha, window), phiên bản pipeline |

## 3.3. Gói `service/` — `AuthenticationService.kt` — Dịch vụ xác thực liên tục
Foreground service chấm điểm theo cửa sổ và quản lý máy trạng thái 3 mức.
| Hàm | Nhiệm vụ | Tham số quan trọng |
|-----|----------|--------------------|
| `onCreate()` | Nạp engine + aggregator (ưu tiên **ngưỡng per-owner** nếu đã hiệu chuẩn), đăng ký kích hoạt thu (màn hình bật, significant motion, lấy mẫu định kỳ) | `PERIODIC_CAPTURE_INTERVAL_MS=4000` |
| `runCaptureLoop()` | Vòng lặp: thu cửa sổ → **cổng phát hiện cử động** → chấm điểm → cập nhật trạng thái → thông báo; vào WARNING/UNKNOWN thì hẹn giờ đệm rồi mới fallback | `COOLDOWN_MS=5000`, `WARNING_TIMEOUT_MS=15000`, `UNKNOWN_GRACE_MS=6000` |
| `hasEnoughMotion(window)` | Tính phương sai độ lớn gia tốc **thô**; máy bất động → bỏ qua (tránh chấm điểm trên nhiễu) | `MOTION_VAR_THRESHOLD=0.15` |
| `launchFallbackActivity()` | Mở màn fallback khi nguy hiểm kéo dài; tôn trọng `FALLBACK_GRACE_MS=30000` | |
| `onFallbackVerified()` / `onFallbackMaxFailed()` | Đặt lại điểm khi xác thực lại đúng / khóa fallback khi sai quá số lần | |

## 3.4. Gói `ui/` — Các màn hình
| File | Vai trò | Hàm/tham số đáng chú ý |
|------|---------|------------------------|
| `ModeSelectActivity.kt` | Chọn ngữ cảnh mô hình (Đi bộ/Toàn bộ). Đổi mode sau khi enroll sẽ xóa hồ sơ và enroll lại | `wireCard()` vô hiệu mode thiếu assets |
| `OwnerEnrollmentActivity.kt` | **Đăng ký chủ máy**: thu `ANCHOR_COUNT=20` cửa sổ → embedding làm anchor, train RF (`NEG_POOL_RATIO=4` lần số anchor), tính cohort + ngưỡng per-owner, lưu hồ sơ | `trainRfInertial(anchors)` |
| `FallbackEnrollActivity.kt` | Đăng ký "mật khẩu lắc" (nhập 2 lần phải khớp) | dùng `ShakeDetector` + `PatternStorage` |
| `QuizActivity.kt` | Màn chính: khởi động dịch vụ, hiển thị trạng thái/điểm tin cậy thời gian thực; quiz để người dùng tương tác (sinh chuyển động cho cảm biến) | `updateAuthStatus()` đọc state mỗi giây |

## 3.5. Gói `fallback/` — Cơ chế dự phòng "mật khẩu lắc"
| File | Vai trò | Tham số quan trọng |
|------|---------|--------------------|
| `ShakeDetector.kt` | Phát hiện lắc và **tách thành dãy chữ số**: mỗi đỉnh gia tốc vượt ngưỡng là 1 lần lắc, một cụm dừng > `DIGIT_GAP_MS=1200` chốt thành 1 chữ số (1–9) | `PEAK_THRESHOLD_MPS2=6.0`, `DEBOUNCE_MS=200` |
| `PatternStorage.kt` | Lưu mật khẩu lắc an toàn: **không lưu dãy thô**, lưu salt + SHA-256(salt‖dãy) trong EncryptedSharedPreferences; khóa sau `MAX_FAILED_ATTEMPTS=3` | `validate()` ép dãy 4–8 chữ số |
| `FallbackActivity.kt` | Màn nhập lại dãy lắc khi nghi ngờ; đúng → `onFallbackVerified`, sai quá 3 lần → `onFallbackMaxFailed` (yêu cầu PIN hệ thống) | |

## 3.6. Gói `util/` + `AuthenticatorApp.kt`
| File | Vai trò |
|------|---------|
| `ContextMode.kt` | Enum WALKING/ALL; lưu/đọc lựa chọn vào SharedPreferences; `assetPath(mode, file)` ghép đường dẫn `assets/<mode>/...`; `isAvailable()` kiểm tra có `backbone.tflite` |
| `BootReceiver.kt` | Tự khởi động dịch vụ sau khi bật máy **nếu** đã enroll đầy đủ và bật `auto_start_on_boot` |
| `NotificationHelper.kt` | Tạo kênh thông báo cho foreground service |
| `AuthenticatorApp.kt` | Application: khởi tạo kênh thông báo lúc app start |

---

# 4. `data_collectiom/DataCollectV2/` — App thu thập dữ liệu (Kotlin)

Luồng: `Main` (splash) → `Registration` (đăng ký + đồng ý) → `SensorCollection` (thu IMU 3 hoạt động) → `Form` (thu touch qua khảo sát) → `Upload` (nén ZIP + chia sẻ).

| File | Vai trò | Hàm/tham số đáng chú ý |
|------|---------|------------------------|
| `MainActivity.kt` | Màn splash, điều hướng theo trạng thái đăng nhập | `SPLASH_DELAY_MS=1400` |
| `RegistrationActivity.kt` | Form đăng ký (tên, tuổi, giới tính, tay thuận, thiết bị, **đồng ý tham gia**); sinh `userId` và ghi `metadata.csv` | `validateForm()`, `saveMetadata(userId)` |
| `UserSession.kt` | Lưu/đọc hồ sơ người dùng trong SharedPreferences | `saveAndLogin`, `getProfile`, `logout` |
| `SensorForegroundService.kt` | Foreground service đếm giờ thu, cập nhật thông báo (đang thu/đã dừng), giữ thời lượng tích lũy | `startRecording(session)`, `stopRecording()`, `currentElapsedMs()`, callback `onTick` |
| `RecordingSession.kt` | Trạng thái 1 lần thu (nhãn hoạt động + thời lượng tích lũy); `totalElapsed(anchorMs)` cộng thời gian đang chạy | |
| `SensorCollectionActivity.kt` | **Thu IMU**: đăng ký acc/gyro/mag (`SENSOR_DELAY_GAME`), gom buffer, vẽ chart, **lưu CSV** đúng định dạng `timestamp_ms,acc_*,gyro_*,mag_*,activity,session_id`; quản lý chỉ tiêu/tiến độ mỗi hoạt động | `MIN_SAMPLES=150`; `saveToCSV(size, attempt)`; `DEFAULT_TARGET_SECS=360s`/hoạt động |
| `FormActivity.kt` | **Thu touch**: dựng form trắc nghiệm + chọn nhiều; bắt sự kiện chạm (`TapEvent` DOWN/UP + `hold_ms`) và cuộn (`ScrollEvent` DOWN/MOVE/UP đa chạm); lưu `tap_r<round>.csv`, `scroll_r<round>.csv`; `TOTAL_ROUNDS=2` để tăng gấp đôi mẫu | `setupScrollTracking()`, `saveFormData()` |
| `data/TapEvent.kt` | Bản ghi 1 sự kiện chạm: thời gian, x, y, áp lực, kích thước, pha, `hold_ms` | |
| `data/ScrollEvent.kt` | Bản ghi 1 điểm cuộn: thời gian, x, y, áp lực, kích thước, pha, `pointer_id` | |
| `UploadActivity.kt` | Thống kê dữ liệu đã thu, **nén toàn bộ phiên thành ZIP** và chia sẻ; sau khi tạo ZIP thì xóa dữ liệu gốc; `incrementSessionNumber()` để thu phiên tiếp | `createZip()`, `shareZip()` |
| `PermissionHelper.kt` | Xin quyền POST_NOTIFICATIONS + ACTIVITY_RECOGNITION kèm giải thích | `checkAndRequest(onAllGranted)` |
| `view/SensorBarChartView.kt` | View tùy biến vẽ biểu đồ cột độ lớn gia tốc thời gian thực (40 cột gần nhất) | `push(value)` |

*(`ExampleUnitTest.kt`, `ExampleInstrumentedTest.kt` là test mẫu mặc định của Android Studio, không thuộc logic nghiệp vụ.)*
