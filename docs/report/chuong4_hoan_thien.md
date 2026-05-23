# CHƯƠNG 4. XÂY DỰNG DATASET VÀ TIỀN XỬ LÝ DỮ LIỆU

## 4.1. Ứng dụng thu thập BioAuth Data Collection

### 4.1.1. Tổng quan kiến trúc ứng dụng

Ứng dụng BioAuth Data Collection được phát triển bằng Kotlin cho nền tảng Android (`minSdk = 24` tương ứng Android 7.0 Nougat, `targetSdk = 36`), với ViewBinding và Material Design 3 nhằm đảm bảo trải nghiệm thống nhất giữa nhiều dòng máy. Toàn bộ source code tập trung trong package `com.datn.datacollectv2`, với khoảng 2.500 dòng Kotlin chia theo trách nhiệm rõ ràng.

Theo phương pháp mô tả kiến trúc của B2auth [2], em trình bày ứng dụng theo bốn **phase tuần tự** kết hợp với một **module Background Coordination** chạy xuyên suốt, tương ứng với toàn bộ luồng dữ liệu từ khi người dùng đăng ký tới khi xuất file ZIP. Mỗi phase nhận đầu vào, thực hiện một nhiệm vụ thu hoặc xử lý, và sinh ra đầu ra ở dạng file CSV được lưu vào hệ thống thư mục có cấu trúc.

![Kiến trúc tổng quan ứng dụng BioAuth Data Collection — 4 phase tuần tự + Background Coordination](architecture_overview.png)

*Hình 4.1. Kiến trúc tổng quan ứng dụng BioAuth Data Collection. Bốn phase chính (Enrollment, Inertial Sensor Capture, Behavioral Interaction Capture, Data Export) chạy tuần tự với mũi tên đậm, có cơ chế quay vòng (mũi tên đứt ở Phase 4) khi người dùng nhấn "Thu thêm dữ liệu" để tăng session. Module Background Coordination (góc phải) phối hợp giữa các phase qua Foreground Service, timer tick và session manager.*

**Phase 1 — User Enrollment** thu metadata người tham gia (họ tên, tuổi, giới tính, tay thuận, mã thiết bị) qua một form có validation, sinh ra `userId = user_<timestamp>` duy nhất rồi ghi `metadata.csv` vào thư mục `getExternalFilesDir(null)/<userId>/`. Phase này được hiện thực bởi `RegistrationActivity` cùng lớp `UserSession` (Kotlin `object` Singleton lưu hồ sơ vào `SharedPreferences` để các phase sau truy cập).

**Phase 2 — Inertial Sensor Capture** thu cảm biến chuyển động cho ba hoạt động (đi bộ, đứng, ngồi). Module này đăng ký `SensorManager` lắng nghe ba cảm biến (accelerometer, gyroscope, magnetometer) ở chế độ `SENSOR_DELAY_GAME` (~50 Hz), ghi liên tục cho đến khi người dùng nhấn nút Dừng. Một **Quality Gate** kiểm tra ngưỡng `MIN_SAMPLES = 150` (tương đương 3 giây ở 50 Hz) — nếu phiên thu ngắn hơn, dữ liệu bị từ chối; nếu đạt ngưỡng, sinh ra file `<activity>_att<N>.csv` 12 cột (`timestamp_ms, acc_x/y/z, gyro_x/y/z, mag_x/y/z, activity, session_id`). Pha này được hiện thực bởi `SensorCollectionActivity` với cờ `keepScreenOn=true` để màn hình không tắt trong suốt phiên thu.

**Phase 3 — Behavioral Interaction Capture** thu đồng thời ba kênh dữ liệu hành vi qua một bảng câu hỏi gồm 14 câu trắc nghiệm + 7 câu trả lời tự do + 5 câu chọn nhiều có cuộn, được điền hai lần liên tiếp (2-Round Form) để gấp đôi số mẫu. Tap dynamics được capture qua `setOnTouchListener` trên các card lựa chọn (lưu `x, y, pressure, size, hold_ms`); keystroke dynamics được capture qua `TextWatcher` trên các trường nhập liệu (lưu `inter_key_ms, is_delete`); scroll dynamics được capture qua `setOnTouchListener` trên `NestedScrollView` ngoài cùng (lưu raw `MotionEvent` với `pointer_id` cho mỗi chạm). Đầu ra là ba file CSV: `tap_rN.csv`, `keystroke_rN.csv`, `scroll_rN.csv`. Phase này được hiện thực bởi `FormActivity`.

**Phase 4 — Data Export** thực hiện ba nhiệm vụ: hiển thị thống kê tổng hợp (số file, số dòng, dung lượng), nén toàn bộ thư mục `<userId>/` thành ZIP qua `ZipOutputStream` (chạy trong `Thread` riêng tránh chặn UI), và chia sẻ ZIP qua `Intent.ACTION_SEND` với `FileProvider`. Sau khi tạo ZIP thành công, hệ thống xóa toàn bộ thư mục con cũ (giữ `metadata.csv`) để tránh trùng lặp dữ liệu giữa các session, đồng thời cho phép tăng session qua nút "Thu thêm dữ liệu". Phase này được hiện thực bởi `UploadActivity`.

**Background Coordination** — chạy xuyên suốt Phase 2 và 3 — gồm ba thành phần phụ trợ. `SensorForegroundService` được khai báo với `foregroundServiceType="dataSync"` trong manifest, hiển thị notification ổn định trên thanh trạng thái để giữ process priority và ngăn OS suspend; đồng thời chạy timer 100 ms qua `Handler.postDelayed` để cập nhật notification và phát callback `onTick` về Activity nhằm vẽ đồng hồ đếm thời gian real-time. `Session Manager` (lưu trong `SharedPreferences` với key `current_session_id`) quản lý vòng đời session từ `session_1` đến `session_6` — cơ chế tăng session khi người dùng nhấn "Thu thêm dữ liệu" trong `UploadActivity` chính là yếu tố tạo ra cấu trúc 6 phiên/người trong dataset cuối cùng.

Để chi tiết hơn về quan hệ kế thừa và composition giữa các class Kotlin hiện thực kiến trúc trên, Hình 4.2 trình bày class diagram đầy đủ. Hình này bổ sung cho Hình 4.1 ở góc độ implementation: Activity nào kế thừa `AppCompatActivity`, Service nào triển khai pattern `LocalBinder`, các data class nào lưu trữ event, và quan hệ phụ thuộc giữa chúng.

![Class diagram chi tiết của ứng dụng BioAuth Data Collection](class_diagram.png)

*Hình 4.2. Class diagram chi tiết — 5 Activity kế thừa `AppCompatActivity`, một `Service` triển khai pattern `LocalBinder`, các Singleton (`UserSession`) và data class (`RecordingSession`, `TapEvent`, `KeystrokeEvent`, `ScrollEvent`) hỗ trợ.*

Luồng người dùng tương ứng với bốn phase ở Hình 4.1 đi qua năm Activity tuần tự: (1) `MainActivity` hiển thị splash 1,4 giây với hiệu ứng fade-in tuần tự rồi điều hướng dựa trên trạng thái đăng nhập trong `UserSession`; (2) nếu chưa đăng nhập, `RegistrationActivity` thực hiện Phase 1; (3) `SensorCollectionActivity` thực hiện Phase 2; (4) `FormActivity` thực hiện Phase 3; (5) `UploadActivity` thực hiện Phase 4. `SensorForegroundService` chạy song song với `SensorCollectionActivity`, được bind theo mô hình `LocalBinder` để cung cấp API điều khiển và callback timer ngược về Activity mỗi 100 ms.

### 4.1.2. Cơ chế thu cảm biến liên tục

`SensorCollectionActivity` triển khai interface `SensorEventListener` và là thành phần kỹ thuật trọng tâm của giai đoạn thu inertial. Sau khi user chạm vào card hoạt động, Activity gọi `sensorManager.registerListener()` đăng ký lắng nghe ba cảm biến (`TYPE_ACCELEROMETER`, `TYPE_GYROSCOPE`, `TYPE_MAGNETIC_FIELD`) đồng thời ở chế độ `SENSOR_DELAY_GAME` — tần số tham chiếu ~50 Hz. Mỗi `SensorEvent` đến qua callback `onSensorChanged()` được clone (`event.values.clone()`) và đẩy vào ba `MutableList<FloatArray>` riêng biệt cho ba cảm biến; riêng buffer accelerometer có thêm `accTimestampBuffer` ghi `event.timestamp` (giá trị nano giây từ `elapsedRealtimeNanos()`).

Để bảo vệ tiến trình thu khỏi bị Android suspend khi ứng dụng vào nền, `SensorCollectionActivity` đồng thời (i) gọi `startForegroundService()` với `SensorForegroundService` đăng ký `foregroundServiceType="dataSync"` trong manifest, hiển thị notification ổn định trên thanh trạng thái — đây là cơ chế chính đảm bảo OS không kill process; và (ii) gắn cờ `keepScreenOn=true` ở Activity để màn hình không tắt trong phiên thu. Activity sau đó bind vào Service qua `LocalBinder` pattern để gọi `startRecording(RecordingSession)` và đăng ký callback `onTick: ((Long) -> Unit)`. Service không thực hiện đăng ký cảm biến — toàn bộ logic `SensorEventListener` đặt ở Activity — Service chỉ giữ vai trò: giữ process priority bằng Foreground Service, chạy timer định kỳ 100 ms qua `Handler.postDelayed` để cập nhật notification và phát hiện callback `onTick` về Activity nhằm vẽ đồng hồ đếm thời gian, và quản lý vòng đời `RecordingSession` (anchor time + accumulated time để hỗ trợ pause/resume sau này nếu cần).

Đặc điểm so sánh với thiết kế cũ (chỉ thu 150 mẫu cố định mỗi lần nhấn) được trình bày trong Bảng 4.1.

| **Tiêu chí** | **Cũ (cố định 150 mẫu)** | **Mới (thu liên tục + nút Dừng)** |
|---|---|---|
| Cơ chế dừng | Tự dừng sau 150 mẫu (~3 s) | Người dùng chủ động nhấn nút Dừng |
| Mục tiêu thời gian | Cố định ~3 s | Mặc định 360 s/hoạt động, chỉnh được 30 s – 1800 s qua UI |
| Windows / lần thu | 1 window duy nhất | Hàng trăm windows (tuỳ thời lượng) |
| Windows / người (11 người) | ~90 windows tổng — không đủ huấn luyện | ~6.770 windows trung bình — đủ huấn luyện |
| Kiểm soát chất lượng | Không có | Từ chối nếu thu được < `MIN_SAMPLES = 150` mẫu (~3 giây ở 50 Hz) |
| Phản hồi trực quan | Không có | Đồ thị cột cuộn real-time hiển thị \|\|acc\|\| qua `SensorBarChartView` |
| Bảo vệ chống suspend | Không có | `SensorForegroundService` (`dataSync`) + `keepScreenOn=true` |

*Bảng 4.1. So sánh cơ chế thu cũ và mới.*

Khi người dùng nhấn nút Dừng, hàm `stopAndSave()` kiểm tra `minSize = min(accBuffer, gyroBuffer, magBuffer)`. Nếu `minSize < MIN_SAMPLES`, toàn bộ buffer bị xóa và hiển thị thông báo "Quá ngắn"; nếu đạt ngưỡng, dữ liệu được ghi vào `<userId>/<sessionId>/<activity>_att<N>.csv` với 12 cột: `timestamp_ms, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z, activity, session_id`. Số `att<N>` tăng dần để cho phép thu nhiều lượt cho cùng một hoạt động trong cùng một session — ví dụ `walking_att1.csv`, `walking_att2.csv`. Cơ chế attempt counter này giải thích biểu đồ G ở phần phân tích pre-preprocessing: một số người dùng có nhiều file cho cùng một (session, activity) do nhấn dừng và thu lại nhiều lần.

![Sequence diagram thu cảm biến qua SensorCollectionActivity và SensorForegroundService](seq_sensor.png)

*Hình 4.3. Sequence diagram luồng thu cảm biến — bind Service, đăng ký SensorManager, thu liên tục, kiểm tra ngưỡng và ghi CSV.*

### 4.1.3. Thu thập dữ liệu tương tác màn hình

`FormActivity` thu thập đồng thời ba loại sự kiện hành vi — tap, keystroke và scroll — thông qua một bảng câu hỏi gồm ba phần: 14 câu trắc nghiệm bốn lựa chọn, 7 câu trả lời tự do (multi-line text), và 5 câu chọn nhiều có cuộn (mỗi câu liệt kê 12 mục). Toàn bộ câu hỏi được điền hai lần liên tiếp (`currentRound = 1` rồi 2) để gấp đôi số mẫu thu được; sau khi hoàn tất lần một, dialog xác nhận hiển thị nhắc người dùng tiếp tục lần hai trước khi sang `UploadActivity`. Kiểm tra hoàn thiện được thực hiện trước khi cho phép submit: tất cả câu trắc nghiệm phải có đáp án và tất cả câu text phải không trống; nếu thiếu, toast hiển thị câu hỏi đầu tiên chưa hoàn thành.

Ba loại sự kiện được capture qua ba cơ chế kỹ thuật riêng biệt. Tap dynamics được thu bằng `setOnTouchListener` gắn lên từng `MaterialCardView` của lựa chọn trắc nghiệm: khi `ACTION_DOWN` xảy ra, ứng dụng lưu `event.eventTime` thô làm `downEventTime` và đẩy một `TapEvent(phase="DOWN")` với tọa độ chạm `(x, y)`, áp lực `pressure`, diện tích tiếp xúc `size` vào `tapEvents`; khi `ACTION_UP` xảy ra, ứng dụng tính `hold_ms = eventTime - downEventTime` (raw — raw, đảm bảo không cộng hai lần UTC offset gây sai số) và đẩy thêm một `TapEvent(phase="UP", hold_ms=...)`. Keystroke dynamics được thu bằng `TextWatcher.onTextChanged()` gắn vào từng `EditText`: mỗi lần text thay đổi, ứng dụng tính `inter_key_ms = uptimeMillis() - lastKeyUptime` và lập cờ `is_delete = (count == 0)` — tham số `count` của callback chính là số ký tự được thêm, bằng 0 nghĩa là người dùng vừa xóa thay vì gõ. Scroll dynamics được thu bằng `setOnTouchListener` gắn lên `NestedScrollView` ngoài cùng: ứng dụng lưu raw `MotionEvent` với `phase ∈ {DOWN, MOVE, UP}` cho mỗi pointer (hỗ trợ multi-touch qua `pointer_id`). Ứng dụng KHÔNG tính các đặc trưng phái sinh (vận tốc, hướng, khoảng cách) ở phía client — toàn bộ feature engineering được thực hiện ở pipeline Python (Mục 4.5) để giữ raw data đầy đủ và tránh chôn chặt thuật toán trích xuất vào app.

![Sequence diagram thu touch trong FormActivity](seq_form.png)

*Hình 4.4. Sequence diagram luồng thu touch — ba kênh capture song song (tap, keystroke, scroll) qua ba listener Android khác nhau.*

Sau khi submit lần hai (hoặc submit lần một nếu người dùng chọn "Hoàn tất" sớm), `saveFormData()` ghi ba file CSV vào cùng thư mục với inertial: `tap_r<round>.csv` (cột `timestamp_ms, x, y, pressure, size, phase, hold_ms, session_id`), `keystroke_r<round>.csv` (cột `timestamp_ms, field_id, char_count, inter_key_ms, is_delete, session_id`) và `scroll_r<round>.csv` (cột `timestamp_ms, x, y, pressure, size, phase, pointer_id, session_id`). Cấu trúc 8–9 cột ngắn gọn giúp file kích thước nhỏ (~50–200 KB mỗi file) và parse dễ trong giai đoạn extract feature.

### 4.1.4. Lưu trữ và xuất dữ liệu

`UploadActivity` thực hiện ba nhiệm vụ. Thứ nhất, hiển thị thống kê tổng hợp: số file inertial CSV (kiểm tra qua tiền tố `walking_att`, `standing_att`, `sitting_att`), số dòng tap/keystroke/scroll (đếm `bufferedReader().lineSequence().count() - 1` để bỏ header), tổng dung lượng tính bằng KB/MB. Nếu phát hiện ít hơn 3 file inertial, hiển thị cảnh báo nhắc người dùng kiểm tra lại trước khi xuất. Thứ hai, nén toàn bộ thư mục `<userId>/` thành ZIP theo cấu trúc cây gốc qua `ZipOutputStream` (chạy trong `Thread` riêng để không chặn UI), lưu vào `cacheDir/export/<userId>_<timestamp>.zip`. Thứ ba, chia sẻ ZIP qua `Intent.ACTION_SEND` với `FileProvider` cấp `FLAG_GRANT_READ_URI_PERMISSION` để các ứng dụng nhận (Drive, Email, Telegram...) có thể đọc file.

Một thiết kế đặc biệt: ngay sau khi tạo ZIP thành công, hàm `deleteAllSessionData()` xóa toàn bộ thư mục con (giữ lại `metadata.csv` ở root) trước khi mở dialog chia sẻ. Cách làm này có hai lý do: (i) tránh trùng lặp dữ liệu giữa các session (nếu user thu session_2 mà session_1 chưa xóa, ZIP lần sau sẽ chứa cả hai); (ii) buộc người dùng chia sẻ ngay (không thể "tạo ZIP rồi sau đó tạo ZIP lần nữa từ cùng dữ liệu"). Dialog xác nhận trước khi tạo ZIP cảnh báo rõ ràng điều này. Nút "Hoàn tất → Thu thêm dữ liệu" gọi `incrementSessionNumber()` tăng `session_<N>` lên `session_<N+1>` qua `SharedPreferences` rồi quay về `SensorCollectionActivity`, reset toàn bộ progress của 3 hoạt động về 0 và bắt đầu chu trình mới.

### 4.1.5. Đồng nhất timestamp giữa các nguồn

Một vấn đề kỹ thuật quan trọng là đồng nhất timestamp giữa inertial và touch để có thể fusion ở cấp độ session. Android cung cấp ba nguồn thời gian khác nhau trong cùng một quá trình: `System.currentTimeMillis()` (UTC wall clock, có thể nhảy nếu user chỉnh đồng hồ), `SystemClock.elapsedRealtimeNanos()` (đếm liên tục kể cả khi máy sleep, dùng cho `event.timestamp` của `SensorEvent`) và `SystemClock.uptimeMillis()` (đếm chỉ khi máy thức, dùng cho `event.eventTime` của `MotionEvent` và `TextWatcher`). Để chuyển tất cả về UTC chung mà không bị ảnh hưởng nếu user chỉnh đồng hồ giữa chừng, ứng dụng tính UTC offset đúng một lần khi bắt đầu phiên và cộng cố định:

- Trong `SensorCollectionActivity.startRecording()`:
  ```
  recordingUtcOffsetMs = System.currentTimeMillis() - elapsedRealtimeNanos() / 1_000_000
  // Mỗi mẫu: timestamp_ms = recordingUtcOffsetMs + event.timestamp / 1_000_000
  ```

- Trong `FormActivity` (lazy init):
  ```
  utcOffsetMs = System.currentTimeMillis() - SystemClock.uptimeMillis()
  // Mỗi event: timestamp_ms = utcOffsetMs + event.eventTime
  ```

Cần lưu ý hai nguồn thời gian dùng cho inertial (`elapsedRealtime`) và touch (`uptimeMillis`) khác nhau ở chỗ `elapsedRealtime` đếm cả thời gian máy sleep, trong khi `uptimeMillis` thì không. Trên thực tế, do toàn bộ phiên thu của một hoạt động kéo dài dưới 30 phút và Activity giữ `keepScreenOn=true` nên màn hình luôn ON, máy không vào deep sleep — chênh lệch giữa hai đồng hồ là không đáng kể (dưới 100 ms trong điều kiện thử nghiệm). Tuy nhiên đây là một điểm cần cải thiện cho các phiên bản tiếp theo khi muốn thu trong nền lâu hơn — lúc đó nên dùng cùng một nguồn `elapsedRealtimeNanos` cho cả hai.

## 4.2. Quy trình thu thập thực tế

### 4.2.1. Kịch bản thu thập

Mỗi người tham gia hoàn thành tổng cộng sáu phiên thu (`session_1` đến `session_6`) theo cùng một kịch bản chuẩn hóa. Trong mỗi phiên, ứng dụng được khởi động lại để tạo `sessionId` mới, sau đó người tham gia thực hiện ba hoạt động (đi bộ, đứng, ngồi) với thời lượng 4–5 phút mỗi hoạt động, kết thúc bằng việc điền form câu hỏi để thu tap, scroll và keystroke. Việc lặp sáu phiên trong điều kiện khác nhau (tay thuận / tay không thuận, các thời điểm trong ngày) giúp tăng tính biến thiên tự nhiên của dữ liệu — một biến số mà IntelliAuth [1] đã chỉ ra có tác động đáng kể đến độ chính xác của nhận diện dựa trên inertial sensors.

| **Phần** | **Hoạt động** | **Thời gian / phiên** | **Số phiên** | **Windows trung bình / người** |
|---|---|---|---|---|
| A — Inertial | Đi bộ (~50 m) | ~4 phút | 6 | ~2.241 |
|  | Đứng dùng điện thoại | ~4 phút | 6 | ~2.237 |
|  | Ngồi dùng điện thoại | ~4 phút | 6 | ~2.293 |
| B — Touch / Keystroke | Điền form chuẩn hóa | ~5 phút | 6 | ≥720 mỗi loại sự kiện |

*Bảng 4.2. Cấu trúc thu thập (1 người tham gia, 6 phiên độc lập).*

### 4.2.2. Thành phần người tham gia

Nghiên cứu tuyển 11 người tham gia với tiêu chí đa dạng về giới tính và độ tuổi (18–35 tuổi), sở hữu thiết bị Android cá nhân. Để đảm bảo tính riêng tư, danh tính thực của người tham gia được mã hóa thành userA, userB, userC, userD, userE, userF, userG, userH, userI, userK, userL — bỏ qua ký tự J để tránh nhầm lẫn với chữ I. Mỗi người tham gia ký vào phiếu đồng ý cho phép sử dụng dữ liệu cảm biến và tương tác màn hình thu được trong phạm vi nghiên cứu của đề tài.

## 4.3. Kết quả thu thập thực tế

### 4.3.1. Thống kê dữ liệu inertial

Sau khi hoàn thành thu thập và tiền xử lý, tổng cộng **74.490 cửa sổ inertial** đã được tạo ra từ 11 người dùng × 6 phiên. Tất cả người dùng đều vượt xa ngưỡng tối thiểu 4.500 mẫu mỗi người, với tổng số mẫu raw dao động từ ~325.000 đến ~370.000 mẫu/người. Phân phối giữa ba lớp hoạt động rất cân bằng — chênh lệch tối đa giữa lớp ít nhất và nhiều nhất chỉ 2,5%, không cần can thiệp cân bằng mạnh khi huấn luyện.

| **Chỉ số** | **Giá trị thu được** | **Mục tiêu** | **Đánh giá** |
|---|---|---|---|
| Số người tham gia | 11 người | 10–15 người | **Đạt** |
| Tổng windows — Đi bộ | 24.657 (33,1%) | ≥10.000 | **Đạt** |
| Tổng windows — Đứng | 24.606 (33,0%) | ≥10.000 | **Đạt** |
| Tổng windows — Ngồi | 25.227 (33,9%) | ≥10.000 | **Đạt** |
| Tổng windows inertial | **74.490** | ≥30.000 | **Đạt** |
| Windows tối thiểu / người | 6.524 (userL) | ≥4.500 | **Đạt** |
| Windows tối đa / người | 7.347 (userG) | — | — |
| Mức lệch Hz ngoài ±15% | 0,1% (Đi bộ); 0,1% (Đứng); 0,3% (Ngồi) | <5% | **Vượt mục tiêu** |
| Đoạn giữ lại sau lọc gap | 99,1% (Đi bộ); 98,6% (Đứng); 96,2% (Ngồi) | ≥60% | **Vượt mục tiêu** |

*Bảng 4.3. Thống kê dataset inertial thực tế.*

### 4.3.2. Thống kê dữ liệu touch và keystroke

Dữ liệu tương tác màn hình rất cân bằng và đầy đủ cho tất cả 11 người dùng. Tổng cộng thu được 8.159 sự kiện tap, 8.702 gestures cuộn và 7.900 sự kiện keystroke. Tất cả người dùng đều vượt ngưỡng tối thiểu 600 sự kiện mỗi loại — ngưỡng được thiết lập để đảm bảo profile có đủ độ tin cậy cho phương pháp cosine similarity scoring (xem mục 4.5.4).

| **Loại sự kiện** | **Tổng** | **Khoảng / người** | **Trung bình / người** | **Trạng thái** |
|---|---|---|---|---|
| Tap (lần nhấn) | 8.159 | 675–787 | ~742 | **Tất cả đạt** |
| Scroll (gestures) | 8.702 | 728–889 | ~791 | **Tất cả đạt** |
| Keystroke (intervals) | 7.900 | 667–806 | ~718 | **Tất cả đạt** |

*Bảng 4.4. Thống kê sự kiện cảm ứng theo người dùng.*

## 4.4. Pipeline tiền xử lý (step2_preprocess.py)

Sau khi thu thập, toàn bộ file ZIP từ 11 người tham gia được giải nén và đưa vào pipeline tiền xử lý tự động thực thi qua script `step2_preprocess.py`. Quá trình gồm sáu bước tuần tự được thiết kế để đảm bảo chất lượng dữ liệu đầu vào cho mô hình học máy. Để hỗ trợ chẩn đoán chất lượng dữ liệu thô trước khi đưa vào pipeline, một script bổ trợ `step2b_raw_diagnosis.py` sinh ra bảy biểu đồ chẩn đoán (A–G) cho phép kiểm tra tần số, gap, spike, phân bố kênh, độ dài đoạn và độ phủ của dữ liệu. Các quyết định kỹ thuật trong từng bước dưới đây đều được biện minh bằng kết quả phân tích từ các biểu đồ này.

### 4.4.1. Bước 1 — Kiểm tra và phát hiện tần số lệch

Android `SensorManager` không đảm bảo tần số lấy mẫu đều đặn tuyệt đối, đặc biệt khi CPU bận xử lý các tác vụ khác. Pipeline tính tần số thực tế của mỗi đoạn dựa trên median khoảng cách timestamp: `actual_hz = 1 / median(dt_seconds)`. Nếu `actual_hz` nằm ngoài khoảng [42,5; 57,5] Hz (tương đương ±15% so với mục tiêu 50 Hz), đoạn đó sẽ được đưa vào bước resample.

Phân tích phân phối tần số trong dữ liệu thô (Biểu đồ A) cho thấy chất lượng cảm biến rất ổn định: tỷ lệ khoảng cách thời gian nằm ngoài vùng chấp nhận ±15% chỉ là 0,1% (Đi bộ), 0,1% (Đứng) và 0,3% (Ngồi). Mức độ lệch này nhỏ hơn hai bậc so với ngưỡng 5% được thiết lập làm tiêu chí chấp nhận. Một số ít outlier xuất hiện ở vùng ~100 Hz và ~12 Hz là artifact phần cứng — Android đôi khi đẩy timestamp với double-rate ở vài frame liên tiếp hoặc bị throttle khi CPU bận, nhưng tỷ lệ rất thấp và được bước resample xử lý hoàn toàn.

### 4.4.2. Bước 2 — Tách đoạn và loại gap

Hàm `split_segments()` tách DataFrame thành các đoạn liên tục bằng cách xác định vị trí có khoảng cách timestamp vượt ngưỡng `MAX_GAP_SEC = 0,1` s (tương ứng 5 mẫu ở 50 Hz). Các đoạn có độ dài nhỏ hơn `WINDOW_SIZE = 100` mẫu (2 giây) bị loại bỏ để tránh cửa sổ trượt bắc qua điểm ngắt — điều này sẽ làm ô nhiễm nhãn activity. Cần phân biệt với ngưỡng `MIN_SAMPLES = 150` ở phía app (Mục 4.1.2): app từ chối lưu nếu phiên thu dưới 3 giây (tránh tạo file rác), còn pipeline loại đoạn dưới 2 giây sau khi tách gap (tránh window không đủ dài).

Phân tích khoảng cách timestamp (Biểu đồ B) trên một file mẫu cho thấy `dt ≈ 19–20` ms rất đều đặn, không phát hiện bất kỳ gap nào vượt ngưỡng 100 ms. Đây là minh chứng cho chất lượng cao của ứng dụng thu thập BioAuth Data Collection: cơ chế Foreground Service `dataSync` kết hợp với `keepScreenOn=true` ngăn được quá trình thu bị OS suspend giữa chừng. Sau khi áp dụng bộ lọc loại đoạn ngắn (<100 mẫu), tỷ lệ giữ lại rất cao: đi bộ giữ 99,1% (112/113 đoạn), đứng giữ 98,6% (146/148 đoạn) và ngồi giữ 96,2% (126/131 đoạn). Những đoạn bị loại đến từ các lần người dùng nhấn nút Dừng quá sớm (dưới 2 giây) — số lượng không đáng kể và không ảnh hưởng đến phân phối dữ liệu cuối cùng. Biểu đồ E minh họa rõ phân phối độ dài đoạn với median 10.148 mẫu (đi bộ), 3.536 mẫu (đứng) và 8.085 mẫu (ngồi), cho thấy phần lớn đoạn dài hàng nghìn mẫu — đủ tạo ra hàng trăm cửa sổ sau sliding window.

### 4.4.3. Bước 3 — Trực quan hóa phân bố để xác định outlier

Sau khi tách đoạn, pipeline thực hiện kiểm tra phân bố và spike trên từng kênh cảm biến bằng phương pháp Z-score per-window — đếm các điểm có |Z| > 3 trong từng cửa sổ với cùng kích thước `WINDOW_SIZE = 100` và stride 50 mà pipeline thực sự sử dụng. Phân tích trên một đoạn đi bộ 8 giây điển hình (Biểu đồ C) cho thấy số spike xuất hiện ở mức thấp và phân tán: khoảng 1–2 spike trên Accelerometer X, 6–8 spike trên Accelerometer Z, 2–4 spike trên Gyroscope X và 0–4 spike trên Gyroscope Z. Cụm spike trên Acc Z phản ánh các thay đổi đột ngột do gia tốc trọng lực khi cử động chân, là tín hiệu chuyển động thực, không phải nhiễu cảm biến.

Spike không được loại bỏ trực tiếp ở bước này; chúng được xử lý gián tiếp qua chuẩn hóa Z-score per-channel per-window ở Bước 5. Cách này bảo toàn dữ liệu thô và để mô hình học được tính biến thiên thực tế của tín hiệu, đồng thời tránh thông tin bị mất do clipping cứng.

### 4.4.4. Bước 4 — Resample về 50 Hz

Hàm `resample_segment()` sử dụng `numpy.interp` để nội suy từng kênh lên lưới thời gian đều với bước 20 ms (1/50 s). Phương pháp nội suy tuyến tính được chọn vì: (1) đơn giản, nhanh, không yêu cầu giả thiết về phổ tín hiệu; (2) không tạo overshoot hay ringing như spline bậc cao; (3) phù hợp với tín hiệu IMU tần số thấp mà không làm thay đổi hình dạng tín hiệu đáng kể.

Biểu đồ F minh họa hiệu quả của bước resample trên một đoạn đi bộ thực tế của userA (session_1). Trước resample, dt dao động giữa 19–20 ms (tần số trung bình 50,3 Hz, lệch nhẹ trong vùng chấp nhận ±15%). Sau resample, khoảng cách timestamp đều hoàn toàn 20 ms và hình dạng tín hiệu acc_x được bảo toàn tốt — các đỉnh và rãnh chính của tín hiệu giữ nguyên vị trí. Vì tỷ lệ đoạn cần resample trong dataset rất nhỏ (tổng cộng <1% các đoạn nằm ngoài ±15%), bước này chủ yếu đóng vai trò an toàn (precaution) hơn là sửa lỗi quy mô lớn.

### 4.4.5. Bước 5 — Chuẩn hóa Z-score per-channel per-window

Hàm `normalize_windows()` thực hiện chuẩn hóa Z-score độc lập trên từng kênh trong từng cửa sổ: với mỗi cửa sổ kích thước (100, 9), tính mean và std theo chiều thời gian, sau đó chuẩn hóa để đầu ra có mean ≈ 0, std ≈ 1. Giá trị std được clamp tối thiểu ở 1×10⁻⁸ để tránh chia cho 0 trong các đoạn tín hiệu bằng phẳng.

Phương pháp này giải quyết đồng thời ba vấn đề quan trọng được xác định qua phân tích dữ liệu thô (Biểu đồ D):

- DC offset và orientation bias: giá trị trung bình của acc_z khi đứng yên dao động trong khoảng 5–8 m/s² tùy theo cách cầm điện thoại (ngửa, sấp, nghiêng) — Z-score per-window loại bỏ offset này, làm cho đặc trưng bất biến với hướng đặt thiết bị.

- Khác biệt scale giữa các cảm biến: dữ liệu thô cho thấy Accelerometer có biên độ ~−42 → +39 m/s², Gyroscope ~−21 → +17 rad/s, Magnetometer ~−1.486 → +1.299 µT. Tỷ lệ scale chênh nhau hai bậc — nếu không chuẩn hóa, từ trường sẽ lấn át hoàn toàn hai cảm biến còn lại trong gradient của mạng nơ-ron.

- Khác biệt giữa người dùng: Biểu đồ 5 cho thấy mean ||acc|| khi đi bộ dao động từ 9,8 m/s² (userE, userK) đến 13,4 m/s² (userA) — chuẩn hóa per-window giúp mô hình tập trung vào pattern chuyển động thay vì giá trị tuyệt đối khác biệt giữa cá nhân.

Biểu đồ 4 minh họa window đã chuẩn hóa cho từng hoạt động: cả ba lớp đều có Z-score nằm trong khoảng hợp lý [−2,5; +2,5], tín hiệu đi bộ có dao động lớn và tần suất cao đặc trưng của chuyển động chu kỳ, trong khi đứng và ngồi hiển thị mẫu Z-score gần phẳng — phù hợp với bản chất tĩnh của hai hoạt động này.

### 4.4.6. Bước 6 — Cân bằng dữ liệu (chiến lược)

Pipeline cung cấp bốn chiến lược cân bằng dữ liệu (`none` / `oversample` / `undersample` / `augment`) thông qua tham số `BALANCE_STRATEGY`. Tuy nhiên, để tránh data leakage, hàm `balance_inertial()` được gọi sau bước split train/val/test (trong giai đoạn huấn luyện ở Chương 5), KHÔNG gọi trong `step2_preprocess.py`. Cách làm này đảm bảo các sample tạo ra qua augmentation không "rò" từ tập train sang tập test thông qua cùng một window gốc.

Trong dataset thực tế, tỷ lệ min/max windows giữa các activity đạt 24.606 / 25.227 ≈ 0,975 — gần như cân bằng hoàn hảo, vượt xa ngưỡng `IMBALANCE_THRESHOLD = 0,80` và không cần can thiệp cân bằng giữa các lớp activity. Augmentation sẽ được áp dụng trên lớp owner trong giai đoạn huấn luyện 1-vs-all (xem Chương 5) để cân bằng giữa owner (1 class) và impostors (10 class gộp lại) — đây là vấn đề mất cân bằng thực sự cần xử lý của bài toán open-set recognition.

## 4.5. Trích xuất đặc trưng touch và keystroke

Khác với inertial data được phân đoạn thành N windows để huấn luyện mạng CNN, touch data có đặc thù quan trọng: được thu riêng biệt theo phiên form (không đồng bộ từng giây với inertial) và số lượng events mỗi người dùng tương đối ít (675–787 taps, 728–889 scroll gestures, 667–806 keystroke events). Do đó, thay vì xây dựng một mô hình neural network riêng cho touch (vốn đòi hỏi hàng nghìn samples để train ổn định), đề tài áp dụng phương pháp profile-based scoring: mỗi session test được đại diện bởi một vector đặc trưng touch riêng (12 chiều), và điểm xác thực được tính bằng cosine similarity giữa profile của session đó với profile của chủ sở hữu (đã build từ train sessions). Phương pháp này phù hợp về mặt thống kê với lượng dữ liệu hiện có, đồng thời cho phép score-level fusion trực tiếp với điểm từ mô hình inertial.

### 4.5.1. Đặc trưng tap dynamics

Hàm `extract_tap_features()` trích xuất đặc trưng từ từng sự kiện tap hợp lệ (DOWN → UP liên tiếp) với lọc hold duration 0–500 ms để loại bỏ tap bất thường. Năm đặc trưng được trích xuất: `hold_ms` (thời gian giữ ngón tay, đã được tính chính xác từ raw `eventTime` ở phía app — Mục 4.1.3), tọa độ `x` và `y` (vị trí chạm), `pressure` (áp lực) và `size` (diện tích tiếp xúc). Biểu đồ 8 (trái) cho thấy phân phối hold_ms giữa các người dùng có sự khác biệt rõ ràng — median dao động từ ~46 ms (userL) đến ~94 ms (userF) — đây là đặc trưng tốt cho xác thực cá nhân, gần như tăng gấp đôi giữa hai người dùng.

### 4.5.2. Đặc trưng scroll dynamics

Hàm `extract_scroll_features()` ghép chuỗi sự kiện DOWN → MOVE... → UP theo `pointer_id` thành gesture hoàn chỉnh, lọc các gesture có thời gian 10 < dt < 5.000 ms. Tám đặc trưng được tính từ raw events thu được ở phía app: `duration_ms`, `delta_y`, `delta_x`, `distance_px`, `velocity` (pixel/ms), `n_moves` (số sự kiện MOVE), `pressure_mean` và `direction` (up/down). Việc tính các đặc trưng phái sinh ở phía Python thay vì ở client cho phép thay đổi công thức (vd thử median velocity thay vì mean) mà không cần cập nhật app. Heatmap vùng chạm (Biểu đồ 8 phải) cho thấy phần lớn tương tác tập trung tại vùng trung tâm-trên màn hình (~150–250 pixel theo trục X, ~50–80 pixel theo trục Y), phù hợp với thao tác điện thoại tự nhiên — vùng "thumb zone" của người thuận tay phải.

### 4.5.3. Đặc trưng keystroke dynamics

Hàm `extract_keystroke_features()` xử lý inter-key interval (IKT) — khoảng thời gian giữa hai lần nhấn phím liên tiếp — với các bộ lọc: loại hàng đầu (IKT = 0), loại thao tác xóa (`is_delete = True` đã được app gắn cờ ở Mục 4.1.3 qua kiểm tra `count == 0`) và loại outlier IKT > 3.000 ms. Timestamp keystroke đã được đồng nhất với inertial qua UTC offset (Mục 4.1.5), tránh lệch nguồn thời gian. Kết quả là phân phối IKT thuần túy phản ánh tốc độ và nhịp điệu gõ phím cá nhân. Biểu đồ 9 (fingerprint đa phương thức) cho thấy userL có IKT median = −2,4σ và IKT σ = +0,2σ so với trung bình — đặc trưng gõ phím rất nhanh nhưng không ổn định, là dấu hiệu định danh đặc thù.

### 4.5.4. Touch profile và phương pháp cosine similarity scoring

Sau khi trích xuất các đặc trưng tap, scroll và keystroke, hàm `make_touch_feat_vector()` tổng hợp thành một vector profile 12 chiều đại diện cho hành vi touch: `[holdMs_mean, holdMs_std, holdMs_median, pressure_mean, size_mean, velocity_mean, velocity_std, duration_ms_mean, distance_px_mean, interKeyMs_mean, interKeyMs_std, interKeyMs_median]`. Vector này được lưu vào `touch_feats.npy` dưới dạng shape `(12,)`.

Điểm xác thực touch (touch score) cho một session candidate được tính như sau: lúc TRAIN, owner profile được build từ TRAIN sessions của owner; scaler Z-score được fit trên train profiles của tất cả user đã biết. Lúc SCORE, cho mỗi test session, profile của riêng session đó được tính trực tiếp từ CSV (chỉ giữ rows thuộc đúng `session_id` này), Z-score normalize bằng scaler đã fit, sau đó tính cosine similarity với owner profile và ánh xạ về [0, 1]: `score_t = (cos(v_session, v_owner) + 1) / 2`. Cách thiết kế này đảm bảo open-set evaluation trung thực — hai test sessions của cùng một impostor có thể nhận điểm khác nhau (vì hành vi mỗi session khác nhau), thay vì cùng một điểm pre-computed dựa trên identity.

## 4.6. Kết quả tiền xử lý và phân tích chất lượng dataset

### 4.6.1. Phân phối dữ liệu inertial sau tiền xử lý

Sau khi hoàn thành pipeline tiền xử lý, dữ liệu inertial đạt chất lượng cao với 74.490 windows được phân bổ rất cân bằng giữa ba lớp: Đi bộ chiếm 33,1% (24.657 windows), Đứng chiếm 33,0% (24.606 windows) và Ngồi chiếm 33,9% (25.227 windows). Sự chênh lệch tối đa giữa lớp ít nhất và nhiều nhất chỉ 2,5%, gần như không cần can thiệp cân bằng khi huấn luyện. Biểu đồ 1 (trái) hiển thị tổng windows theo activity, Biểu đồ 1 (phải) hiển thị tổng samples theo người dùng — tất cả vượt xa ngưỡng tối thiểu 4.500 samples với khoảng dao động hẹp 325.000–370.000 samples/người, cho thấy tính nhất quán cao trong khối lượng dữ liệu thu được giữa các cá nhân.

Biểu đồ 10 hiển thị số windows thực tế trên từng (user, session) tách theo hoạt động. Hầu hết các giá trị nằm trong khoảng 300–500 windows/session, phản ánh tốt thiết kế thu thập 4–5 phút/hoạt động. Một số trường hợp ngoại lệ có giá trị cao đáng kể: userE đứng session_1 đạt 577 windows, userG ngồi session_3 đạt 805 windows — những phiên này có thời gian thu dài hơn bình thường (do người dùng quên dừng kịp thời) nhưng không phải artifact lỗi dữ liệu. Sự khác biệt này là hợp lệ và sẽ được hấp thụ tự nhiên qua bước split train/val/test theo session ở Chương 5.

### 4.6.2. Inter-user variability — cơ sở của behavioral biometrics

Phân tích inter-user variability (Biểu đồ 5) qua giá trị mean ||acc|| cho thấy sự khác biệt hành vi giữa các người dùng là có thực và đủ lớn để phân biệt. Đối với hoạt động đi bộ (động), khoảng giá trị trải rộng từ 9,8 m/s² (userE, userK) đến 13,4 m/s² (userA) — chênh lệch ~37% giữa người có dáng đi nhẹ nhàng nhất và người mạnh nhất. Trong khi đó, ở các hoạt động tĩnh (đứng, ngồi), khoảng dao động hẹp lại còn 9,7–10,1 m/s² (gần với gia tốc trọng lực 9,81 m/s²) vì điện thoại gần như đứng yên. Đây là bằng chứng định lượng cho luận điểm cốt lõi của behavioral biometrics: pattern hoạt động động cung cấp tín hiệu định danh mạnh hơn hoạt động tĩnh, vì nó phản ánh đặc trưng vận động cá nhân — dáng đi, lực bước chân, nhịp di chuyển.

Biểu đồ 3 (violin plot) khẳng định thêm điều này ở góc độ phân phối: ||acc|| khi đi bộ phân bố rộng (4,5–21 m/s², median ~10,1) trong khi đứng và ngồi gần như chỉ là một đỉnh nhọn quanh 9,81 m/s². Sự khác biệt giữa pattern động và tĩnh sẽ được mô hình CNN 1D học để phân biệt hoạt động, sau đó tận dụng pattern động để định danh người dùng.

### 4.6.3. Cấu trúc tương quan kênh cảm biến

Ma trận tương quan giữa các kênh cảm biến (Biểu đồ 6) tiết lộ ba điểm quan trọng cho thiết kế mô hình. Thứ nhất, khi đi bộ (hoạt động động), cặp acc_y ↔ acc_z có hệ số tương quan 0,7 — phản ánh đồng pha của hai trục tiếp xúc gia tốc trọng lực khi cơ thể nghiêng theo nhịp bước; trong khi cùng cặp này khi ngồi chỉ có hệ số −0,1 do điện thoại gần như đứng yên. Thứ hai, kênh mag_x có tương quan −0,3 với acc_x khi đi bộ nhưng chỉ −0,1 khi ngồi — khẳng định từ trường thay đổi theo orientation khi di chuyển. Thứ ba, gyroscope có tương quan rất thấp với accelerometer và magnetometer trong cả hai hoạt động (giá trị đều dưới 0,1 trị tuyệt đối), xác nhận tính bổ sung độc lập giữa ba loại cảm biến và lý do giữ cả 9 kênh khi dùng CNN 1D.

Với SVM dùng đặc trưng thủ công, có thể cân nhắc PCA giảm chiều cho cụm magnetometer (3 kênh có tương quan trung bình giữa các trục) trong khi giữ nguyên accelerometer và gyroscope độc lập.

### 4.6.4. Fingerprint đa phương thức và khả năng xác thực

Biểu đồ 9 (fingerprint đa phương thức) tổng hợp Z-score của 11 đặc trưng đại diện — 6 inertial (Acc walk μ/σ, Acc stand μ/σ, Acc sit μ/σ) và 5 touch (Hold med, Scroll vel, Scroll dist, IKT med, IKT σ) — cho từng người dùng, cho phép quan sát "vân tay hành vi" của 11 cá nhân trong một bảng duy nhất.

Phân tích cho thấy inertial và touch bổ sung cho nhau theo nghĩa: userA nổi bật rõ qua inertial (Acc walk μ = +2,5σ, Acc stand μ/σ = +1,3/+1,8σ) và touch (Scroll vel = +2,1σ, Scroll dist = +1,5σ) — fingerprint mạnh ở cả hai modal. Nhưng userL chỉ nổi bật qua touch (IKT med = −2,4σ, Hold med = −2,2σ — gõ rất nhanh, giữ phím rất ngắn) trong khi inertial gần với trung bình. Ngược lại, userE chỉ nổi bật qua inertial (Acc walk σ = −1,7σ — đi bộ rất nhẹ nhàng) trong khi touch trung tính. Đây là bằng chứng định lượng mạnh mẽ cho thiết kế multi-modal fusion: với một số người dùng, inertial là tín hiệu định danh chính; với người khác, touch chiếm ưu thế. Kết hợp cả hai modal qua score-level fusion (Chương 5) sẽ cải thiện độ chính xác cho cả hai trường hợp, đặc biệt cho những người dùng có một nguồn tín hiệu ít đặc trưng.

| **Tiêu chí** | **Kết quả thực tế** | **Đánh giá** |
|---|---|---|
| Số lượng dữ liệu | 74.490 windows inertial; ≥667 sự kiện cho mỗi loại touch trên 11 người | **Đủ để train và đánh giá** |
| Cân bằng lớp activity | Đi bộ 33,1% / Đứng 33,0% / Ngồi 33,9% | **Cân bằng gần hoàn hảo** |
| Chất lượng chuẩn hóa | Z-score per-window loại bỏ offset và scale differences | **Tốt** |
| Inter-user variability | userA, userL nổi bật rõ; chênh lệch 37% trong mean \|\|acc\|\| đi bộ | **Cơ sở xác thực rõ ràng** |
| Bổ sung đa modal | userA mạnh ở cả hai modal; userE/userL bổ sung lẫn nhau | **Fusion sẽ cải thiện EER** |
| Sẵn sàng train model | Dataset đã pass tất cả kiểm tra chất lượng | **Sẵn sàng** |

*Bảng 4.5. Tổng kết đánh giá chất lượng dataset.*

## 4.7. Hạn chế và hướng cải thiện

Dataset hiện tại tồn tại một số hạn chế cần thừa nhận. Thứ nhất, dữ liệu được thu trong môi trường có kiểm soát (lab-based) với các tác vụ chuẩn hóa, chưa phản ánh đầy đủ sự biến thiên hành vi trong điều kiện sử dụng tự nhiên — người dùng thực tế có thể vừa đi vừa nghe nhạc, vừa chat, hoặc đang ở các tư thế phức tạp hơn ba hoạt động chuẩn. Thứ hai, đề tài chỉ thu 3 hoạt động (đi bộ, đứng, ngồi) thay vì 6 hoạt động như thiết kế ban đầu — đây là đánh đổi có chủ ý nhằm đảm bảo đủ thời gian thu thập chất lượng cao cho mỗi hoạt động trong giới hạn buổi thu, nhưng làm hạn chế khả năng tổng quát hóa khi triển khai. Thứ ba, dữ liệu touch và keystroke được thu trong ngữ cảnh form câu hỏi cố định, chưa mô phỏng đầy đủ sự đa dạng của các tác vụ thực tế như duyệt web, chat đa luồng hay chơi game. Thứ tư, quy mô 11 người tham gia tuy đủ cho one-vs-rest evaluation nhưng còn nhỏ so với các dataset chuẩn behavioral biometrics quốc tế (HMOG có 100 user, BB-MAS có 117 user) — kết quả EER đo trên 11 user có khả năng không phản ánh chính xác hiệu năng thực tế khi triển khai cho hàng nghìn người dùng.

Về mặt kỹ thuật ứng dụng, một điểm cần cải thiện là việc ứng dụng dùng hai nguồn thời gian khác nhau cho inertial (`elapsedRealtimeNanos`) và touch (`uptimeMillis`) — chấp nhận được trong điều kiện hiện tại vì màn hình luôn ON nhưng sẽ tạo lệch nhỏ nếu mở rộng sang chế độ thu nền dài hạn. Đề xuất chuyển toàn bộ về `elapsedRealtimeNanos` ở phiên bản tiếp theo. Ngoài ra, ứng dụng chưa khai thác `WAKE_LOCK` (chỉ khai báo trong manifest, không gọi `acquire()` ở đâu) — có thể bổ sung khi cần thu trong nền liên tục lâu hơn.

Các hướng cải thiện cho nghiên cứu tiếp theo bao gồm: (1) thu thập dữ liệu trong môi trường tự nhiên qua Foreground Service dài hạn, ghi nhận hành vi trong vài tuần thay vì 6 phiên có chủ đích; (2) mở rộng số hoạt động lên 6 (thêm chạy bộ và đi cầu thang) theo thiết kế gốc; (3) tăng số người tham gia lên ≥30 để đánh giá khả năng tổng quát hóa và so sánh trực tiếp với benchmark quốc tế; (4) thử nghiệm trên nhiều dòng máy Android khác nhau để đánh giá tác động của đa dạng phần cứng cảm biến đến độ chính xác xác thực.
