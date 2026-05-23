# Data Collection App

Ứng dụng Android thu thập dữ liệu hành vi (IMU + cảm ứng) để xây dựng dataset huấn luyện.

> **Lưu ý**: Tên thư mục `data_collectiom` có typo (thiếu 'n'), giữ nguyên để không phá vỡ git history.

## Cấu trúc dữ liệu thu thập

Mỗi buổi thu thập tạo ra một `session_X/` với các file:

| File | Nội dung |
|------|----------|
| `walking_att1.csv` | IMU 9-axis khi đi bộ (timestamp_ms, acc_x/y/z, gyro_x/y/z, mag_x/y/z) |
| `standing_att1.csv` | IMU khi đứng yên |
| `sitting_att1.csv` | IMU khi ngồi |
| `tap_r1.csv` | Sự kiện chạm màn hình (phase, x, y, hold_ms, timestamp_ms) |
| `scroll_r1.csv` | Sự kiện vuốt (phase, x, y, pointer_id, timestamp_ms) |
| `keystroke_r1.csv` | Gõ bàn phím (inter_key_ms, is_delete, timestamp_ms) |

## Mục tiêu thu thập tối thiểu mỗi user

| Loại | Mục tiêu |
|------|----------|
| Walking | 18 phút |
| Standing | 18 phút |
| Sitting | 18 phút |
| Tap gestures | 600 |
| Scroll gestures | 600 |
| Keystroke events | 600 |

Kiểm tra bằng: `python ml_pipeline/step1_quality_check.py` (chạy từ thư mục `ml_pipeline/`)

## Quy trình

1. Cài app lên điện thoại Android của người dùng
2. Thu thập qua nhiều buổi (session), mỗi buổi ~30-60 phút
3. Export CSV về máy tính → đặt vào `ml_pipeline/data/<user_id>/session_X/`
4. Chạy `step1_quality_check.py` để kiểm tra đủ data
5. Chạy `step2_preprocess.py` để tiền xử lý
