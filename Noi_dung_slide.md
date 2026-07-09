# Kịch bản nội dung slide bảo vệ ĐATN

Đề tài: **Phát triển hệ thống nhận diện người dùng qua hành vi sử dụng điện thoại với cơ chế xác thực dự phòng bằng cử chỉ chuyển động** — SV Mai Văn Đăng.

> Bố cục khớp với template 13 slide. Mỗi slide gồm **Tiêu đề** + **các gạch đầu dòng** (giữ ngắn gọn) + **📊/🖼️ chỗ cần chèn bảng/ảnh**. Dòng *Nói:* là gợi ý lời thuyết trình (không đưa lên slide).

---

## Slide 1 — Trang bìa
- **Tên đề tài:** Phát triển hệ thống nhận diện người dùng qua hành vi sử dụng điện thoại với cơ chế xác thực dự phòng bằng cử chỉ chuyển động
- Sinh viên: **Mai Văn Đăng** — MSSV 20225699
- Giảng viên hướng dẫn: *(điền)*
- Trường/Chương trình: *(điền)* · Hà Nội, *(tháng/năm)*

## Slide 2 — Nội dung trình bày
- Đặt vấn đề & mục tiêu
- Tổng quan giải pháp
- Xây dựng dữ liệu & mô hình
- Kết quả đánh giá
- Triển khai & sản phẩm
- Kết luận

## Slide 3 — [Trang phân mục] ĐẶT VẤN ĐỀ & MỤC TIÊU
*(Slide phân mục, chỉ ghi tiêu đề lớn)*

## Slide 4 — Đặt vấn đề & Mục tiêu
- Xác thực hiện nay chỉ diễn ra **một lần** lúc mở khóa (PIN/vân tay/khuôn mặt) → mở khóa xong **không kiểm soát lại** danh tính
- **Sinh trắc học hành vi:** xác thực **liên tục, ngầm**, tận dụng cảm biến có sẵn — không cần phần cứng riêng
- Khoảng trống nghiên cứu: thiếu **phát hiện người lạ (tập mở)** và thiếu **cơ chế phản ứng** khi bất thường
- **Mục tiêu:** hệ thống xác thực hành vi liên tục **chạy trên thiết bị**, phát hiện người lạ, kèm **dự phòng bằng cử chỉ lắc**

> *Nói:* nhấn mạnh điểm yếu "xác thực một lần" và hai khoảng trống (open-set + phản ứng) → đó là động lực đề tài.

## Slide 5 — Tổng quan giải pháp
- Pipeline **4 giai đoạn:** Thu thập → Tiền xử lý → Huấn luyện & hàm quyết định → Triển khai on-device
- Nhánh chính: **cảm biến quán tính** (đi bộ/đứng/ngồi); nhánh **chạm/cuộn**: đối chứng
- Sản phẩm: **2 app Android** (thu thập + xác thực) + **web demo** trực quan
- 🖼️ **Chèn Hình:** sơ đồ kiến trúc tổng quan — `docs/report/DATN/Hinh_ve/kientruc_tongquan.png`

## Slide 6 — Xây dựng dữ liệu
- **26 người**, **19 dòng thiết bị**, thu trong điều kiện sử dụng **tự nhiên**
- Mỗi người **6 phiên**: 3 hoạt động + biểu mẫu tương tác (chạm, cuộn)
- Tiền xử lý: phân đoạn → **cửa sổ 200×9** (~4 giây ở 50 Hz) → **chuẩn hóa z-score**
- Quy mô: **163.145** cửa sổ (toàn bộ) / **54.823** (đi bộ); **34.836** sự kiện chạm
- 🖼️ *(tùy chọn)* Hình phân bố dữ liệu — `Hinh_ve/Preprocess/phanbo_theo_nguoi.png`

## Slide 7 — Xây dựng mô hình
- **CNN 1D** mã hóa cửa sổ 200×9 → **embedding 128 chiều** (khảo sát cùng CNN-LSTM, CNN-BiLSTM)
- Nhận dạng tập mở: **anchor → cosine → z-norm cohort → sigmoid** (hàm `cos_znorm`)
- Nhánh touch **33 chiều** (Random Forest) + dung hợp điểm — vai trò **đối chứng**
- 📊 **Chèn Bảng 4.1:** số tham số / dung lượng 3 kiến trúc (CNN ≈ 80k / 323 KB…)

> *Nói:* backbone chỉ để học không gian embedding; xác thực bằng đối sánh anchor nên **thêm người mới không cần train lại**.

## Slide 8 — [Trang phân mục] KẾT QUẢ ĐÁNH GIÁ & TRIỂN KHAI

## Slide 9 — Kết quả: So sánh kiến trúc
- Chỉ số: **FAR, FRR, EER, AUC**; 2 kịch bản **tập đóng** và **tập mở**
- Tập đóng: **CNN 1D ≈ 3,4% EER** (nhóm tốt nhất)
- Tập mở (khó hơn): **CNN 1D tốt nhất** — CNN-LSTM/BiLSTM ~30% (dấu hiệu quá khớp)
- Ngữ cảnh **toàn bộ 3 hoạt động > chỉ đi bộ** → chọn **CNN 1D** (gọn nhẹ nhất + tổng quát tốt nhất)
- 📊 **Chèn Bảng 5.1 + 5.3** (tập đóng / tập mở, cấu hình toàn bộ)

## Slide 10 — Kết quả: Hàm chấm điểm
- So **4 hàm** (cos_mean, cos_knn, **cos_znorm**, maha) trên cùng bộ mã hóa — **leave-users-out 6 vòng**
- **cos_znorm tốt nhất:** EER tập mở **10,83% ± 6,51%**; tập đóng **2,24%**
- Dung hợp nhánh chạm **không cải thiện** tập mở → bản triển khai cuối **đơn modal quán tính**
- 📊 **Chèn Bảng 5.5:** so sánh các hàm chấm điểm

> *Nói:* tách rõ 2 nguồn giảm lỗi — đổi giao thức đánh giá (21% → 14,7%) và đổi hàm chấm điểm (14,7% → 10,8%).

## Slide 11 — Triển khai & sản phẩm
- Suy luận **trực tiếp trên thiết bị** qua **TensorFlow Lite** → riêng tư, chạy ngoại tuyến
- Xác thực liên tục: **máy trạng thái 3 mức**, làm mịn **EWMA**, **ngưỡng hiệu chuẩn riêng từng chủ máy**
- Dự phòng khi nghi ngờ người lạ: **cử chỉ lắc** (mã 4–8 chữ số) — không quay về PIN
- Sản phẩm: app thu thập, app xác thực, **web demo** trực quan hóa quyết định

## Slide 12 — Giao diện & Demo
- **App xác thực:** chọn chế độ → đăng ký sinh trắc → đăng ký cử chỉ lắc → xác thực liên tục
- **Web demo:** hiển thị điểm quán tính / touch / fusion + 2 kết luận để đối chiếu minh bạch
- 🖼️ **Chèn ảnh chụp màn hình:** `Hinh_ve/Demo_app/*` (đăng ký, vận hành, dự phòng) + web demo

## Slide 13 — Kết luận & Cảm ơn
- Đã xây dựng **hệ thống hoàn chỉnh**: dữ liệu → mô hình → đánh giá → triển khai on-device
- Chốt cấu hình: **CNN 1D + cos_znorm**; EER **đóng 2,24%** / **mở 10,83%**
- Hạn chế: tập mở còn ~10–11%; nhánh touch chưa hiệu quả; chưa đánh giá tấn công chủ động
- Hướng phát triển: học biểu diễn (ArcFace), mở rộng dữ liệu, hoàn thiện nhánh touch
- **XIN CẢM ƠN THẦY CÔ!**

---

## Gợi ý khi dựng slide
- **Ít chữ, nhiều ý chính** — mỗi gạch đầu dòng ≤ 1 dòng; nói chi tiết bằng miệng.
- Các con số bôi đậm (2,24% / 10,83% / 163.145…) là **điểm nhấn** — nên để cỡ lớn hoặc màu nổi.
- Slide 5, 7, 9, 10, 12 là **slide "nặng" nhất** (có hình/bảng) — chuẩn bị kỹ vì hội đồng hay hỏi ở đây.
- Nếu thời gian eo hẹp: gộp slide 3 và 8 (trang phân mục) vào slide nội dung liền sau.
- Ảnh/bảng lấy sẵn trong `docs/report/DATN/Hinh_ve/` và các bảng trong `Chuong/5_Danh_gia_thuc_nghiem.tex`.
