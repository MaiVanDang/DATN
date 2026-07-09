# Kịch bản nội dung slide bảo vệ ĐATN

Đề tài: **Phát triển hệ thống nhận diện người dùng qua hành vi sử dụng điện thoại với cơ chế xác thực dự phòng bằng cử chỉ chuyển động** — SV Mai Văn Đăng.

> Bố cục 13 slide, đi theo mạch: **Vấn đề → Giải pháp → Chức năng → Thu thập dữ liệu → Tiền xử lý → Mô hình → Kết quả → Triển khai → Kết luận**. Mỗi slide gồm **Tiêu đề** + **gạch đầu dòng ngắn** + **📊/🖼️ chỗ cần chèn bảng/ảnh**.

---

## Slide 1 — Trang bìa
- **Tên đề tài** (đầy đủ)
- Sinh viên: **Mai Văn Đăng** — MSSV 20225699
- GVHD: *(điền)* · Trường/Chương trình: *(điền)* · Hà Nội, *(tháng/năm)*

## Slide 2 — Nội dung trình bày
- Đặt vấn đề & mục tiêu
- Tổng quan giải pháp & chức năng
- Xây dựng dữ liệu và tiền xử lý
- Xây dựng mô hình
- Kết quả đánh giá
- Triển khai & kết luận

## Slide 3 — Đặt vấn đề & Mục tiêu
- Xác thực hiện nay chỉ **một lần** lúc mở khóa → mở khóa xong **không kiểm soát lại** danh tính
- **Sinh trắc học hành vi:** xác thực **liên tục, ngầm**, dùng cảm biến có sẵn
- Khoảng trống: thiếu **phát hiện người lạ (tập mở)** + thiếu **cơ chế phản ứng** khi bất thường
- **Mục tiêu:** xác thực hành vi liên tục **on-device** + phát hiện người lạ + **dự phòng cử chỉ lắc**

## Slide 4 — Tổng quan giải pháp
- Pipeline **4 giai đoạn:** Thu thập → **Tiền xử lý** → Huấn luyện & hàm quyết định → Triển khai on-device
- Nhánh chính: **cảm biến quán tính**; nhánh **chạm/cuộn**: đối chứng
- 🖼️ **Chèn Hình:** sơ đồ kiến trúc tổng quan — `Hinh_ve/kientruc_tongquan.png`

## Slide 5 — Chức năng hệ thống (use-case)
- **Người tham gia** → thu thập dữ liệu (đăng ký, thu cảm biến, điền biểu mẫu, xuất dữ liệu)
- **Chủ sở hữu** → xác thực (chọn chế độ, đăng ký sinh trắc, đăng ký cử chỉ lắc, xác thực liên tục, dự phòng)
- **Người lạ** → bị nhánh xác thực phát hiện & từ chối
- 🖼️ **Chèn 2 ảnh cạnh nhau:** `Hinh_ve/UC_BioAuth_Data_Collection.png` + `Hinh_ve/UC_BioAuth_Authenticator.png`

> *Nói:* hệ thống là **2 app** — một để tạo dữ liệu, một để xác thực; sơ đồ cho thấy mỗi app làm gì, cho ai.

## Slide 6 — Thu thập dữ liệu
- Tự xây dựng **app thu thập** chuyên dụng: đăng ký → thu cảm biến → biểu mẫu tương tác → xuất dữ liệu
- Bộ dữ liệu **thực tế**: **26 người**, **19 dòng thiết bị**, điều kiện sử dụng **tự nhiên**
- Mỗi người **6 phiên**: 3 hoạt động (đi bộ, đứng, ngồi) + thao tác chạm/cuộn
- 🖼️ *(tùy chọn)* ảnh giao diện app thu thập — `Hinh_ve/Collectdata/screen_collect.jpg`

> *Nói:* điểm khác biệt là thu **đa thiết bị, tự nhiên** → sát điều kiện triển khai thật hơn phòng lab.

## Slide 7 — Tiền xử lý dữ liệu  ⭐
- **Kiểm tra chất lượng & phân đoạn:** cắt tín hiệu tại khoảng trống > ngưỡng, bỏ đoạn quá ngắn
- **Cửa sổ trượt:** cắt thành cửa sổ **200 mẫu (~4 giây)**, bước trượt 20 → chồng lấn cao, tăng số mẫu
- **Chuẩn hóa z-score** theo từng cửa sổ → đưa mọi thiết bị/người về **cùng thang**, khử lệch nền
- Xuất 2 cấu hình: **chỉ đi bộ (54.823 cửa sổ)** và **toàn bộ (163.145 cửa sổ)**; touch → **34.836** sự kiện chạm
- 🖼️ **Chèn 2–3 hình:** phân đoạn (`Preprocess/phan_doan_a.png`), cửa sổ trượt (`Preprocess/cua_so_truot.png`), z-score trước/sau (`Preprocess/2a_zscore_truoc.png` + `2b_zscore_sau.png`)

> *Nói:* đây là bước quyết định chất lượng đầu vào — nhấn vào **z-score giúp khử khác biệt phần cứng** giữa 19 dòng máy.

## Slide 8 — Xây dựng mô hình
- **CNN 1D** mã hóa cửa sổ 200×9 → **embedding 128 chiều** (khảo sát cùng CNN-LSTM, CNN-BiLSTM)
- Nhận dạng tập mở: **anchor → cosine → z-norm cohort → sigmoid** (hàm `cos_znorm`)
- Nhánh touch **33 chiều** (Random Forest) + dung hợp — vai trò **đối chứng**
- 📊 **Chèn Bảng 4.1:** số tham số / dung lượng 3 kiến trúc

> *Nói:* backbone chỉ để học không gian embedding; xác thực bằng đối sánh anchor nên **thêm người mới không cần train lại**.

## Slide 9 — Kết quả: So sánh kiến trúc
- Chỉ số: **FAR, FRR, EER, AUC**; 2 kịch bản **tập đóng** và **tập mở**
- Tập đóng: **CNN 1D ≈ 3,4% EER** (nhóm tốt nhất)
- Tập mở (khó hơn): **CNN 1D tốt nhất** — CNN-LSTM/BiLSTM ~30% (quá khớp)
- Toàn bộ 3 hoạt động **> chỉ đi bộ** → chọn **CNN 1D**
- 📊 **Chèn Bảng 5.1 + 5.3** (tập đóng / tập mở)

## Slide 10 — Kết quả: Hàm chấm điểm
- So **4 hàm** (cos_mean, cos_knn, **cos_znorm**, maha) — **leave-users-out 6 vòng**
- **cos_znorm tốt nhất:** EER tập mở **10,83% ± 6,51%**; tập đóng **2,24%**
- Dung hợp nhánh chạm **không cải thiện** tập mở → triển khai **đơn modal quán tính**
- 📊 **Chèn Bảng 5.5**

> *Nói:* tách 2 nguồn giảm lỗi — đổi giao thức đánh giá (21% → 14,7%) và đổi hàm chấm điểm (14,7% → 10,8%).

## Slide 11 — Triển khai & sản phẩm
- Suy luận **trực tiếp trên thiết bị** qua **TensorFlow Lite** → riêng tư, chạy ngoại tuyến
- Xác thực liên tục: **máy trạng thái 3 mức**, làm mịn **EWMA**, **ngưỡng riêng từng chủ máy**
- Dự phòng khi nghi ngờ người lạ: **cử chỉ lắc** (mã 4–8 chữ số), không quay về PIN
- Sản phẩm: app thu thập, app xác thực, **web demo**

## Slide 12 — Giao diện & Demo
- **App xác thực:** chọn chế độ → đăng ký sinh trắc → đăng ký cử chỉ lắc → xác thực liên tục
- **Web demo:** hiển thị điểm quán tính / touch / fusion + 2 kết luận để đối chiếu minh bạch
- 🖼️ **Chèn ảnh:** `Hinh_ve/Demo_app/*` + web demo

## Slide 13 — Kết luận & Cảm ơn
- Hệ thống **hoàn chỉnh**: dữ liệu → tiền xử lý → mô hình → đánh giá → triển khai on-device
- Chốt: **CNN 1D + cos_znorm**; EER **đóng 2,24%** / **mở 10,83%**
- Hạn chế: tập mở còn ~10–11%; touch chưa hiệu quả; chưa đánh giá tấn công chủ động
- Hướng phát triển: học biểu diễn (ArcFace), mở rộng dữ liệu, hoàn thiện touch
- **XIN CẢM ƠN THẦY CÔ!**

---

## Gợi ý dựng slide
- **Ít chữ, nhiều ý chính** — mỗi gạch đầu dòng ≤ 1 dòng; nói chi tiết bằng miệng.
- Con số bôi đậm (2,24% / 10,83% / 163.145…) là **điểm nhấn** → cỡ lớn/màu nổi.
- Slide **7 (Tiền xử lý)** và **9–10 (Kết quả)** là trọng tâm, hội đồng hay hỏi → chuẩn bị kỹ.
- Ảnh/bảng lấy trong `docs/report/DATN/Hinh_ve/` và bảng trong `Chuong/5_Danh_gia_thuc_nghiem.tex`.
