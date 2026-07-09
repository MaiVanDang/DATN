# Script thuyết trình bảo vệ ĐATN

Đề tài: *Phát triển hệ thống nhận diện người dùng qua hành vi sử dụng điện thoại với cơ chế xác thực dự phòng bằng cử chỉ chuyển động* — SV Mai Văn Đăng.

> Văn nói, xưng **"em"**. Tổng ~**11–12 phút**. `[~Xs]` = thời lượng gợi ý. *(chỉ slide)* = lúc chỉ vào hình/bảng.

---

### Slide 1 — Bìa `[~30s]`
Em kính chào các thầy, các cô trong hội đồng. Em là Mai Văn Đăng. Hôm nay em xin phép được trình bày đồ án tốt nghiệp với đề tài **"Phát triển hệ thống nhận diện người dùng qua hành vi sử dụng điện thoại, kèm cơ chế xác thực dự phòng bằng cử chỉ chuyển động"**. Sau đây em xin bắt đầu.

### Slide 2 — Nội dung `[~15s]`
Bài trình bày gồm các phần: **đặt vấn đề và mục tiêu**, **tổng quan giải pháp và chức năng**, sau đó là **cách em xây dựng dữ liệu và tiền xử lý**, rồi **mô hình**, **kết quả đánh giá**, và cuối cùng là **triển khai và kết luận**.

### Slide 3 — Đặt vấn đề & Mục tiêu `[~1p15s]`
Thưa hội đồng, điện thoại ngày nay lưu gần như toàn bộ thông tin cá nhân và tài chính, nhưng lại chủ yếu được bảo vệ bằng **xác thực một lần** lúc mở khóa — như PIN, vân tay hay khuôn mặt. Vấn đề là **sau khi mở khóa, hệ thống không kiểm tra lại danh tính nữa**; nếu máy rơi vào tay người khác lúc đang mở, họ dùng tiếp mà không bị chặn.

Hướng giải quyết là **sinh trắc học hành vi**: thay vì một lần, hệ thống **liên tục** nhận diện người dùng qua cách họ cầm máy, đi lại, chạm màn hình — hoàn toàn ngầm, chỉ dùng cảm biến **có sẵn**.

Khi khảo sát các nghiên cứu trước, em thấy hai khoảng trống: **một là** phần lớn chưa **phát hiện được người lạ** — người chưa từng thấy khi huấn luyện; **hai là** khi phát hiện bất thường thì cũng chỉ khóa màn hình, chưa có cách **xác thực lại**. Vì vậy **mục tiêu** của em là một hệ thống xác thực hành vi **liên tục, chạy trên thiết bị**, **phát hiện người lạ**, và có **dự phòng bằng cử chỉ lắc**.

### Slide 4 — Tổng quan giải pháp `[~45s]`
*(chỉ sơ đồ)* Đây là kiến trúc tổng thể, gồm **bốn giai đoạn**: thu thập dữ liệu, **tiền xử lý**, huấn luyện và hình thành hàm quyết định, rồi triển khai xuống thiết bị. Nhánh chính là **cảm biến quán tính**, còn nhánh **chạm và cuộn** em giữ ở vai trò đối chứng.

### Slide 5 — Chức năng hệ thống `[~45s]`
*(chỉ 2 sơ đồ use-case)* Hệ thống được hiện thực thành **hai ứng dụng**. Ứng dụng thứ nhất dành cho **người tham gia** để **thu thập dữ liệu** — gồm đăng ký, thu cảm biến, điền biểu mẫu và xuất dữ liệu. Ứng dụng thứ hai dành cho **chủ máy** để **xác thực** — chọn chế độ, đăng ký sinh trắc, đăng ký cử chỉ lắc, rồi xác thực liên tục; còn **người lạ** là tác nhân bị hệ thống phát hiện và từ chối.

### Slide 6 — Thu thập dữ liệu `[~1p]`
Trước hết em xây dựng một **ứng dụng thu thập** chuyên dụng để lấy dữ liệu nhất quán trên nhiều người và nhiều máy. Bộ dữ liệu gồm **26 người tham gia**, thu trên **19 dòng thiết bị**, và quan trọng là trong **điều kiện sử dụng tự nhiên** — em không ép tư thế hay môi trường. Mỗi người thực hiện sáu phiên, gồm ba hoạt động đi bộ, đứng, ngồi, cùng phần chạm và cuộn. Việc thu **đa thiết bị, tự nhiên** như vậy giúp dữ liệu sát điều kiện triển khai thật hơn.

### Slide 7 — Tiền xử lý dữ liệu `[~1p30s]`
Dữ liệu thô không đưa thẳng vào mô hình được, mà phải qua **tiền xử lý**, gồm ba bước.

**Thứ nhất**, do tần số lấy mẫu dao động và đôi khi mất mẫu, em **kiểm tra tính liên tục** và **cắt tín hiệu tại các khoảng trống**, đồng thời bỏ những đoạn quá ngắn. *(chỉ hình phân đoạn)*

**Thứ hai**, mỗi đoạn liên tục được cắt thành **cửa sổ 200 mẫu — khoảng 4 giây**, trượt với bước 20 nên **chồng lấn cao**, vừa tăng số lượng mẫu vừa cho mô hình thấy hành vi ở nhiều vị trí thời gian. *(chỉ hình cửa sổ trượt)*

**Thứ ba**, mỗi cửa sổ được **chuẩn hóa z-score**. Bước này rất quan trọng với bộ dữ liệu 19 dòng máy: *(chỉ hình trước/sau)* tín hiệu thô của các thiết bị lệch nhau về thang và độ lệch nền, nhưng sau chuẩn hóa thì được **đưa về cùng một dải**, giúp khử khác biệt phần cứng. Kết quả cuối cùng là hơn **163 nghìn cửa sổ** cho cấu hình toàn bộ, và khoảng **55 nghìn** cho cấu hình chỉ đi bộ.

### Slide 8 — Xây dựng mô hình `[~1p]`
Với mô hình, em dùng mạng **CNN một chiều** biến mỗi cửa sổ thành **vector đặc trưng 128 chiều**. Điểm mấu chốt là **hàm quyết định theo hướng tập mở**: khi đăng ký, chủ máy được biểu diễn bằng một tập vector **anchor**; khi xác thực, em tính **độ tương đồng cosine** tới các anchor, **chuẩn hóa theo nhóm người nền** rồi qua sigmoid. Nhờ vậy **đăng ký người mới chỉ cần vài mẫu, không phải huấn luyện lại**. Em cũng có nhánh chạm dùng Random Forest ở vai trò đối chứng.

### Slide 9 — Kết quả: So sánh kiến trúc `[~1p15s]`
Em đánh giá bằng **tỉ lệ lỗi cân bằng EER** trên hai kịch bản: **tập đóng** — người đã biết, và **tập mở** — người hoàn toàn chưa từng thấy. *(chỉ bảng)* Ở tập đóng, CNN 1D đạt EER khoảng **3,4%**. Ở tập mở, khó hơn nhiều, CNN 1D vẫn **tốt nhất**, trong khi hai biến thể LSTM lên tới ~30% — dấu hiệu **quá khớp**. Dùng **cả ba hoạt động** cũng tốt hơn chỉ đi bộ. Vì vậy em chọn **CNN 1D**, vừa nhẹ nhất vừa tổng quát tốt nhất.

### Slide 10 — Kết quả: Hàm chấm điểm `[~1p15s]`
Em còn khảo sát riêng **hàm chấm điểm**, giữ nguyên bộ mã hóa, so bốn cách chấm bằng **kiểm thử chéo sáu vòng** để kết quả đáng tin. *(chỉ bảng)* Hàm **cos_znorm** tốt nhất: EER tập mở khoảng **10,8%**, tập đóng **2,24%**. Em xin nói thẳng, mức **10,8% ở tập mở vẫn còn cao** — vì nhận dạng người hoàn toàn lạ trên số danh tính hạn chế là bài toán khó. Một điểm nữa: **dung hợp thêm nhánh chạm không cải thiện** ở tập mở, nên bản triển khai cuối em chỉ dùng **nhánh quán tính**.

### Slide 11 — Triển khai & sản phẩm `[~1p]`
Cấu hình đã chọn được triển khai **trực tiếp trên thiết bị** qua **TensorFlow Lite**, nên dữ liệu không rời khỏi máy và chạy được cả khi **không có mạng**. Khi vận hành, một dịch vụ chấm điểm liên tục và điều khiển **máy trạng thái ba mức**, với ngưỡng **hiệu chuẩn riêng cho từng chủ máy**. Khi nghi ngờ người lạ, thay vì khóa máy, hệ thống yêu cầu người dùng **lắc điện thoại theo một mã bí mật** để xác thực lại.

### Slide 12 — Giao diện & Demo `[~40s]`
*(chỉ ảnh)* Đây là giao diện thực tế: chọn chế độ, đăng ký sinh trắc, đăng ký cử chỉ lắc, rồi xác thực liên tục. Bên cạnh là **web demo**, hiển thị song song điểm của ba nhánh, giúp thấy rõ và minh bạch rằng dung hợp không tốt hơn nhánh quán tính.

### Slide 13 — Kết luận & Cảm ơn `[~45s]`
Tóm lại, đề tài đã xây dựng **một hệ thống hoàn chỉnh** từ thu thập dữ liệu, tiền xử lý, mô hình, đánh giá đến triển khai trên thiết bị; chốt cấu hình **CNN 1D với hàm cos_znorm**, đạt EER **2,24% ở tập đóng** và **10,8% ở tập mở**. Đề tài vẫn còn **hạn chế**: lỗi tập mở còn cao, nhánh chạm chưa phát huy, và chưa đánh giá trước tấn công chủ động. **Hướng phát triển** là học biểu diễn tốt hơn, mở rộng dữ liệu và hoàn thiện nhánh tương tác.

Phần trình bày của em đến đây là hết. **Em xin cảm ơn các thầy cô đã lắng nghe**, và rất mong nhận được góp ý ạ.

---

## Ghi chú khi thuyết trình
- **Đừng đọc slide** — slide là ý chính, lời nằm ở script này; luyện vài lần cho thuộc mạch.
- Mạch giờ liền: **vấn đề → giải pháp → chức năng → thu thập → tiền xử lý → mô hình → kết quả → triển khai**.
- Nếu **thiếu giờ**: rút gọn slide 5 (use-case) và 12 (demo); **giữ nguyên** slide 7 (tiền xử lý) và 9–10 (kết quả).
- **Câu chốt cần nhớ:** *"CNN 1D + cos_znorm, EER 2,24% tập đóng và 10,8% tập mở; dung hợp touch không cải thiện nên triển khai đơn modal."*
- **Trung thực về hạn chế** (10–11% tập mở) → tạo thiện cảm, tránh bị bắt bài.
