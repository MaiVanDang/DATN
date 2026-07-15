# Script thuyết trình bảo vệ ĐATN

Đề tài: *Phát triển hệ thống nhận diện người dùng qua hành vi sử dụng điện thoại với cơ chế xác thực dự phòng bằng cử chỉ chuyển động* — SV Mai Văn Đăng.

> Văn nói, xưng **"em"**. Tổng ~**13 phút**. `[~Xs]` = thời lượng gợi ý. *(chỉ slide)* = lúc chỉ vào hình/bảng. Cấu trúc: **Tổng quan → Phân tích bài toán → Giải quyết bài toán → Kết quả → Triển khai**.

---

### Slide 1 — Bìa `[~30s]`
Em kính chào các thầy, các cô trong hội đồng. Em là Mai Văn Đăng. Hôm nay em xin phép được trình bày đồ án tốt nghiệp với đề tài **"Phát triển hệ thống nhận diện người dùng qua hành vi sử dụng điện thoại, kèm cơ chế xác thực dự phòng bằng cử chỉ chuyển động"**. Sau đây em xin bắt đầu.

### Slide 2 — Nội dung `[~15s]`
Bài trình bày gồm: **đặt vấn đề và mục tiêu**, **các nghiên cứu liên quan và điểm mới**, **phân tích bài toán**, rồi **tổng quan giải pháp và chức năng**, sau đó **dữ liệu, tiền xử lý và mô hình**, tiếp đến **kết quả đánh giá**, và cuối cùng là **triển khai và kết luận**.

### Slide 3 — Đặt vấn đề & Mục tiêu `[~1p10s]` — *Tổng quan*
Thưa hội đồng, điện thoại ngày nay lưu gần như toàn bộ thông tin cá nhân và tài chính, nhưng lại chủ yếu được bảo vệ bằng **xác thực một lần** lúc mở khóa — như PIN, vân tay hay khuôn mặt. Vấn đề là **sau khi mở khóa, hệ thống không kiểm tra lại danh tính nữa**; nếu máy rơi vào tay người khác lúc đang mở, họ dùng tiếp mà không bị chặn.

Hướng giải quyết là **sinh trắc học hành vi**: thay vì một lần, hệ thống **liên tục** nhận diện người dùng qua cách họ cầm máy, đi lại, chạm màn hình — hoàn toàn ngầm, chỉ dùng cảm biến **có sẵn**.

Vì vậy **mục tiêu** của em là một hệ thống xác thực hành vi **liên tục, chạy trên thiết bị**, **phát hiện người lạ**, và có **dự phòng bằng cử chỉ lắc**. Để thấy rõ điểm mới, em xin điểm qua các nghiên cứu trước.

### Slide 4 — Nghiên cứu liên quan & Điểm mới `[~1p]` — *Tổng quan*
*(chỉ bảng)* Em khảo sát bốn công trình tiêu biểu. **IntelliAuth** khai thác cảm biến quán tính theo ngữ cảnh hoạt động, nhưng dừng ở bài toán **tập đóng**, chưa phát hiện người lạ. **B2auth** làm sinh trắc chạm chạy trên thiết bị rất thực tế, nhưng **chỉ dùng màn hình cảm ứng** nên hỏng khi người dùng không chạm. **MultiLock** xác thực liên tục từ nhiều nguồn, nhưng khi bất thường thì **chỉ khóa màn hình** chứ không xác thực lại. **M2Auth** dung hợp quán tính và chạm, song vẫn là **tập đóng**.

Từ đó em rút ra **ba điểm mới**: thứ nhất là **phát hiện người lạ theo hướng tập mở**; thứ hai là **cơ chế dự phòng bằng cử chỉ lắc thay vì khóa máy**; và thứ ba là **hệ đa phương thức chạy trực tiếp trên thiết bị**. Đây chính là ba khoảng trống mà **chưa công trình nào giải quyết đồng thời**.

### Slide 5 — Phát biểu bài toán & Ràng buộc `[~1p10s]` — *Phân tích bài toán*
Em xin **phát biểu bài toán một cách hình thức**. Bài toán là: **xác thực người dùng liên tục theo hành vi theo hướng tập mở** — cho hồ sơ hành vi của chủ máy, hệ thống phải quyết định **chấp nhận hay từ chối** chuỗi tín hiệu đang quan sát, **kể cả khi nó thuộc một người hoàn toàn chưa từng xuất hiện lúc huấn luyện**.

**Hàm mục tiêu** là **tối thiểu hóa EER** — tỉ lệ lỗi tại ngưỡng mà **FAR bằng FRR**. Trong đó **FAR** là tỉ lệ nhận nhầm người lạ, còn **FRR** là tỉ lệ từ chối nhầm chủ máy. Hai lỗi này **đánh đổi lẫn nhau theo ngưỡng** — siết ngưỡng thì FAR giảm nhưng FRR tăng — nên **không thể tối thiểu đồng thời**; vì vậy em gộp lại thành **một chỉ số duy nhất là EER** để tối ưu và so sánh giữa các cấu hình.

*(chỉ phần ràng buộc)* Bài toán chịu **tám ràng buộc**, quan trọng nhất là: danh tính lúc kiểm thử **có thể không thuộc tập huấn luyện**; đăng ký người dùng mới **không được huấn luyện lại** mô hình; toàn bộ suy luận **chạy trên thiết bị**; **ngưỡng hiệu chuẩn riêng** cho từng chủ máy; quyết định dựa trên **điểm đã làm mịn** chứ không phải một cửa sổ đơn; và khi từ chối phải có **đường khôi phục** chứ không khóa cứng máy.

Em xin lưu ý: **khác với bài toán tối ưu tổ hợp**, đây là bài toán **quyết định thống kê** nên không có nghiệm tối ưu chứng minh được — chất lượng được đo bằng **thực nghiệm**.

### Slide 6 — Tham số đầu vào & Mong muốn đầu ra `[~50s]` — *Phân tích bài toán*
*(chỉ hai cột)* **Về đầu vào:** không gian tín hiệu là các **cửa sổ 200 mẫu trên 9 kênh**; tập huấn luyện gồm **21 danh tính**; chủ máy được đăng ký bằng một **tập nhỏ vài mẫu**; kèm theo là **nhóm nền cohort** — tập embedding của người khác được **đóng gói sẵn trong ứng dụng**; cùng các tham số như **số chiều embedding 128**, hệ số làm mịn, và khoảng kẹp ngưỡng.

**Về đầu ra mong muốn:** lời giải là **bộ ba** gồm **bộ mã hóa** ánh xạ cửa sổ sang vector 128 chiều, **hàm chấm điểm tin cậy** trong khoảng 0 đến 1, và **ngưỡng riêng của chủ máy**. Từ đó **quy tắc quyết định** là so **điểm đã làm mịn** với ngưỡng để chấp nhận hay từ chối.

Phát biểu này sinh ra **bốn câu hỏi nghiên cứu**: kiến trúc nào cho bộ mã hóa, hàm chấm điểm nào, dung hợp thêm nhánh chạm có ích không, và triển khai cùng cơ chế phản ứng ra sao. Các phần sau em lần lượt trả lời.

### Slide 7 — Tổng quan giải pháp `[~45s]` — *Giải quyết*
*(chỉ sơ đồ)* Đây là kiến trúc tổng thể, gồm **bốn giai đoạn**: thu thập dữ liệu, **tiền xử lý**, huấn luyện và hình thành hàm quyết định, rồi triển khai xuống thiết bị. Nhánh chính là **cảm biến quán tính**, còn nhánh **chạm và cuộn** em giữ ở vai trò đối chứng.

### Slide 8 — Chức năng hệ thống `[~40s]`
*(chỉ 2 sơ đồ use-case)* Hệ thống được hiện thực thành **hai ứng dụng**. Ứng dụng thứ nhất dành cho **người tham gia** để **thu thập dữ liệu**. Ứng dụng thứ hai dành cho **chủ máy** để **xác thực** — chọn chế độ, đăng ký sinh trắc, đăng ký cử chỉ lắc, rồi xác thực liên tục; còn **người lạ** là tác nhân bị hệ thống phát hiện và từ chối.

### Slide 9 — Thu thập dữ liệu `[~55s]`
Trước hết em xây dựng một **ứng dụng thu thập** chuyên dụng để lấy dữ liệu nhất quán trên nhiều người và nhiều máy. Bộ dữ liệu gồm **26 người tham gia**, thu trên **19 dòng thiết bị**, trong **điều kiện sử dụng tự nhiên** — em không ép tư thế hay môi trường. Mỗi người thực hiện sáu phiên, gồm ba hoạt động đi bộ, đứng, ngồi, cùng phần chạm và cuộn.

### Slide 10 — Tiền xử lý dữ liệu `[~1p]` — *giải ràng buộc bất biến thiết bị*
Dữ liệu thô phải qua **tiền xử lý**. Trước hết em **cắt tín hiệu tại các khoảng trống** và bỏ đoạn quá ngắn để bảo đảm tính liên tục. Sau đó cắt thành **cửa sổ 200 mẫu, khoảng 4 giây**, để đầu vào có kích thước cố định. Bước quan trọng nhất là **chuẩn hóa z-score từng cửa sổ**: *(chỉ hình trước/sau)* tín hiệu của 19 dòng máy vốn lệch nhau về thang và độ lệch nền, sau chuẩn hóa được đưa về cùng một dải, **khử khác biệt phần cứng** — đây chính là lời giải cho **ràng buộc bất biến thiết bị**. Kết quả là hơn **163 nghìn cửa sổ** cho cấu hình toàn bộ.

### Slide 11 — Xây dựng mô hình `[~1p]` — *trả lời câu hỏi ① + ②*
Với **bộ mã hóa**, em dùng mạng **CNN một chiều** biến mỗi cửa sổ thành **vector đặc trưng 128 chiều**. Với **hàm chấm điểm**, em thiết kế theo hướng **tập mở**: khi đăng ký, chủ máy được biểu diễn bằng một tập vector **anchor**; khi xác thực, em tính **độ tương đồng cosine** tới các anchor, **chuẩn hóa theo nhóm người nền** rồi qua sigmoid. Nhờ vậy **đăng ký người mới chỉ cần vài mẫu, không phải huấn luyện lại** — đúng **ràng buộc số 2**. Em cũng có nhánh chạm dùng Random Forest ở vai trò đối chứng.

### Slide 12 — Kết quả: So sánh kiến trúc `[~1p10s]` — *kết quả câu hỏi ①*
Em đánh giá bằng bốn chỉ số: **FAR** và **FRR** đánh đổi theo ngưỡng nên tổng hợp bằng **EER**; ngoài ra dùng **AUC** đo khả năng phân biệt tổng thể. *(chỉ bảng)* Ở **tập đóng**, CNN 1D đạt EER khoảng **3,4%** và AUC tới **0,99**. Ở **tập mở**, khó hơn nhiều, CNN 1D vẫn **tốt nhất** — AUC **0,85**, cao hơn hẳn hai biến thể LSTM vốn EER quanh **30%** và có dấu hiệu **quá khớp**. Dùng **cả ba hoạt động** cũng tốt hơn chỉ đi bộ. Vì vậy em chọn **CNN 1D**, vừa nhẹ nhất vừa tổng quát tốt nhất.

### Slide 13 — Kết quả: Hàm chấm điểm `[~1p10s]` — *kết quả câu hỏi ② + ③*
Em khảo sát riêng **hàm chấm điểm**, giữ nguyên bộ mã hóa, so bốn cách chấm bằng **kiểm thử chéo sáu vòng**. *(chỉ bảng)* Hàm **cos_znorm** tốt nhất: EER tập mở khoảng **10,8%**, tập đóng **2,24%**. Em xin nói thẳng, mức **10,8% ở tập mở vẫn còn cao** — vì nhận dạng người hoàn toàn lạ trên số danh tính hạn chế là bài toán khó. Trả lời cho **câu hỏi đa phương thức**: **dung hợp thêm nhánh chạm không cải thiện** ở tập mở, nên bản triển khai cuối em chỉ dùng **nhánh quán tính**.

### Slide 14 — Triển khai & sản phẩm `[~1p]` — *trả lời câu hỏi ④*
Cấu hình đã chọn được triển khai **trực tiếp trên thiết bị** qua **TensorFlow Lite**, nên dữ liệu không rời khỏi máy và chạy được cả khi **không có mạng** — thỏa **ràng buộc on-device và tài nguyên**. Khi vận hành, một dịch vụ chấm điểm liên tục và điều khiển **máy trạng thái ba mức**, với **ngưỡng hiệu chuẩn riêng từng chủ máy** và **điểm làm mịn EWMA**. Khi nghi ngờ người lạ, thay vì khóa máy, hệ thống yêu cầu **lắc điện thoại theo một mã bí mật** để xác thực lại — đúng **ràng buộc phải có đường khôi phục**.

### Slide 15 — Giao diện & Demo `[~40s]`
*(chỉ ảnh)* Đây là giao diện thực tế: chọn chế độ, đăng ký sinh trắc, đăng ký cử chỉ lắc, rồi xác thực liên tục. Bên cạnh là **web demo**, hiển thị song song điểm của ba nhánh, giúp thấy rõ và minh bạch rằng dung hợp không tốt hơn nhánh quán tính.

### Slide 16 — Kết luận & Cảm ơn `[~45s]`
Tóm lại, đề tài đã xây dựng **một hệ thống hoàn chỉnh** từ thu thập dữ liệu, tiền xử lý, mô hình, đánh giá đến triển khai trên thiết bị; chốt cấu hình **CNN 1D với hàm cos_znorm**, đạt EER **2,24% ở tập đóng** và **10,8% ở tập mở**. Ba điểm mới là **phát hiện người lạ theo hướng tập mở**, **dự phòng bằng cử chỉ lắc**, và **hệ đa phương thức chạy trên thiết bị**. Đề tài vẫn còn **hạn chế**: lỗi tập mở còn cao, nhánh chạm chưa phát huy, và chưa đánh giá trước tấn công chủ động. **Hướng phát triển** là học biểu diễn tốt hơn, mở rộng dữ liệu và hoàn thiện nhánh tương tác.

Phần trình bày của em đến đây là hết. **Em xin cảm ơn các thầy cô đã lắng nghe**, và rất mong nhận được góp ý ạ.

---

## Ghi chú khi thuyết trình
- **Đừng đọc slide** — slide là ý chính, lời nằm ở script này.
- **Mạch bám mẫu:** Slide 3–4 (Tổng quan) → **Slide 5–6 (Phân tích bài toán — phát biểu hình thức)** → Slide 7–14 (Giải quyết + kết quả) → Slide 15 (demo).
- **Nối ràng buộc ↔ lời giải:** Slide 10→bất biến thiết bị · 11→không train lại · 14→on-device, ngưỡng riêng, đường khôi phục. Nhắc lại để hội đồng thấy **mạch khép kín**.
- ⚠️ **Đừng gọi bài toán là NP-hard / tối ưu tổ hợp** — nếu bị hỏi, trả lời: *"đây là bài toán quyết định thống kê, các ràng buộc là yêu cầu thiết kế chứ không phải điều kiện khả thi toán học."*
- Nếu **thiếu giờ**: rút gọn slide 8 (use-case) và 15 (demo); **giữ nguyên** slide 5–6 (phân tích bài toán) và 12–13 (kết quả).
- **Câu chốt cần nhớ:** *"CNN 1D + cos_znorm, EER 2,24% tập đóng và 10,8% tập mở; dung hợp touch không cải thiện nên triển khai đơn modal."*
- **Trung thực về hạn chế** (10–11% tập mở) → tạo thiện cảm, tránh bị bắt bài.
