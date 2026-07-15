# Kịch bản nội dung slide bảo vệ ĐATN

Đề tài: **Phát triển hệ thống nhận diện người dùng qua hành vi sử dụng điện thoại với cơ chế xác thực dự phòng bằng cử chỉ chuyển động** — SV Mai Văn Đăng.

> Bố cục 16 slide, bám mẫu **Tổng quan → Phân tích bài toán → Giải quyết bài toán → Kết quả → Triển khai → Kết luận**. Mỗi slide gồm **Tiêu đề** + **gạch đầu dòng ngắn** + **📊/🖼️ chỗ cần chèn bảng/ảnh**.

---

## Slide 1 — Trang bìa
- **Tên đề tài** (đầy đủ)
- Sinh viên: **Mai Văn Đăng** — MSSV 20225699
- GVHD: *(điền)* · Trường/Chương trình: *(điền)* · Hà Nội, *(tháng/năm)*

## Slide 2 — Nội dung trình bày
- Đặt vấn đề & mục tiêu
- Nghiên cứu liên quan & điểm mới
- **Phân tích bài toán** (phát biểu hình thức)
- Tổng quan giải pháp & chức năng
- Xây dựng dữ liệu, tiền xử lý & mô hình
- Kết quả đánh giá
- Triển khai & kết luận

## Slide 3 — Đặt vấn đề & Mục tiêu  *(Tổng quan)*
- Xác thực hiện nay chỉ **một lần** lúc mở khóa → mở khóa xong **không kiểm soát lại** danh tính
- **Sinh trắc học hành vi:** xác thực **liên tục, ngầm**, dùng cảm biến có sẵn
- Khoảng trống: thiếu **phát hiện người lạ (tập mở)** + thiếu **cơ chế phản ứng** khi bất thường
- **Mục tiêu:** xác thực hành vi liên tục **on-device** + phát hiện người lạ + **dự phòng cử chỉ lắc**

## Slide 4 — Nghiên cứu liên quan & Điểm mới  *(Tổng quan)*
- **IntelliAuth** — quán tính + ngữ cảnh hoạt động, nhưng **tập đóng**, chưa phát hiện người lạ
- **B2auth** — sinh trắc chạm **on-device** thực tế, nhưng **chỉ touch** → hỏng khi không chạm màn hình
- **MultiLock** — xác thực liên tục đa nguồn, nhưng bất thường thì **chỉ khóa màn hình**, không xác thực lại
- **M2Auth** — dung hợp quán tính + chạm, nhưng vẫn **tập đóng**, không phát hiện người lạ
- ⇒ **3 điểm mới của đề tài:** ① **phát hiện người lạ (tập mở)** · ② **dự phòng cử chỉ lắc** (thay vì khóa) · ③ **đa phương thức chạy on-device**
- 📊 **Chèn Bảng 2.1** (so sánh 6 tiêu chí × 4 công trình + Đề tài)

## Slide 5 — Phát biểu bài toán & Ràng buộc  ⭐

**1. Bài toán**
- **Xác thực người dùng liên tục theo hành vi — nhận dạng tập mở:** cho hồ sơ hành vi của chủ máy, quyết định **chấp nhận / từ chối** chuỗi tín hiệu đang quan sát, **kể cả khi thuộc người chưa từng thấy lúc huấn luyện**

**Với mục tiêu:**
- Tối thiểu hóa **EER** — tỉ lệ lỗi tại ngưỡng **cân bằng FAR = FRR**
- **FAR** = nhận nhầm người lạ *(an ninh)* · **FRR** = từ chối nhầm chủ máy *(trải nghiệm)*
- Hai lỗi **đánh đổi** theo ngưỡng → **không tối thiểu đồng thời được** → gộp thành **một chỉ số EER**

**2. Các ràng buộc**
1. **Tập mở:** danh tính lúc kiểm thử có thể **∉ tập huấn luyện**
2. **Không huấn luyện lại** mô hình khi đăng ký chủ máy mới
3. Suy luận **hoàn toàn trên thiết bị**, dữ liệu không rời máy
4. **Ngưỡng hiệu chuẩn riêng** từng chủ máy, kẹp trong $[0{,}10;\,0{,}45]$
5. Quyết định dựa trên **điểm đã làm mịn (EWMA)**, không phải một cửa sổ đơn
6. Khi từ chối phải có **đường khôi phục** (cử chỉ lắc), **không khóa cứng**
7. **Bất biến thiết bị** — hoạt động trên 19 dòng máy khác nhau
8. **Tài nguyên** đủ nhỏ (dung lượng, độ trễ) để chạy **liên tục**

> *Nói:* ⚠️ khác bài toán tối ưu tổ hợp — đây là **quyết định thống kê**, không có nghiệm tối ưu chứng minh được; chất lượng đo bằng **thực nghiệm** (FAR/FRR/EER).

## Slide 6 — Tham số đầu vào & Mong muốn đầu ra  ⭐
*(trình bày 2 cột như mẫu)*

**Tham số đầu vào**
- $\mathcal{X}\subset\mathbb{R}^{200\times 9}$: không gian cửa sổ tín hiệu (200 mẫu × 9 kênh)
- $\mathcal{U}=\{u_1,\dots,u_N\}$, $N=21$: tập danh tính dùng huấn luyện
- $\mathcal{D}=\{(x_i,y_i)\}$: dữ liệu huấn luyện, $x_i\in\mathcal{X},\ y_i\in\mathcal{U}$
- $\mathcal{E}_o=\{x_1^o,\dots,x_K^o\}$: **mẫu đăng ký** của chủ máy $o$ ($K$ nhỏ)
- $\mathcal{C}\subset\mathbb{R}^{d}$: **nhóm nền (cohort)** — embedding người khác, đóng gói sẵn trong app
- $d=128$: số chiều embedding · $\alpha$: hệ số làm mịn EWMA
- $[\theta_{\min},\theta_{\max}]=[0{,}10;\,0{,}45]$: khoảng kẹp ngưỡng

**Mong muốn đầu ra**
$$\text{Solution} = \big(f_\phi,\; s,\; \theta_o\big)$$
- $f_\phi:\mathcal{X}\to\mathbb{R}^{128}$ — **bộ mã hóa** (embedding)
- $s:\mathcal{X}\times\mathcal{E}_o\times\mathcal{C}\to[0,1]$ — **hàm chấm điểm tin cậy**
- $\theta_o\in[\theta_{\min},\theta_{\max}]$ — **ngưỡng riêng** chủ máy $o$

$$g(x)=\begin{cases}\text{chấp nhận} & \hat{s}(x)\ge\theta_o\\ \text{từ chối} & \hat{s}(x)<\theta_o\end{cases}\qquad \hat{s}_t=\alpha s_t+(1-\alpha)\hat{s}_{t-1}$$

> *Nói:* từ phát biểu này sinh ra **4 câu hỏi nghiên cứu**: ① kiến trúc nào cho $f_\phi$? ② hàm chấm điểm $s$ nào? ③ dung hợp touch có ích? ④ triển khai & phản ứng ra sao? — các slide sau lần lượt trả lời.

## Slide 7 — Tổng quan giải pháp  *(Giải quyết)*
- Pipeline **4 giai đoạn:** Thu thập → **Tiền xử lý** → Huấn luyện & hàm quyết định → Triển khai on-device
- Nhánh chính: **cảm biến quán tính**; nhánh **chạm/cuộn**: đối chứng
- 🖼️ **Chèn Hình:** sơ đồ kiến trúc tổng quan — `Hinh_ve/kientruc_tongquan.png`

## Slide 8 — Chức năng hệ thống (use-case)
- **Người tham gia** → thu thập dữ liệu (đăng ký, thu cảm biến, điền biểu mẫu, xuất dữ liệu)
- **Chủ sở hữu** → xác thực (chọn chế độ, đăng ký sinh trắc, đăng ký cử chỉ lắc, xác thực liên tục, dự phòng)
- **Người lạ** → bị nhánh xác thực phát hiện & từ chối
- 🖼️ **Chèn 2 ảnh cạnh nhau:** `Hinh_ve/UC_BioAuth_Data_Collection.png` + `Hinh_ve/UC_BioAuth_Authenticator.png`

## Slide 9 — Thu thập dữ liệu
- Tự xây dựng **app thu thập** chuyên dụng: đăng ký → thu cảm biến → biểu mẫu tương tác → xuất dữ liệu
- Bộ dữ liệu **thực tế**: **26 người**, **19 dòng thiết bị**, điều kiện sử dụng **tự nhiên**
- Mỗi người **6 phiên**: 3 hoạt động (đi bộ, đứng, ngồi) + thao tác chạm/cuộn
- 🖼️ *(tùy chọn)* ảnh giao diện app thu thập — `Hinh_ve/Collectdata/screen_collect.jpg`

## Slide 10 — Tiền xử lý dữ liệu  ⭐  *(giải ràng buộc 7)*
- **Phân đoạn:** cắt tín hiệu tại khoảng trống > ngưỡng, bỏ đoạn quá ngắn
- **Cửa sổ trượt 200 mẫu (~4 giây)** → đầu vào kích thước cố định
- **Chuẩn hóa z-score** từng cửa sổ → **khử khác biệt phần cứng** giữa 19 dòng máy *(điểm mấu chốt)*
- Kết quả: **163.145 cửa sổ** (toàn bộ) / **54.823** (đi bộ); touch → **34.836** sự kiện chạm
- 🖼️ **Chèn hình:** phân đoạn (`Preprocess/phan_doan_a.png`) + cửa sổ trượt (`Preprocess/cua_so_truot.png`) + z-score trước/sau (`Preprocess/2a_zscore_truoc.png` + `2b_zscore_sau.png`)

## Slide 11 — Xây dựng mô hình  *(trả lời câu hỏi ① + ②)*
- **CNN 1D** mã hóa cửa sổ 200×9 → **embedding 128 chiều** — lời giải cho $f_\phi$
- Nhận dạng tập mở: **anchor → cosine → z-norm cohort → sigmoid** (hàm `cos_znorm`) — lời giải cho $s$
- Nhánh touch **33 chiều** (Random Forest) + dung hợp — vai trò **đối chứng**
- 📊 **Chèn Bảng 4.1:** số tham số / dung lượng 3 kiến trúc

> *Nói:* backbone chỉ để học không gian embedding; xác thực bằng đối sánh anchor nên **thêm người mới không cần train lại** (thỏa ràng buộc 2).

## Slide 12 — Kết quả: So sánh kiến trúc  *(kết quả câu hỏi ①)*
- **FAR** và **FRR** đánh đổi theo ngưỡng → tổng hợp bằng **EER**; kèm **AUC** (càng gần 1 càng tốt)
- Tập đóng: **CNN 1D** — EER ≈ **3,4%**, AUC **0,99** (nhóm tốt nhất)
- Tập mở (khó hơn): **CNN 1D tốt nhất** — AUC **0,85**, vượt CNN-LSTM/BiLSTM (EER ~30%, AUC ~0,82; quá khớp)
- Toàn bộ 3 hoạt động **> chỉ đi bộ** → chọn **CNN 1D**
- 📊 **Chèn Bảng 5.1 + 5.3** (đủ 4 cột AUC / EER / FAR / FRR)

## Slide 13 — Kết quả: Hàm chấm điểm  *(kết quả câu hỏi ② + ③)*
- So **4 hàm** (cos_mean, cos_knn, **cos_znorm**, maha) — **leave-users-out 6 vòng**
- **cos_znorm tốt nhất:** EER tập mở **10,83% ± 6,51%**; tập đóng **2,24%**
- Dung hợp nhánh chạm **không cải thiện** tập mở → triển khai **đơn modal quán tính**
- 📊 **Chèn Bảng 5.5**

## Slide 14 — Triển khai & sản phẩm  *(trả lời câu hỏi ④)*
- Suy luận **trực tiếp trên thiết bị** qua **TensorFlow Lite** → thỏa ràng buộc 3, 8
- Xác thực liên tục: **máy trạng thái 3 mức**, làm mịn **EWMA**, **ngưỡng riêng từng chủ máy** → ràng buộc 4, 5
- Dự phòng khi nghi ngờ người lạ: **cử chỉ lắc** (mã 4–8 chữ số), không khóa cứng → ràng buộc 6
- Sản phẩm: app thu thập, app xác thực, **web demo**

## Slide 15 — Giao diện & Demo
- **App xác thực:** chọn chế độ → đăng ký sinh trắc → đăng ký cử chỉ lắc → xác thực liên tục
- **Web demo:** hiển thị điểm quán tính / touch / fusion + 2 kết luận để đối chiếu minh bạch
- 🖼️ **Chèn ảnh:** `Hinh_ve/Demo_app/*` + web demo

## Slide 16 — Kết luận & Cảm ơn
- Hệ thống **hoàn chỉnh**: dữ liệu → tiền xử lý → mô hình → đánh giá → triển khai on-device
- Chốt: **CNN 1D + cos_znorm**; EER **đóng 2,24%** / **mở 10,83%**
- **Điểm mới:** phát hiện người lạ (tập mở) · dự phòng cử chỉ lắc · đa phương thức on-device
- Hạn chế: tập mở còn ~10–11%; touch chưa hiệu quả; chưa đánh giá tấn công chủ động
- Hướng phát triển: học biểu diễn (ArcFace), mở rộng dữ liệu, hoàn thiện touch
- **XIN CẢM ƠN THẦY CÔ!**

---

## Gợi ý dựng slide
- **Slide 5 & 6 là phần "Phân tích bài toán"** theo mẫu hình thức: Slide 5 = *Bài toán + Mục tiêu + Ràng buộc*, Slide 6 = *Đầu vào | Đầu ra* (dựng **2 cột**).
- Ràng buộc ở Slide 5 được **nhắc lại khi giải** (Slide 10→RB7, 11→RB2, 14→RB3,4,5,6,8) để hội đồng thấy **mạch khép kín**.
- **Ít chữ, nhiều ý chính**; con số bôi đậm (2,24% / 10,83% / 163.145…) → cỡ lớn/màu nổi.
- ⚠️ **Đừng gọi bài toán là NP-hard / tối ưu tổ hợp** — đây là bài toán **quyết định thống kê**.
- Bảng Slide 4 lấy từ `Chuong/2_Co_so_ly_thuyet.tex` (`tab:related_work_compare`); ảnh/bảng khác trong `docs/report/DATN/Hinh_ve/` và `Chuong/5_Danh_gia_thuc_nghiem.tex`.
