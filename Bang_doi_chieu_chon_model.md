# Phiếu dẫn chứng — Căn cứ lựa chọn mô hình

> **Dùng khi vấn đáp.** Mục đích: chứng minh việc chọn **CNN 1D** dựa trên **so sánh có hệ thống**, không phải chọn ngẫu nhiên.
> ⚠️ **Đọc kỹ mục 5 (điểm phải thừa nhận) trước khi dùng** — có chỗ CNN không thắng, cần chủ động nói ra.

---

## 1. Dẫn chứng A — So 3 kiến trúc học sâu *(nguồn: Bảng 4.1, 5.1, 5.3 trong báo cáo)*

Cùng dữ liệu, cùng giao thức, **trung bình 5 lần chạy**, nhánh quán tính, cấu hình `all`:

| Kiến trúc | Số tham số | Dung lượng | Tập đóng EER | **Tập mở EER** | **Tập mở AUC** |
|---|---:|---:|---:|---:|---:|
| **CNN 1D** ✅ | **80,0 nghìn** | **323 KB** | 3,35% | **21,00%** | **0,850** |
| CNN-LSTM | 162,6 nghìn | 643 KB | 2,88% | 30,33% | 0,832 |
| CNN-BiLSTM | 129,8 nghìn | 516 KB | 4,81% | 29,00% | 0,824 |

**Đọc bảng:**
- CNN-LSTM tốt hơn chút ở **tập đóng** (2,88 vs 3,35) nhưng **sụp ở tập mở** (30,33 vs 21,00) → **dấu hiệu quá khớp**.
- CNN 1D **nhẹ nhất** (½ tham số của CNN-LSTM) mà **tổng quát tốt nhất** ở đúng kịch bản mục tiêu.

---

## 2. Dẫn chứng B — So với hướng truyền thống *(chạy mới, đặc trưng thủ công + ML cổ điển)*

Cùng dữ liệu `processed/`, cùng giao thức (session-disjoint, held-out `user22–26`, gộp phiên EWMA), **metric macro per-owner**, 5 seed:

### Cấu hình `all` (bản triển khai)
| Mô hình | Tập đóng EER | Tập đóng AUC | **Tập mở EER** | **Tập mở AUC** |
|---|---:|---:|---:|---:|
| **CNN 1D** | **1,46% ± 1,95** | **1,00** | 20,00% ± 13,54 | 0,86 |
| Thủ công + RF ⚠️ | 5,29% ± 8,61 | 0,974 | **18,83% ± 15,96** | **0,887** |
| Thủ công + SVM | 7,66% ± 11,09 | 0,959 | 21,50% ± 17,64 | 0,833 |
| Thủ công + kNN | 10,43% ± 13,03 | 0,934 | 28,50% ± 21,43 | 0,715 |

### Cấu hình `walking`
| Mô hình | Tập đóng EER | **Tập mở EER** |
|---|---:|---:|
| **CNN 1D** | **0,78% ± 1,25** | 21,67% ± 14,53 |
| Thủ công + RF ⚠️ | 5,60% ± 8,72 | **10,33% ± 14,88** |
| Thủ công + SVM | 9,22% ± 13,14 | 10,83% ± 13,94 |
| Thủ công + kNN | 10,48% ± 14,69 | 11,50% ± 12,53 |

*(Bộ phân loại trong các bài tham chiếu: IntelliAuth dùng DT/kNN/BN/SVM. **RF là baseline bổ sung do em tự thêm**, không thuộc công trình nào.)*

---

## 3. Dẫn chứng C — Độ bền khi đổi ngữ cảnh *(tập mở, walking → all)*

| Mô hình | walking | all | Thay đổi |
|---|---:|---:|---|
| **CNN 1D** | 21,67% | **20,00%** | ✅ **ổn định** |
| Thủ công + RF | 10,33% | 18,83% | 🔴 xấu đi **82%** |
| Thủ công + SVM | 10,83% | 21,50% | 🔴 xấu đi **99%** |
| Thủ công + kNN | 11,50% | 28,50% | 🔴 xấu đi **148%** |

→ **Đặc trưng thủ công sụp đổ khi trộn 3 hoạt động; CNN thì không.** Đây là lập luận mạnh nhất cho cấu hình triển khai thật (`all`).

---

## 4. Bốn căn cứ chọn CNN 1D

| # | Căn cứ | Bằng chứng |
|---|---|---|
| 1 | **Tổng quát tốt nhất ở tập mở** trong nhóm học sâu | 21,00% vs 30,33% / 29,00% (Dẫn chứng A) |
| 2 | **Nhẹ nhất** — hợp ràng buộc on-device | 80,0K tham số / 323 KB, bằng ½ CNN-LSTM |
| 3 | **Biểu diễn vượt trội ở tập đóng** so với thủ công | EER 1,46% vs 5,29% (~3,6 lần), AUC 1,00, ổn định ±1,95 vs ±8,61 |
| 4 | **Bền khi trộn hoạt động** | Giữ ~20% trong khi baseline xấu đi 82–148% (Dẫn chứng C) |

---

## 5. ⚠️ Ba điểm PHẢI chủ động thừa nhận

**1. Ở tập mở, CNN KHÔNG thắng baseline thủ công.**
- `all`: CNN 20,00% vs RF **18,83%** → **hòa** (chênh 1,2 điểm, nằm trong nhiễu).
- `walking`: CNN 21,67% vs RF **10,33%** → **CNN thua rõ**.
→ **Nói trước, đừng đợi bị hỏi.**

**2. Không khác biệt nào ở tập mở đạt ý nghĩa thống kê.**
Chỉ **5 owner** (`user22–26`), độ lệch **±13–16%**. Với n=5, sai số chuẩn ~6–7 → **không kết luận được ai thắng**, kể cả CNN.

**3. So sánh bị nhiễu biến.**
Baseline **huấn luyện RF phân biệt riêng từng chủ máy** lúc đăng ký; CNN chỉ **cosine tới anchor**, không học gì lúc đăng ký. → Chưa tách được CNN thua vì **biểu diễn** hay vì **cách chấm điểm**.

---

## 6. Lưu ý kỹ thuật (phòng bị soi)

**Vì sao EER tập đóng của CNN lúc là 3,35% lúc là 1,46%?**
Hai **metric khác nhau**:
- **3,35%** = **gộp toàn cục** (dồn điểm mọi chủ máy vào một rổ, một ngưỡng chung) — dùng trong Bảng 5.1.
- **1,46%** = **macro per-owner** (tính EER từng người rồi lấy trung bình) — dùng khi so với baseline.

→ **Trong mỗi bảng chỉ dùng MỘT loại metric.** Bảng mục 1 toàn bộ là gộp; bảng mục 2 toàn bộ là macro. **Không trộn hai bảng với nhau.**

*(Ở tập mở hai metric gần trùng — 21,00% gộp vs 20,00% macro — vì chỉ có 5 owner.)*

---

## 7. Câu trả lời mẫu

**❓ "Sao em chọn CNN 1D?"**
> *"Em không chọn cảm tính mà so sánh có hệ thống trên ba trục. Một, trong ba kiến trúc học sâu, CNN 1D tổng quát tốt nhất ở tập mở — EER 21% so với 29–30% của hai biến thể LSTM, dù LSTM nhỉnh hơn chút ở tập đóng, cho thấy chúng quá khớp. Hai, CNN nhẹ nhất với 80 nghìn tham số, bằng một nửa CNN-LSTM, hợp ràng buộc chạy trên thiết bị. Ba, em còn dựng baseline theo hướng truyền thống — đặc trưng thủ công cộng ML cổ điển — chạy trên cùng dữ liệu và cùng giao thức."*

**❓ "Em có thử phương pháp truyền thống không?"**
> *"Có ạ. Em cài baseline đặc trưng thủ công 216 chiều — gồm nhịp, phổ tần số, tương quan chéo kênh — rồi phân loại bằng SVM, kNN và Random Forest, chạy đúng giao thức của em. Ở tập đóng CNN vượt rõ: 1,46% so với 5,29%. **Nhưng em xin nói thẳng: ở tập mở thì CNN chỉ ngang Random Forest — 20% so với 18,8% — và ở cấu hình chỉ đi bộ thì CNN còn thua.** Với chỉ 5 người ở tập mở và độ lệch trên 13%, khác biệt này chưa có ý nghĩa thống kê."*

**❓ "Vậy sao vẫn chọn CNN?"**
> *"Vì khi hiệu năng tập mở ngang nhau, em cân bằng ba yếu tố khác. Thứ nhất, CNN bền khi trộn ba hoạt động — giữ khoảng 20% trong khi baseline thủ công xấu đi từ 10% lên 19–28%; mà cấu hình triển khai của em chính là cả ba hoạt động. Thứ hai, về triển khai, trích đặc trưng nằm gọn trong mô hình TFLite nên không phải port hàng trăm dòng công thức FFT sang Kotlin và không rủi ro lệch số. Thứ ba, CNN ổn định hơn nhiều giữa các chủ máy — độ lệch 1,95% so với 8,61%."*

**❓ "Random Forest tốt hơn sao không dùng RF?"**
> *"Random Forest không nằm trong công trình tham chiếu nào — đó là baseline em tự thêm. Và nó chỉ tốt hơn ở cấu hình chỉ đi bộ, còn ở cấu hình triển khai là cả ba hoạt động thì hai bên ngang nhau. Ngoài ra baseline đó được lợi thế là huấn luyện một bộ phân loại riêng cho từng chủ máy lúc đăng ký, trong khi CNN chỉ đối sánh cosine. Em ghi nhận đây là hạn chế và là hướng cần khảo sát thêm."*

**❓ "Em có kiểm định thống kê không?"**
> *"Chưa ạ. Với 5 người ở tập mở, em chỉ so trung bình và độ lệch chuẩn, chưa làm kiểm định ý nghĩa. Đây là hạn chế em thừa nhận — cần mở rộng số danh tính mới kết luận chắc chắn được."*

---

## 8. Nguồn số liệu

| Bảng | Nguồn |
|---|---|
| Dẫn chứng A | `docs/report/DATN/Chuong/4_Xay_dung_mo_hinh.tex` (Bảng 4.1) · `5_Danh_gia_thuc_nghiem.tex` (Bảng 5.1, 5.3) |
| Dẫn chứng B — CNN | `ml_pipeline/artifacts/cnn/results_{walking,all}/{closedset,openset}_per_owner.csv` (macro) |
| Dẫn chứng B — baseline | `ml_pipeline/baseline_handcrafted.py` → `baseline_handcrafted_results.json` |
| Dẫn chứng C | Suy từ Dẫn chứng B |
