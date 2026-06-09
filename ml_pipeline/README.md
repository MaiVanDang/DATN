# ML Pipeline — Xác thực sinh trắc học hành vi trên Android

Pipeline machine learning cho đề tài xác thực người dùng liên tục dựa trên hành vi
(cảm biến quán tính + tương tác màn hình). Từ dữ liệu thô thu trên thiết bị Android,
pipeline thực hiện kiểm tra chất lượng, tiền xử lý, huấn luyện backbone học không gian
embedding, và xuất mô hình sang TensorFlow Lite để triển khai trên thiết bị.

## Cấu trúc thư mục

```
ml_pipeline/
├── data/                       Dữ liệu THÔ:  user<N>/session_<M>/*.csv
│   ├── walking_att*.csv        Tín hiệu quán tính 9 kênh (acc/gyro/mag) khi đi bộ
│   ├── sitting_att*.csv        … khi ngồi
│   ├── standing_att*.csv       … khi đứng
│   ├── tap_r*.csv              Sự kiện chạm
│   ├── scroll_r*.csv           Sự kiện cuộn
│   └── keystroke_r*.csv        Sự kiện gõ phím
│
├── processed/                  Dữ liệu SAU tiền xử lý:  user<N>/
│   ├── X_inertial.npy / y_*    Cửa sổ (N, 200, 9) cấu hình ALL (3 hoạt động)
│   ├── X_walking.npy  / y_*    Cửa sổ cấu hình WALKING (chỉ đi bộ)
│   └── *_gestures.csv, touch_session_features.csv   Đặc trưng touch 48-D
│
├── emb_out/                    Embedding xuất ra (.npz) để phân tích hàm chấm điểm
├── plots/                      Biểu đồ minh họa cho báo cáo
├── artifacts/                  Checkpoint mô hình (cnn / convlstm / convlstm_bi)
│
├── step1_quality_check.py      Kiểm tra chất lượng & phát hiện phiên bất thường
├── step2_preprocess.py         Tiền xử lý quán tính + trích đặc trưng touch
├── touch_features.py           Định nghĩa & dựng vector đặc trưng touch (48-D)
├── backbone_train.py           Huấn luyện backbone học embedding (chạy trên Colab)
├── export_embeddings.py        Xuất embedding 3 backbone × 2 ngữ cảnh (chạy trên Colab)
├── dataset_statistics.py       Thống kê số cửa sổ / người dùng
├── ve_bieu_do.py               Sinh toàn bộ biểu đồ minh họa dữ liệu
├── export/
│   ├── export_tflite.py        Export Keras/checkpoint → TFLite (hợp nhất 3 kiến trúc)
│   └── export_tflite_cnn.py    Export riêng backbone CNN → TFLite
└── Active_Auth_Training.ipynb  Notebook Colab: huấn luyện đầy đủ (chứa models/dataset/config)
    Active_Auth_RunAll_Colab.ipynb   Notebook Colab: chạy toàn bộ pipeline
```

## Thông số tiền xử lý chính

| Thông số | Giá trị |
|----------|---------|
| Tần số lấy mẫu | ~50 Hz (tần số gốc của thiết bị, **không** resample) |
| Cửa sổ | 200 mẫu (4 giây), bước trượt 20 mẫu (chồng lấn 90%) |
| Chuẩn hóa | Z-score theo từng cửa sổ, từng kênh |
| Kênh quán tính | 9 (acc/gyro/mag × xyz) |
| Đặc trưng touch | 48-D (tap 16 + scroll 23 + keystroke 9) |
| Ngữ cảnh | `walking` (chỉ đi bộ) và `all` (3 hoạt động) |
| Embedding | 128-D, từ backbone CNN 1D / CNN-LSTM / CNN-BiLSTM |

## Quy trình chạy

### Chạy local (chỉ cần numpy, pandas, matplotlib, scipy)

```bash
# 1. Kiểm tra chất lượng dữ liệu thô
python step1_quality_check.py

# 2. Tiền xử lý: data/ -> processed/
python step2_preprocess.py

# 3. Thống kê bộ dữ liệu
python dataset_statistics.py

# 4. Sinh biểu đồ minh họa -> plots/
python ve_bieu_do.py --data_dir ./data --proc_dir ./processed --out ./plots
```

### Chạy trên Colab (cần PyTorch + các module `models.py`, `dataset.py`, `config.py`)

Các script `backbone_train.py` và `export_embeddings.py` phụ thuộc vào `models`,
`dataset`, `config` — các module này nằm trong notebook Colab
(`Active_Auth_Training.ipynb`). Huấn luyện và xuất embedding được thực hiện trên Colab:

```bash
python backbone_train.py        # huấn luyện backbone -> artifacts/<arch>/
python export_embeddings.py --data_dir processed --root .   # -> emb_out/*.npz
```

### Xuất TFLite cho Android

```bash
python export/export_tflite.py      # hợp nhất (CNN / ConvLSTM / ConvLSTM-Bi)
python export/export_tflite_cnn.py  # riêng backbone CNN
```

## Yêu cầu môi trường

- Python 3.10+
- Local: `numpy`, `pandas`, `matplotlib`, `scipy`
- Huấn luyện/Export: `torch`, `tensorflow` (chạy trên Colab có GPU)

## Ghi chú

- Dữ liệu được giữ ở **tần số gốc** của từng thiết bị; không chuẩn hóa lại tần số.
  Cửa sổ cắt theo **số mẫu** (200) nên độ dài thời gian thực có thể chênh nhẹ giữa
  các thiết bị — đây là một hạn chế đã biết.
- App Android (inference) lấy 200 mẫu acc thô gần nhất ở tần số gốc, gyro/mag được
  căn đồng bộ theo dấu thời gian acc → khớp với cách dựng cửa sổ khi huấn luyện.
- `ve_bieu_do.py` đọc `processed/*.npy` cho biểu đồ phân bố và `data/*.csv` cho biểu đồ
  tín hiệu thô; đổi user minh họa qua hằng số `USER_DON` hoặc cờ `--user`.
