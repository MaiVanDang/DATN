# Xác Thực Hành Vi Liên Tục Trên Android (DATN)

Hệ thống xác thực sinh trắc học hành vi (Behavioral Biometric Authentication) chạy hoàn toàn trên thiết bị Android, không cần kết nối mạng.

## Cấu trúc dự án

```
DATN/
├── android_app/          # Ứng dụng Android xác thực
│   └── B_authenticator_app/
├── data_collectiom/      # Ứng dụng Android thu thập dữ liệu
│   └── DataCollectV2/
├── ml_pipeline/          # Toàn bộ pipeline ML (data → model)
│   ├── data/             # Dữ liệu thô (CSV từ app thu thập)
│   ├── processed/        # Dữ liệu đã tiền xử lý (NPY/CSV)
│   ├── training/         # Notebook huấn luyện + artifacts
│   │   └── artifacts/    # Model weights, impostor pools
│   ├── export/           # Script xuất TFLite cho Android
│   ├── step1_quality_check.py
│   ├── step2_preprocess.py
│   └── step3_visualize.py
├── web_demo/             # Demo Streamlit chạy trên PC
└── docs/                 # Tài liệu báo cáo
```

## Pipeline tổng quan

```
Thu thập dữ liệu          Tiền xử lý             Huấn luyện
(data_collectiom app) --> ml_pipeline/step2  --> training notebook
                                                       |
                                                  backbone.pt
                                                       |
                                                  export_tflite_keras.py
                                                       |
                                                  backbone.tflite
                                                       |
                                             android_app (TFLite inference)
```

## Mô hình

- **Backbone CNN**: Conv1D × 3 + AdaptiveAvgPool → 128-D embedding từ tín hiệu IMU (walking, 4s window)
- **Inertial scoring**: Cosine similarity so sánh embedding hiện tại với anchors đã enroll
- **Touch scoring**: Random Forest 200 cây trên 48-D feature vector (tap + scroll + keystroke)
- **Score fusion**: EMA (α=0.8) trên 5 cửa sổ gần nhất

## Ngưỡng xác thực

| Mức | Score |
|-----|-------|
| TRUSTED | ≥ 0.75 |
| WARNING | ≥ 0.45 |
| UNKNOWN | < 0.45 |

## Yêu cầu

- **ML pipeline**: Python 3.10+, PyTorch, TensorFlow 2.x, scikit-learn
- **Android app**: Android Studio, minSdk 26, TFLite runtime
- **Web demo**: `pip install -r web_demo/requirements.txt`
