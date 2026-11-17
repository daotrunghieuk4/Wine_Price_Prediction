# Wine Rating Prediction

Phân tích dữ liệu rượu vang và dự đoán **rating** với CatBoost theo ba nhóm số lượng đánh giá. Tên repository được giữ là `Wine_Price_Prediction`; trong ứng dụng hiện tại, `Price` là đầu vào, `SmoothedRating` là mục tiêu huấn luyện và `PredictedRating` là đầu ra.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `Train/Data/` | CSV Red, White, Rose, Sparkling và Varieties |
| `Train/DSS/` | Notebook CatBoost và các mô hình so sánh |
| `Train/models/` | Ba model CatBoost dùng chung cho ứng dụng |
| `WineRating/app.py` | Demo Streamlit: xếp hạng và điều chỉnh giá/tuổi |
| `WineRating/File_data_test/` | Bốn CSV mẫu để thử demo |

Mã nguồn, dữ liệu và model đã được tách khỏi ZIP để có thể xem và tải trực tiếp trên GitHub. Môi trường ảo và cache không được đưa vào repo.

## Cài đặt và chạy demo

```powershell
git clone https://github.com/daotrunghieuk4/Wine_Price_Prediction.git
cd Wine_Price_Prediction
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r WineRating/requirements.txt
python -m streamlit run WineRating/app.py
```

Trên Linux/macOS, kích hoạt bằng `source .venv/bin/activate`. Mở địa chỉ Streamlit in trong terminal (mặc định `http://localhost:8501`) và tải một CSV trong `WineRating/File_data_test/`.

Kiểm tra model và dự đoán trên cả bốn CSV mẫu bằng `python tests/smoke_test.py`.

CSV cần các cột `Name`, `Country`, `Region`, `Winery`, `Year`, `Rating`, `Price`, `NumberOfRatings`, `WineStyle`. Các dòng năm `N.V.` hoặc thiếu trường bắt buộc sẽ bị loại trong tiền xử lý. Ứng dụng dùng mốc năm **2025** để giữ cách tính tuổi tương ứng với thí nghiệm gốc.

## Dữ liệu và notebook

Dữ liệu CSV đã có trong `Train/Data/`, không cần tải riêng. Nguồn tải và giấy phép phân phối của dữ liệu chưa được ghi đầy đủ trong bản gốc; cần bổ sung thông tin này trước khi tái phân phối bên ngoài dự án.

```powershell
python -m pip install -r requirements.txt
cd Train/DSS
python -m notebook CatBoost_ByGroup.ipynb
```

Notebook đọc CSV từ `../Data/` và lưu model vào `../models/`. Chạy các cell theo thứ tự. Huấn luyện lại sẽ ghi đè model tương ứng; các output đang hiển thị là kết quả được lưu từ thí nghiệm gốc. Các notebook so sánh có thể cần thêm `autogluon.tabular`, `lightgbm`, `xgboost` hoặc `torch`, tùy mô hình; xem các import ở đầu notebook tương ứng.

## Kết quả đã lưu

Các dòng dưới đây lấy từ output đánh giá trong `CatBoost_ByGroup.ipynb`, chưa phải một lần chạy lại trong môi trường mới.

| Phép đánh giá | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mid holdout | 0.1039 | 0.1366 | 0.7595 |
| Low-only validation | 0.1185 | 0.1588 | 0.6433 |
| High-only validation | 0.0932 | 0.1309 | 0.8111 |

## Giới hạn

Ứng dụng yêu cầu rating có sẵn và tạo các đặc trưng điểm theo quốc gia/vùng/nhà sản xuất từ rating của chính dữ liệu đầu vào. Quy trình này có nguy cơ rò rỉ nhãn; các chỉ số trên chưa chứng minh khả năng dự đoán rating của sản phẩm hoàn toàn chưa được đánh giá. Trước khi dùng làm benchmark, cần tách train/test trước khi học các thống kê và lưu bộ biến đổi từ tập train. Phần điều chỉnh giá/tuổi là mô phỏng phản ứng của model, không chứng minh quan hệ nhân quả.
