# Wine Rating Prediction

Ứng dụng phân tích dữ liệu rượu vang và ước lượng **rating** bằng CatBoost, kết hợp giao diện Streamlit để khám phá kết quả và mô phỏng thay đổi giá, tuổi rượu. Mô hình được tổ chức theo ba nhóm số lượng đánh giá, giúp biểu diễn các mức độ hỗ trợ khác nhau của dữ liệu rating.

Trong pipeline, `Price` là một đặc trưng đầu vào, `SmoothedRating` là mục tiêu huấn luyện và `PredictedRating` là đầu ra phục vụ phân tích. Dự án bao gồm dữ liệu, notebook thực nghiệm, model và ứng dụng, cho phép theo dõi quy trình từ chuẩn bị dữ liệu đến tương tác với kết quả.

## Chức năng

- Nhập CSV và tạo các đặc trưng giá, tuổi, tần suất xuất hiện và điểm tổng hợp theo nhóm.
- Chọn model CatBoost tương ứng với mức độ hỗ trợ rating của mỗi dòng dữ liệu.
- Xếp hạng sản phẩm theo chênh lệch giữa rating dự đoán và rating đã làm trơn.
- Thử các giá trị giá/tuổi khác nhau và quan sát phản hồi của model.
- Đối chiếu CatBoost với các phương pháp khác trong bộ notebook thực nghiệm.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `Train/Data/` | CSV Red, White, Rose, Sparkling và Varieties |
| `Train/DSS/` | Notebook CatBoost và các mô hình so sánh |
| `Train/models/` | Ba model CatBoost dùng chung cho ứng dụng |
| `WineRating/app.py` | Demo Streamlit: xếp hạng và điều chỉnh giá/tuổi |
| `WineRating/File_data_test/` | Bốn CSV mẫu để thử demo |

Ba model được lưu tập trung trong `Train/models/`. Ứng dụng xác định đường dẫn theo vị trí mã nguồn, thuận tiện khi khởi chạy từ thư mục gốc repo hoặc từ thư mục ứng dụng.

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

Dữ liệu được cung cấp trực tiếp trong `Train/Data/` và tải cùng repo khi clone.

| Tệp | Số dòng | Vai trò |
|---|---:|---|
| `Red.csv` | 8.666 | Dữ liệu rượu đỏ |
| `White.csv` | 3.764 | Dữ liệu rượu trắng |
| `Rose.csv` | 397 | Dữ liệu rượu rosé |
| `Sparkling.csv` | 1.007 | Dữ liệu rượu sparkling |
| `Varieties.csv` | 1.512 | Danh sách tên giống nho |

Notebook bổ sung thông tin loại rượu khi kết hợp các CSV nguồn. Để chạy ứng dụng ngay, dùng các file trong `WineRating/File_data_test/`, đã có cột `WineStyle` theo cấu trúc đầu vào của demo. Quyền sử dụng dữ liệu cần được xác nhận theo nguồn cung cấp khi áp dụng ngoài phạm vi thực nghiệm của repository.

```powershell
python -m pip install -r requirements.txt
cd Train/DSS
python -m notebook CatBoost_ByGroup.ipynb
```

Notebook đọc CSV từ `../Data/` và lưu model vào `../models/`. Chạy các cell theo thứ tự. Huấn luyện lại sẽ ghi đè model tương ứng; các output đang hiển thị là kết quả được lưu từ thí nghiệm gốc. Các notebook so sánh có thể cần thêm `autogluon.tabular`, `lightgbm`, `xgboost` hoặc `torch`, tùy mô hình; xem các import ở đầu notebook tương ứng.

## Phương pháp

Rating được làm trơn bằng trung bình chung và số lượng đánh giá của từng sản phẩm:

```text
SmoothedRating = (Rating × NumberOfRatings + GlobalMean × 5)
                 / (NumberOfRatings + 5)
```

Ứng dụng phân nhóm theo `NumberOfRatings`: **Low** dưới 35, **Medium** từ 35 đến dưới 838 và **High** từ 838 trở lên. Mỗi nhóm sử dụng một model CatBoost tương ứng. Các đặc trưng kết hợp thông tin phân loại như quốc gia, vùng, nhà sản xuất với giá, tuổi và các thống kê được tạo trong bước tiền xử lý.

Giao diện gồm ba vùng phân tích: nhóm có chênh lệch dự đoán cao, mô phỏng giá/tuổi và nhóm có chênh lệch dự đoán thấp. Chênh lệch được tính bằng `PredictedRating - SmoothedRating`, dùng để sắp xếp kết quả trong phiên phân tích.

## Kết quả thực nghiệm

Bảng dưới tổng hợp các phép đánh giá được lưu trong [notebook CatBoost](Train/DSS/CatBoost_ByGroup.ipynb). MAE và RMSE đo sai số dự đoán rating; R² mô tả mức độ giải thích biến thiên của mục tiêu.

| Phép đánh giá | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mid holdout | 0.1039 | 0.1366 | 0.7595 |
| Low-only validation | 0.1185 | 0.1588 | 0.6433 |
| High-only validation | 0.0932 | 0.1309 | 0.8111 |

## Thiết lập đánh giá và diễn giải

Pipeline hiện phục vụ phân tích bộ sản phẩm đã có rating: các điểm tổng hợp theo quốc gia, vùng và nhà sản xuất được tạo từ rating trong tập đầu vào. Vì vậy, bảng kết quả được đọc trong bối cảnh thực nghiệm này. Với bài toán dự đoán sản phẩm chưa có rating, cần tách dữ liệu trước khi học thống kê và tái sử dụng bộ biến đổi từ tập train để tránh đưa thông tin nhãn vào đặc trưng đánh giá.

Chức năng điều chỉnh giá/tuổi thể hiện độ nhạy của dự đoán theo đầu vào model. Các thay đổi hiển thị được dùng để khám phá kịch bản, không được diễn giải thành tác động nhân quả của giá hoặc tuổi lên rating.

## Kiểm tra và mở rộng

`tests/smoke_test.py` nạp ba model và thực hiện dự đoán trên bốn CSV mẫu, kiểm tra đầu ra hữu hạn và sự hiện diện của cả ba nhóm. Kiểm tra này giúp xác nhận đường dẫn model, định dạng dữ liệu và luồng suy luận khi thiết lập môi trường.

Để bổ sung một model, bắt đầu từ notebook tương ứng trong `Train/DSS/`, thống nhất đặc trưng đầu vào và ghi lại cách chia dữ liệu. Khi thay model cho ứng dụng, giữ đồng bộ tên đặc trưng, nhóm độ tin cậy và cách tiền xử lý giữa bước huấn luyện và suy luận.
