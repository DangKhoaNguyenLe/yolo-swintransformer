# Hệ Thống Nhận Diện Gương Mặt Đa Góc Độ (Multi-Angle Face Recognition System)

## Giới thiệu
Dự án Nhận diện Gương mặt đa góc độ, sử dụng kết hợp giữa **OpenCV** để phát hiện mặt và hướng dẫn định vị góc xoay đầu, cùng với kiến trúc học sâu **Swin Transformer** để trích xuất đặc trưng (embedding) cho độ chính xác cao. Hệ thống cung cấp quy trình (pipeline) hoàn chỉnh từ việc **Đăng ký đa góc**, **Huấn luyện (Trích xuất Embedding)** đến **Nhận diện thời gian thực qua webcam**.

## Mô hình và Các thành phần chính

1. **Phát hiện khuôn mặt & Ước lượng góc xoay (Head Pose Estimation):**
   - Vận dụng `haarcascades` của OpenCV (`frontalface_default`, `profileface`, `eye`) nhằm đưa ra một bộ phát hiện gọn nhẹ và nhanh chóng.
   - **Chiến lược thông minh dựa vào mắt:** Thay vì dùng các framework nặng nề để ước lượng góc 3D, hệ thống sử dụng vị trí vùng mắt (Eye Detection) để phân loại góc Ngẩng lên (UP) hay Cúi xuống (DOWN); và kết hợp xoay lật frame (`cv2.flip`) để bắt chuẩn góc nghiêng TRÁI/PHẢI.

2. **Trích xuất đặc trưng (Face Embedding Model - Swin Transformer):**
   - Lõi nhận diện sử dụng các mô hình họ Vision Transformer – cụ thể là **Swin Transformer**.
   - Mạng sẽ tiến hành nhận đầu vào là các vùng khuôn mặt (Face crops) đã cắt và mã hóa thành một vector đặc trưng không gian đa chiều (Embedding).
   - Mô hình Transform này có khả năng trích xuất các đặc điểm khuôn mặt một cách mạnh mẽ kể cả trong điều kiện khuôn mặt bị suy giảm chi tiết hay bị quay đi các góc khác nhau.

3. **So khớp & Nhận diện độ tương đồng (Similarity Search):**
   - So sánh khoảng cách hoặc độ tương đồng (ví dụ: Cosine similarity) giữa vector đầu vào trên frame của Webcam và các vectors đã ghi chép lại trong `database`.
   - Nếu điểm nhận dạng vượt qua ngưỡng an toàn (mặc định > 0.6), hệ thống sẽ gán nhãn tên chính xác cho người dùng.

## Luồng hoạt động (Workflow)

Chương trình vận hành bằng **Giao diện dòng lệnh (CLI)** tương tác dễ dàng, chia làm các bước:

*   **Bước 1 - Đăng ký (Enrollment):** 
    Người dùng chọn chức năng `1. Đăng ký gương mặt mới` -> Nhập tên. Giao diện trực quan trên cam sẽ dẫn dắt người dùng thực hiện đủ các thao tác nhìn thẳng, quay đầu trái phải, ngẩng/cúi. Camera có cơ chế kiểm tra (Check), chỉ tự động bấm máy chụp (`auto-capture`) khi người dùng đã nhìn ĐÚNG GÓC.
    - *Đầu ra:* Các file ảnh đã tự động cắt gọn quanh khuôn mặt nằm trong `dataset/persons/<Tên_Người_Dùng>`.

*   **Bước 2 - Huấn luyện dữ liệu (Training):** 
    Sử dụng chức năng `2. Huấn luyện embeddings`. Hệ thống sẽ dọn dẹp các luồng ảnh hiện có, phát hiện mặt, crop lại cẩn thận rồi đưa qua mạng Swin Transformer. 
    - *Đầu ra:* Kết quả là các vector đa chiều được "đóng gói" vào tệp tin `database/embeddings.pkl`.

*   **Bước 3 - Nhận diện trực tiếp (Real-time Recognition):** 
    Chức năng `3. Nhận diện gương mặt`. Hệ thống liên tục thu thập từng frame ảnh, dò tìm mặt và vector hoá. Sau đó làm phép so sánh nhanh với cơ sở dữ liệu và hiển thị nhãn `Tên người dùng (Mức độ khớp)` theo thời gian thực. (Hiện nhãn <Unknown> nếu người lạ hoặc chưa vượt ngưỡng nhận diện).

### Sơ đồ dữ liệu
`Webcam` ➔ `Crop Mắt/Mặt (OpenCV)` ➔ `Swin Transformer (timm, torch)` ➔ `Vector Đặc Trưng` ➔ `So sánh Tương đồng` ➔ `Tên & Độ tin cậy (UI)`

## Cách Cài đặt & Sử dụng

### 1. Yêu cầu & Cài đặt Thư Viện
Yêu cầu bạn đã thiết lập **Python 3.8+**. Để cài đặt các dependency phụ thuộc, hãy trỏ tới thư mục dự án và chạy:

```bash
pip install -r requirements.txt
```

> **Danh sách thư viện sử dụng:** 
> - `torch`, `torchvision`, `timm` (chạy học sâu và Swin Transformer)
> - `opencv-python` (Thao tác ảnh, xử lý camera, dò mặt Haar)
> - `numpy`, `scikit-learn`...

### 2. Tiền huấn luyện mô hình với LFW (Khuyên dùng)
Trước khi chạy hệ thống chính, bạn nên tiền huấn luyện mô hình Swin Transformer trên tập dữ liệu LFW (Labeled Faces in the Wild) để tăng cường độ chuẩn xác khi trích xuất đặc trưng. Mở Terminal và chạy lệnh sau (quá trình này sẽ sinh ra file trọng số `database/lfw_swin.pth`):

```bash
python train_lfw.py
```

> **Chỉ số đánh giá huấn luyện (Training Metrics):** Quá trình huấn luyện này sẽ tự động xuất ra một biểu đồ `lfw_training_metrics.png` đánh giá tổng quan với hai chỉ số cơ bản:
> - **Loss (Hàm suy hao - CrossEntropy):** Đo lường mức độ sai lệch giữa dự đoán của mạng Swin Transformer và danh tính thực tế. Đồ thị Loss càng giảm dần về mức thấp (gần 0) chứng tỏ mạng càng nhận diện sắc bén và ít sai sót hơn.
> - **Accuracy (Độ chính xác):** Thể hiện tỷ lệ phần trăm các khuôn mặt được phân loại đúng nhãn ở mỗi Epoch. Đồ thị Accuracy tăng dần và hướng tới trần 1.0 (100%) khẳng định khả năng trích xuất đặc điểm (embedding) của mô hình đã đạt đến mức chuyên gia.

### 3. Khởi chạy Hệ Thống
Mở dự án ở Terminal hoặc Command Prompt, chạy điểm neo chính của chương trình:

```bash
python main.py
```

### 4. Giao diện Menu Chính
Màn hình dòng lệnh sẽ hiển thị Menu lựa chọn:

```text
==================================================
HỆ THỐNG NHẬN DIỆN GƯƠNG MẶT
==================================================
1. Đăng ký gương mặt mới
2. Huấn luyện embeddings
3. Nhận diện gương mặt
4. Xem danh sách người dùng
5. Xóa người dùng
0. Thoát chương trình
==================================================
Chọn chức năng (0-5): 
```

Bạn nên tuân thủ quy trình cơ bản: **Đăng ký (1) ➔ Huấn luyện (2) ➔ Nhận diện thử nghiệm (3)**.

### 5. Cấu trúc Dự án của Dữ Liệu Tạo Ra
Khi hệ thống chạy, nó sẽ tự quản lý hai thư mục chính phục vụ lưu trữ như sau:
```
mask-detection/
│
├── dataset/                     # Chứa kho ảnh thô đã capture 
│   └── persons/
│       ├── NguyenVanA/          # File ảnh của người A
│       └── ...
│
├── database/                    # Lưu file Database Vector Trích xuất
│   ├── lfw_swin.pth             # Mạng được pre-train bằng train_lfw.py
│   └── embeddings.pkl           # Trọng số Embedding để so sánh nhanh
```
