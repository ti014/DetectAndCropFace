## Project này có gì?:
### Cấu trúc:
`project_folder/`
│
├── `modules/`
│   ├── `__init__.py`
│   ├── `utils.py`
│   ├── `visualization.py`
│   ├── `face_detection.py`
│
├── `main.py`

Trong đó:

`__init__.py:` File này thông báo cho Python rằng thư mục modules được xem như một package.
`utils.py:` Chứa các hàm tiện ích như _normalized_to_pixel_coordinates.
`visualization.py:` Chứa các hàm liên quan đến việc hiển thị và chú thích hình ảnh.
`face_detection.py:` Chứa các hàm liên quan đến việc nhận diện khuôn mặt và xử lý kết quả.

### Hàm (Functions):
1. `_normalized_to_pixel_coordinates`: Chuyển đổi các giá trị normalized thành tọa độ pixel trên hình ảnh.
2. `visualize`: Vẽ khung giới hạn và các điểm chính trên hình ảnh đầu vào và trả về hình ảnh đã được chú thích.
3. `calculate_face_score`: Tính toán điểm số của một khuôn mặt dựa trên kích thước và sự gần gũi với trung tâm của hình ảnh.
4. `find_largest_and_closest_to_center_face`: Tìm khuôn mặt lớn nhất và gần trung tâm nhất trên hình ảnh.
5. `visualize_largest_and_closest_to_center_face`: Hiển thị khuôn mặt lớn nhất và gần trung tâm nhất trên hình ảnh.

### Lớp (Classes):
Không có lớp nào trong project này.

### Module

1. **Module `utils`:**
   - Chứa các hàm tiện ích như `_normalized_to_pixel_coordinates` để thực hiện các tác vụ chung như chuyển đổi từ tọa độ normalized thành tọa độ pixel trên hình ảnh.

2. **Module `visualization`:**
   - Bao gồm các hàm liên quan đến việc hiển thị và chú thích hình ảnh, như `visualize` để vẽ khung giới hạn và điểm chính trên hình ảnh, cũng như `visualize_largest_and_closest_to_center_face` để hiển thị khuôn mặt lớn nhất và gần trung tâm nhất trên hình ảnh.

3. **Module `face_detection`:**
   - Chứa các hàm liên quan đến việc nhận diện khuôn mặt và xử lý kết quả, như `calculate_face_score` để tính điểm số của khuôn mặt và `find_largest_and_closest_to_center_face` để tìm khuôn mặt lớn nhất và gần trung tâm nhất trên hình ảnh.

4. **Module `main`:**
   - Bao gồm hàm `main`, chính là điểm khởi đầu của chương trình, chứa mã để khởi tạo camera, xử lý hình ảnh và hiển thị kết quả.

