import numpy as np
import cv2
import math
from typing import Tuple, Union
import mediapipe as mp

class FaceProcessor:
    def __init__(self):
        self.margin = 10  # pixels
        self.row_size = 10  # pixels
        self.font_size = 1
        self.font_thickness = 1
        self.text_color = (255, 0, 0)  # red
        self.face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

    def _normalized_to_pixel_coordinates(
        self, normalized_x: float, normalized_y: float, image_width: int, image_height: int
    ) -> Union[None, Tuple[int, int]]:
        """Converts normalized value pair to pixel coordinates."""
        def is_valid_normalized_value(value: float) -> bool:
            return 0 <= value <= 1

        if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
            return None
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px

    def visualize(
        self, image, detections
    ) -> np.ndarray:
        """Draws bounding boxes and keypoints on the input image and returns it."""
        annotated_image = image.copy()
        height, width, _ = image.shape

        if detections:
            for detection in detections:
                bbox = detection.location_data.relative_bounding_box
                start_point = int(bbox.xmin * width), int(bbox.ymin * height)
                end_point = int((bbox.xmin + bbox.width) * width), int((bbox.ymin + bbox.height) * height)
                cv2.rectangle(annotated_image, start_point, end_point, self.text_color, 3)
            




                for keypoint in detection.location_data.relative_keypoints:
                    keypoint_px = self._normalized_to_pixel_coordinates(keypoint.x, keypoint.y, width, height)
                    if keypoint_px:
                        color, thickness, radius = (0, 255, 0), 2, 2
                        cv2.circle(annotated_image, keypoint_px, radius, color, thickness)

                category_id = detection.label_id
                confidence_score = detection.score
                result_text = f'Label ID: {category_id}, Confidence: {confidence_score}'
                text_location = (self.margin + start_point[0], self.margin + self.row_size + start_point[1])
                cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, self.font_size, self.text_color, self.font_thickness)

        return annotated_image

    def find_largest_and_closest_to_center_face(self, detections, image_width, image_height):
        """Find the largest face closest to the center of the image."""
        best_face_bbox = None
        best_score = -float('inf')

        for detection in detections:
            bbox = detection.location_data.relative_bounding_box
            face_score = self.calculate_face_score(bbox, image_width, image_height)

            if face_score > best_score:
                best_face_bbox = bbox
                best_score = face_score

        return best_face_bbox

    def calculate_face_score(self, bbox, image_width, image_height):
        """Calculate the score of a face based on its size and proximity to the center of the image."""
        area = bbox.width * bbox.height

        # Calculate the coordinates of the center of the face
        face_center_x = bbox.xmin + bbox.width / 2
        face_center_y = bbox.ymin + bbox.height / 2

        image_center_x = image_width / 2
        image_center_y = image_height / 2
        distance_to_center = np.sqrt((image_center_x - face_center_x) ** 2 + (image_center_y - face_center_y) ** 2)

        return area * (1 / (distance_to_center + 1))

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.face_detection.process(rgb_frame)

        annotated_image = frame.copy()
        if results.detections:
            best_face_bbox = self.find_largest_and_closest_to_center_face(results.detections, frame.shape[1], frame.shape[0])
            if best_face_bbox:
                # Use the best_face_bbox to determine the coordinates of the largest and closest face to the center
                ymin = max(0, int(best_face_bbox.ymin * frame.shape[0]))
                xmin = max(0, int(best_face_bbox.xmin * frame.shape[1]))
                ymax = min(frame.shape[0], int((best_face_bbox.ymin + best_face_bbox.height) * frame.shape[0]))
                xmax = min(frame.shape[1], int((best_face_bbox.xmin + best_face_bbox.width) * frame.shape[1]))

                # Check if the face region is valid before proceeding
                if ymin < ymax and xmin < xmax:
                    cropped_face = frame[ymin:ymax, xmin:xmax]

                    if cropped_face.size > 0:
                        cv2.imwrite("cropped_face.jpg", cropped_face)
                        print("Khuôn mặt đã được cắt và lưu.")
                        annotated_image = self.visualize(annotated_image, results.detections)
                    else:
                        print("Khuôn mặt cắt ra là rỗng hoặc không hợp lệ.")
                else:
                    print("Vùng khuôn mặt cắt ra không hợp lệ.")

            else:
                print("Không phát hiện khuôn mặt hoặc thiếu thông tin về bounding box!")
            #  Kiểm tra độ sáng của hình ảnh
        brightness_result, mean_brightness = self.check_brightness(frame)
        print(f"Độ sáng của hình ảnh: {brightness_result}, Độ sáng trung bình: {mean_brightness}")

        return annotated_image

    def check_brightness(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])

        sum = np.sum(hist)
        dark = 0
        bright = 0
        for i in range(0, 65):
            dark += hist[i, 0]
        for i in range(230, 255):
            bright += hist[i, 0]
        dark_per = dark / sum
        bright_per = bright / sum

        # So sánh với NGƯỠNG
        if dark_per > 0.40:
            result = 'Tối'
            mean = dark_per
        elif bright_per > 0.40:
            result = 'Sáng'
            mean = bright_per
        else:
            result = 'Bình thường'
            mean = dark_per

        return result, mean
    def adjust_brightness(self, frame, alpha=2, beta=0):
        """Tăng độ sáng của hình ảnh."""
        adjusted_image = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        return adjusted_image

# Sử dụng ví dụ:
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    face_processor = FaceProcessor()

    while cap.isOpened():
        success, frame = cap.read()
        if not success or frame is None:
            print("Bỏ qua khung hình trống hoặc không hợp lệ từ camera.")
            continue  # Tiếp tục vòng lặp nếu không có khung hình

        # Kiểm tra và điều chỉnh độ sáng của khung hình
        # adjusted_frame = face_processor.adjust_brightness(frame)
        # annotated_image = face_processor.process_frame(adjusted_frame)
        
        annotated_image = face_processor.process_frame(frame)
        cv2.imshow('DetectAndDropFace', annotated_image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

# face_processor.release()
cap.release()
cv2.destroyAllWindows()