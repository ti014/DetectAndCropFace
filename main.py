# import numpy as np
# import cv2
# import math
# # from typing import Tuple, Union
# import mediapipe as mp

# MARGIN = 10  # pixels
# ROW_SIZE = 10  # pixels
# FONT_SIZE = 1
# FONT_THICKNESS = 1
# TEXT_COLOR = (255, 0, 0)  # red

# def _normalized_to_pixel_coordinates(
#     normalized_x: float, normalized_y: float, image_width: int,
#     image_height: int) -> Union[None, Tuple[int, int]]:
#     """Converts normalized value pair to pixel coordinates."""
#     def is_valid_normalized_value(value: float) -> bool:
#         return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

#     if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
#         return None
#     x_px = min(math.floor(normalized_x * image_width), image_width - 1)
#     y_px = min(math.floor(normalized_y * image_height), image_height - 1)
#     return x_px, y_px

# def calculate_face_score(bbox, image_width, image_height):
#     area = bbox.width * bbox.height
    
#     face_center_x = bbox.xmin + bbox.width / 2
#     face_center_y = bbox.ymin + bbox.height / 2
    
#     image_center_x = image_width / 2
#     image_center_y = image_height / 2
#     distance_to_center = np.sqrt((image_center_x - face_center_x) ** 2 + (image_center_y - face_center_y) ** 2)
    

#     return area * (1 / (distance_to_center + 1))

# def find_largest_and_closest_to_center_face(detections, image_width, image_height):
#     """Find the largest face closest to the center of the image."""
#     best_face_bbox = None
#     best_score = -float('inf')

#     for detection in detections:
#         bbox = detection.location_data.relative_bounding_box
#         face_score = calculate_face_score(bbox, image_width, image_height)

#         if face_score > best_score:
#             best_face_bbox = bbox
#             best_score = face_score

#     return best_face_bbox


# def visualize(image, detections, largest_face_only=False) -> np.ndarray:
#     """Draws bounding boxes and keypoints on the input image and returns it."""
#     annotated_image = image.copy()
#     height, width, _ = image.shape

#     face_count = 0  # Biến đếm số lượng khuôn mặt

#     if detections:
#         if largest_face_only:
#             best_face_bbox = find_largest_and_closest_to_center_face(detections, width, height)
#             detections = [best_face_bbox] if best_face_bbox else []
#         for detection in detections:
#             face_count += 1
#             bbox = detection.location_data.relative_bounding_box
#             start_point = int(bbox.xmin * width), int(bbox.ymin * height)
#             end_point = int((bbox.xmin + bbox.width) * width), int((bbox.ymin + bbox.height) * height)
#             cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

#             # Draw keypoints
#             for keypoint in detection.location_data.relative_keypoints:
#                 keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y, width, height)
#                 color, thickness, radius = (0, 255, 0), 2, 2
#                 cv2.circle(annotated_image, keypoint_px, thickness, color, radius)
#             confidence_score = detection.score
#             result_text = f'Khuon mat {{{face_count}}}, Confidence: {confidence_score}'
#             text_location = (MARGIN + start_point[0], MARGIN + ROW_SIZE + start_point[1])
#             cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

#     # Hiển thị thông báo về số lượng khuôn mặt được phát hiện
#     detection_text = f'Phat hien co {face_count} khuon mat'
#     cv2.putText(annotated_image, detection_text, (MARGIN, MARGIN + ROW_SIZE), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

#     return annotated_image




import cv2
import mediapipe as mp
from function.visualization_utils import visualize
from function.face_detection_utils import find_largest_and_closest_to_center_face
def main():
    cap = cv2.VideoCapture(0)  # Mở camera mặc định
    mp_face_detection = mp.solutions.face_detection

    # Bước 2: Tạo một đối tượng FaceDetection.
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = face_detection.process(rgb_frame)
            

            annotated_image = frame.copy()
            annotated_image = visualize(annotated_image, results.detections)

            cv2.imshow('MediaPipe Face Detection', annotated_image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

