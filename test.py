import numpy as np
import cv2
import math
from typing import Tuple, Union
import mediapipe as mp

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px

def calculate_face_distance_to_screen_center(bbox, image_width, image_height):
    """Calculate the distance of the face to the center of the screen."""
    # Calculate the coordinates of the center of the face
    face_center_x = bbox.xmin + bbox.width / 2
    face_center_y = bbox.ymin + bbox.height / 2
    
    # Calculate the coordinates of the center of the screen
    screen_center_x = image_width / 2
    screen_center_y = image_height / 2
    
    # Calculate the distance from the center of the face to the center of the screen
    face_distance_to_center = np.sqrt((screen_center_x - face_center_x) ** 2 + (screen_center_y - face_center_y) ** 2)
    
    return face_distance_to_center

def find_nearest_face_to_screen_center(detections, image_width, image_height):
    """Find the face nearest to the center of the screen."""
    nearest_face_bbox = None
    nearest_distance = float('inf')

    for detection in detections:
        bbox = detection.location_data.relative_bounding_box
        face_distance_to_screen_center = calculate_face_distance_to_screen_center(bbox, image_width, image_height)

        if face_distance_to_screen_center < nearest_distance:
            nearest_face_bbox = bbox
            nearest_distance = face_distance_to_screen_center

    return nearest_face_bbox

def visualize_nearest_face_to_screen_center(image, detections) -> np.ndarray:
    """Visualize the face nearest to the center of the screen."""
    annotated_image = image.copy()
    height, width, _ = image.shape

    nearest_face_bbox = find_nearest_face_to_screen_center(detections, width, height)
    if nearest_face_bbox is not None:
        start_point = int(nearest_face_bbox.xmin * width), int(nearest_face_bbox.ymin * height)
        end_point = int((nearest_face_bbox.xmin + nearest_face_bbox.width) * width), int((nearest_face_bbox.ymin + nearest_face_bbox.height) * height)
        cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

        # Draw label and score
        category = detections[0].label_id
        confidence_score = detections[0].score
        result_text = f'Label ID: {category}, Confidence: {confidence_score}'
        text_location = (MARGIN + start_point[0], MARGIN + ROW_SIZE + start_point[1])
        cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

        # Draw keypoints
        for keypoint in detections[0].location_data.relative_keypoints:
            keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y, width, height)
            color, thickness, radius = (0, 255, 0), 2, 2
            cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

    return annotated_image

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
            if results.detections:  # Kiểm tra xem có phát hiện nào không
                annotated_image = visualize_nearest_face_to_screen_center(annotated_image, results.detections)

            cv2.imshow('MediaPipe Face Detection', annotated_image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
