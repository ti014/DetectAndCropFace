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

########################################################################################
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

########################################################################################
def visualize(
    image,
    detections
) -> np.ndarray:
    """Draws bounding boxes and keypoints on the input image and returns it."""
    annotated_image = image.copy()
    height, width, _ = image.shape
    

    if detections:
        for detection in detections:
            bbox = detection.location_data.relative_bounding_box
            start_point = int(bbox.xmin * width), int(bbox.ymin * height)
            end_point = int((bbox.xmin + bbox.width) * width), int((bbox.ymin + bbox.height) * height)
            cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

            for keypoint in detection.location_data.relative_keypoints:
                keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y, width, height)
                color, thickness, radius = (0, 255, 0), 2, 2
                cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

            category_id = detection.label_id
            confidence_score = detection.score
            result_text = f'Label ID: {category_id}, Confidence: {confidence_score}'
            text_location = (MARGIN + start_point[0], MARGIN + ROW_SIZE + start_point[1])
            cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return annotated_image

########################################################################################
def calculate_face_score(bbox, image_width, image_height):
    """Calculate the score of a face based on its size and proximity to the center of the image."""
    area = bbox.width * bbox.height
    
    # Calculate the coordinates of the center of the face
    face_center_x = bbox.xmin + bbox.width / 2
    face_center_y = bbox.ymin + bbox.height / 2
    
    image_center_x = image_width / 2
    image_center_y = image_height / 2
    distance_to_center = np.sqrt((image_center_x - face_center_x) ** 2 + (image_center_y - face_center_y) ** 2)
    

    return area * (1 / (distance_to_center + 1))

########################################################################################
def find_largest_and_closest_to_center_face(detections, image_width, image_height):
    """Find the largest face closest to the center of the image."""
    best_face_bbox = None
    best_score = -float('inf')

    for detection in detections:
        bbox = detection.location_data.relative_bounding_box
        face_score = calculate_face_score(bbox, image_width, image_height)

        if face_score > best_score:
            best_face_bbox = bbox
            best_score = face_score

    return best_face_bbox

########################################################################################
def visualize_largest_and_closest_to_center_face(image, detections) -> np.ndarray:
    """Visualize the largest face closest to the center of the image."""
    annotated_image = image.copy()
    height, width, _ = image.shape

    best_face_bbox = find_largest_and_closest_to_center_face(detections, width, height)
    if best_face_bbox is not None:
        start_point = int(best_face_bbox.xmin * width), int(best_face_bbox.ymin * height)
        end_point = int((best_face_bbox.xmin + best_face_bbox.width) * width), int((best_face_bbox.ymin + best_face_bbox.height) * height)
        cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)


        category = detections[0].label_id
        confidence_score = detections[0].score
        result_text = f'Label ID: {category}, Confidence: {confidence_score}'
        text_location = (MARGIN + start_point[0], MARGIN + ROW_SIZE + start_point[1])
        cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return annotated_image

########################################################################################
def main():
    cap = cv2.VideoCapture("videos/video_3.mp4")  
    mp_face_detection = mp.solutions.face_detection

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
            if results.detections:
                best_face_bbox = find_largest_and_closest_to_center_face(results.detections, frame.shape[1], frame.shape[0])
                if best_face_bbox:
                    ymin = int(best_face_bbox.ymin * frame.shape[0])
                    xmin = int(best_face_bbox.xmin * frame.shape[1])
                    ymax = int((best_face_bbox.ymin + best_face_bbox.height) * frame.shape[0])
                    xmax = int((best_face_bbox.xmin + best_face_bbox.width) * frame.shape[1])
                    cropped_face = frame[ymin:ymax, xmin:xmax]
                    cv2.imwrite("cropped_face.jpg", cropped_face)

                    cropped_face_array = np.array(cropped_face)
                    print("Shape of cropped face array:", cropped_face_array.shape)


                    annotated_image = visualize(annotated_image, results.detections)
                else:
                    print("No face detected or bounding box is missing!")

            cv2.imshow('MediaPipe Face Detection', annotated_image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()