# import cv2
# import math
# import numpy as np
# from typing import Tuple, Union

# from .face_detection_utils import find_largest_and_closest_to_center_face

# MARGIN = 10  # pixels
# ROW_SIZE = 10  # pixels
# FONT_SIZE = 1
# FONT_THICKNESS = 1
# TEXT_COLOR = (0, 0, 255)  # red
# BOX_COLOR = (0,255,0) # green

# # def _normalized_to_pixel_coordinates(
# #     normalized_x: float, normalized_y: float, image_width: int,
# #     image_height: int) -> Union[None, Tuple[int, int]]:
# #     """Converts normalized value pair to pixel coordinates."""
# #     def is_valid_normalized_value(value: float) -> bool:
# #         return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

# #     if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
# #         return None
# #     x_px = min(math.floor(normalized_x * image_width), image_width - 1)
# #     y_px = min(math.floor(normalized_y * image_height), image_height - 1)
# #     return x_px, y_px

# def visualize(image, detections, largest_face_only=False) -> np.ndarray:
#     """Draws bounding boxes and keypoints on the input image and returns it."""
#     annotated_image = image.copy()
#     height, width, _ = image.shape

#     face_count = 0 

#     if detections:
#         if largest_face_only:
#             best_face_bbox = find_largest_and_closest_to_center_face(detections, width, height)
#             detections = [best_face_bbox] if best_face_bbox else []
#         for detection in detections:
#             face_count += 1
#             bbox = detection.location_data.relative_bounding_box
#             start_point = int(bbox.xmin * width), int(bbox.ymin * height)
#             end_point = int((bbox.xmin + bbox.width) * width), int((bbox.ymin + bbox.height) * height)
#             cv2.rectangle(annotated_image, start_point, end_point, BOX_COLOR, 3)

#             # Draw keypoints
#             # for keypoint in detection.location_data.relative_keypoints:
#             #     keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y, width, height)
#             #     color, thickness, radius = (0, 255, 0), 2, 2
#             #     cv2.circle(annotated_image, keypoint_px, thickness, color, radius)
#             # confidence_score = detection.score
            
#             # result_text = f'Khuon mat {{{face_count}}}, Confidence: {confidence_score}'
#             # text_location = (MARGIN + start_point[0], MARGIN + ROW_SIZE + start_point[1])
#             # cv2.putText(annotated_image, result_text,text_location, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

#     detection_text = f'Phat hien co {face_count} khuon mat'
#     cv2.putText(annotated_image, detection_text, (MARGIN, MARGIN + ROW_SIZE), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

#     return annotated_image



import cv2
import math
import numpy as np
from typing import List, Tuple, Union

from .face_detection_utils import find_largest_and_closest_to_center_face

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (0, 0, 255)  # red
BOX_COLOR = (0, 255, 0)  # green

def visualize(image, detections, largest_face_only=False) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Draws bounding boxes and keypoints on the input image and returns it along with cropped faces."""
    annotated_image = image.copy()
    height, width, _ = image.shape

    face_count = 0 
    cropped_faces = []

    if detections:
        if largest_face_only:
            best_face_bbox = find_largest_and_closest_to_center_face(detections, width, height)
            detections = [best_face_bbox] if best_face_bbox else []
        for detection in detections:
            face_count += 1
            bbox = detection.location_data.relative_bounding_box
            start_point = int(bbox.xmin * width), int(bbox.ymin * height)
            end_point = int((bbox.xmin + bbox.width) * width), int((bbox.ymin + bbox.height) * height)
            cv2.rectangle(annotated_image, start_point, end_point, BOX_COLOR, 3)

            # Crop the face region
            face_region = image[int(bbox.ymin * height):int((bbox.ymin + bbox.height) * height),
                                int(bbox.xmin * width):int((bbox.xmin + bbox.width) * width)]
            cropped_faces.append(face_region)

    detection_text = f'Phat hien co {face_count} khuon mat'
    cv2.putText(annotated_image, detection_text, (MARGIN, MARGIN + ROW_SIZE), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return annotated_image, cropped_faces
