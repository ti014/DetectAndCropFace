import numpy as np

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

def calculate_face_score(bbox, image_width, image_height):
    area = bbox.width * bbox.height
    
    face_center_x = bbox.xmin + bbox.width / 2
    face_center_y = bbox.ymin + bbox.height / 2
    
    image_center_x = image_width / 2
    image_center_y = image_height / 2
    distance_to_center = np.sqrt((image_center_x - face_center_x) ** 2 + (image_center_y - face_center_y) ** 2)
    

    return area * (1 / (distance_to_center + 1))

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


def find_largest_and_closest_to_center_face(detections, image_width, image_height, max_faces=10, min_face_size=0.1):
    """Find the largest face closest to the center of the image."""
    top_detections = sorted(detections, key=lambda x: x.score, reverse=True)[:max_faces]

    best_face_bbox = None
    best_score = -float('inf')

    for detection in top_detections:
        bbox = detection.location_data.relative_bounding_box
        face_size = bbox.width * bbox.height

        # Bỏ qua các khuôn mặt có kích thước nhỏ
        if face_size < min_face_size:
            continue

        face_score = calculate_face_score(bbox, image_width, image_height)

        if face_score > best_score:
            best_face_bbox = bbox
            best_score = face_score

    return best_face_bbox