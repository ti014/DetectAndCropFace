import cv2
def save_faces(cropped_faces):
    for i, face in enumerate(cropped_faces):
        cv2.imwrite(f'face_{i}.jpg', face)