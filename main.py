import cv2
import mediapipe as mp
from function.visualization_utils import visualize
from function.save_image import save_faces
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

            # Kiểm tra xem kết quả phát hiện có khả dụng hay không
            if results.detections:
                annotated_image, cropped_faces = visualize(frame, results.detections, largest_face_only=True)
                save_faces(cropped_faces)
            else:
                annotated_image = frame

            cv2.imshow('MediaPipe Face Detection', annotated_image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

