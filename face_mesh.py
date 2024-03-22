
# import cv2
# import mediapipe as mp
# import time



# cap = cv2.VideoCapture("videos/video_2.mp4")
# pTime = 0

# mpDraw = mp.solutions.drawing_utils
# mpFaceMesh = mp.solutions.face_mesh
# faceMesh = mpFaceMesh.FaceMesh(max_num_faces = 3)
# while True:
#     success, img = cap.read()
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     result = faceMesh.process(imgRGB)
#     if result.multi_face_landmarks:
#         for faceLms in result.multi_face_landmarks:
#             mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS)
#             for id,lm in enumerate(faceLms.landmark):
#                 # print(lm)
                
#                 ih, iw, ic = img.shape
#                 x, y, z = int(iw * lm.x), int(ih * lm.y), int(ic * lm.z)
#                 print(id,x,y,z)
    
#     cTime = time.time()
#     fps = 1 / (cTime - pTime)
#     pTime = cTime
#     cv2.putText(img, f'FPS: {int(fps)}', (20, 70),
#                 cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)
#     cv2.imshow("img", img)
#     if cv2.waitKey(5) & 0xFF == 27:
#         break

import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("videos/video_2.mp4")
# cap = cv2.VideoCapture(0)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceDetection = mp.solutions.face_detection

with mpFaceDetection.FaceDetection(min_detection_confidence=0.5) as faceDetection:
    while True:
        success, img = cap.read()
        if not success:
            print("Không đọc được khung hình.")
            continue

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = faceDetection.process(imgRGB)

        if result.detections:
            for detection in result.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)
        cv2.imshow("img", img)
        if cv2.waitKey(5) & 0xFF == 27:
            break

