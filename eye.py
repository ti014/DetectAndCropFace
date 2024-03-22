import cv2
import mediapipe as mp
import time

umbral = 0.1

def blink_detection(coordLeftEye, coordRightEye):
    earLeftEye = 0
    earRightEye = 0

    if len(coordLeftEye) > 0:
        earLeftEye = ratio(coordLeftEye)

    if len(coordRightEye) > 0:
        earRightEye = ratio(coordRightEye)

    ear = (earLeftEye + earRightEye) / 2

    return ear != 0 and ear < umbral

def left_blink(coordLeftEye, coordRightEye):
    leftEye = 0
    rightEye = 0

    if len(coordLeftEye) > 0:
        leftEye = ratio(coordLeftEye)

    if len(coordRightEye) > 0:
        rightEye = ratio(coordRightEye)

    return leftEye != 0 and leftEye < umbral and rightEye > umbral and rightEye != 0

def right_blink(coordLeftEye, coordRightEye):
    leftEye = 0
    rightEye = 0

    if len(coordLeftEye) > 0:
        leftEye = ratio(coordLeftEye)

    if len(coordRightEye) > 0:
        rightEye = ratio(coordRightEye)

    return rightEye != 0 and rightEye < umbral and leftEye > umbral and leftEye != 0

def ratio(coordinate):
    d_A = coordinate[1].y - coordinate[5].y
    d_B = coordinate[2].y - coordinate[4].y
    d_C = coordinate[0].x - coordinate[3].x

    return (d_A + d_B) / (2 * d_C)


cap = cv2.VideoCapture("videos/video_7.mp4")
pTime = 0
eye_movement_detected = False  # Biến để theo dõi trạng thái phát hiện chuyển động mắt

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=5)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = faceMesh.process(imgRGB)
    if result.multi_face_landmarks:
        for faceLms in result.multi_face_landmarks:
            # mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS)
            coordLeftEye = [faceLms.landmark[159], faceLms.landmark[145], faceLms.landmark[144], faceLms.landmark[163], faceLms.landmark[362], faceLms.landmark[133], faceLms.landmark[173], faceLms.landmark[157], faceLms.landmark[158], faceLms.landmark[159]]
            coordRightEye = [faceLms.landmark[386], faceLms.landmark[374], faceLms.landmark[373], faceLms.landmark[390], faceLms.landmark[393], faceLms.landmark[384], faceLms.landmark[362], faceLms.landmark[373], faceLms.landmark[374], faceLms.landmark[386]]
            
            if blink_detection(coordLeftEye, coordRightEye) or left_blink(coordLeftEye, coordRightEye) or right_blink(coordLeftEye, coordRightEye):
                eye_movement_detected = True  # Cập nhật trạng thái phát hiện chuyển động mắt

    if eye_movement_detected:
        cv2.putText(img, "REAL", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)
    cv2.imshow("img", img)
    if cv2.waitKey(5) & 0xFF == 27:
        break
