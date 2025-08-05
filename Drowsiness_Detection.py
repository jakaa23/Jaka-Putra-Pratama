from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import torch
from playsound import playsound

# --- EAR Function ---
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# --- Constants ---
thresh = 0.25
frame_check = 20
flag = 0
hp_detected = False
alarm_on = False

# --- Load Dlib Detector ---
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# --- Load YOLOv5 Model ---
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/handphone_debug/weights/best.pt', force_reload=True)

# --- Start Webcam ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    # --- Drowsiness Detection ---
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < thresh:
            flag += 1
        else:
            flag = 0

    # --- Handphone Detection ---
    results = model(frame)
    hp_detected = False
    for *box, conf, cls in results.xyxy[0]:
        label = model.names[int(cls)]
        if label.lower() in ['handphone', 'phone']:
            hp_detected = True
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # --- Trigger Alarm if Drowsy or Using Phone ---
    if flag >= frame_check or hp_detected:
        cv2.putText(frame, "****************ALERT!****************", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "****************ALERT!****************", (10, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if not alarm_on:
            alarm_on = True
            playsound("alarm.wav")
    else:
        alarm_on = False

    # --- Display ---
    cv2.imshow("Drowsiness + Handphone Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
