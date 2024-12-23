from turtledemo.penrose import start

import cv2
import dlib
from scipy.spatial import distance
import time

LEFT_EYE_BEGIN = 36
LEFT_EYE_END = 42
RIGHT_EYE_BEGIN = 42
RIGHT_EYE_END = 48

EYE_THRESHOLD = 0.3 #can be optimized by ml
EYE_DROOP_INTERVAL = 1.5

eyes_heavy = False
eyes_start = 0
eyes_end = 0

def EAR(eye):
    poi_A = distance.euclidean(eye[1], eye[5])
    poi_B = distance.euclidean(eye[2], eye[4])
    poi_C = distance.euclidean(eye[0], eye[3])
    eye_aspect_ratio = (poi_A + poi_B) / (2 * poi_C)
    return eye_aspect_ratio

def capture_eye(arr, begin, end):
    for i in range(begin, end):
        arr.append((face_landmarks.part(i).x, face_landmarks.part(i).y))

def track_EAR():
    global eyes_heavy, eyes_start, eyes_end
    eye_ratio = (EAR(left_eye) + EAR(right_eye)) / 2
    if eye_ratio < EYE_THRESHOLD:
        if eyes_heavy:
            eyes_end = time.time()
            if eyes_end - eyes_start > 1.5:
                print("ALERT: drowsiness alert: eyes drooping")
        else:
            eyes_heavy = True
            eyes_start = time.time()
    else:
        eyes_heavy = False

cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_face_landmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray)

    left_eye = []
    right_eye = []

    for face in faces:
        face_landmarks = dlib_face_landmark(gray, face)

        capture_eye(left_eye, LEFT_EYE_BEGIN, LEFT_EYE_END)
        capture_eye(right_eye, RIGHT_EYE_BEGIN, RIGHT_EYE_END)

        track_EAR()

    # #outlining parts
    # for n in range(0, 68):
    #     x = face_landmarks.part(n).x
    #     y = face_landmarks.part(n).y
    #     cv2.circle(image, (x, y), 1, (0, 255, 255), 1)
    # for n in range(0, 6):
    #     next = n + 1
    #     if next == 6:
    #         next = 0
    #     cv2.line(image, left_eye[n], left_eye[next], (255, 255, 0), 1)
    #     cv2.line(image, right_eye[n], right_eye[next], (255, 255, 0), 1)
    cv2.imshow("Output", image)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()