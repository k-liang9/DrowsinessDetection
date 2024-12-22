import cv2
import dlib
from scipy.spatial import distance

cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_face_landmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def EAR(eye):
    poi_A = distance.euclidean(eye[1], eye[5])
    poi_B = distance.euclidean(eye[2], eye[4])
    poi_C = distance.euclidean(eye[0], eye[3])
    eye_aspect_ratio = (poi_A + poi_B) / (2 * poi_C)
    return eye_aspect_ratio

def capture_eye():
    pass

while True:
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)

    for face in faces:
        face_landmarks = dlib_face_landmark(gray, face)

        for n in range(0, 68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            cv2.circle(image, (x, y), 1, (0, 255, 255), 1)

    cv2.imshow("Output", image)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()