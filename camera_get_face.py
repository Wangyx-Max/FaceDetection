import cv2
import dlib
import numpy as np

path_screenshots = "data/screenshots/"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/dlib/shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    flag, frame = cap.read()
    k = cv2.waitKey(100)
    img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = detector(img_gray, 0)
    if len(faces) == 1:
        landmarks = np.mat([[p.x, p.y] for p in predictor(frame, faces[0]).parts()])
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            cv2.circle(frame, pos, 2, color=(100, 0, 0))

    if k == ord('q'):
        break

    cv2.namedWindow("camera", 1)
    cv2.imshow("camera", frame)

cap.release()
cv2.destroyAllWindows()