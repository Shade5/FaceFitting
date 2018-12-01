import cv2
import dlib
import numpy as np


im = cv2.imread('data/face.jpeg')
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

rects = detector(gray, 1)
shape = predictor(gray, rects[0])
rects = [(rects[0].tl_corner().x, rects[0].tl_corner().y), (rects[0].br_corner().x, rects[0].br_corner().y)]
landmarks = np.zeros((68, 2))
for i, p in enumerate(shape.parts()):
    landmarks[i] = (p.x, p.y)
    im = cv2.circle(im, (p.x, p.y), radius=1, color=(255, 0, 0), thickness=3)
    # Remove this to stop animation
    # cv2.imshow('a', im)
    # cv2.waitKey(100)

im = cv2.rectangle(im, rects[0], rects[1], color=(0, 255, 0), thickness=2)

cv2.imshow('a', im)
cv2.waitKey(0)
cv2.imwrite('results/landmarks.png', im)