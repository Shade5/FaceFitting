import cv2
import dlib
import numpy as np
from MorphabelModel import MorphabelModel
import mesh

bfm = MorphabelModel('models/BFM.mat')

shape_coefficients = np.zeros((bfm.n_shape_para, 1))
color_coefficients = np.zeros((bfm.n_tex_para, 1))
for i in range(100):
    shape_coefficients += 0.2*np.random.randn(bfm.n_shape_para, 1)
    color_coefficients += 0.1*np.random.randn(bfm.n_tex_para, 1)
    vertices = bfm.generate_vertices(shape_coefficients)
    colors = bfm.generate_colors(color_coefficients)

    s = 8e-04
    angles = [0, 0, 0]
    t = [0, 0, 0]
    transformed_vertices = bfm.transform(vertices, s, angles, t)
    projected_vertices = transformed_vertices.copy()

    h = w = 256
    c = 3
    image_vertices = mesh.transform.to_image(projected_vertices, h, w)
    image = mesh.render.render_colors(image_vertices, bfm.triangles, colors, h, w)

    cv2.imshow('a', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cv2.waitKey(1)

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