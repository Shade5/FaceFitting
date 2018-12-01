import cv2
import dlib
import numpy as np
from MorphabelModel import MorphabelModel
import mesh

im = cv2.resize(cv2.imread('data/simple.jpg'), (512, 512))
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

rects = detector(gray, 1)
shape = predictor(gray, rects[0])
rects = [(rects[0].tl_corner().x, rects[0].tl_corner().y), (rects[0].br_corner().x, rects[0].br_corner().y)]
landmarks = np.zeros((68, 2))

for i, p in enumerate(shape.parts()):
    landmarks[i] = [p.x, p.y]
    im = cv2.circle(im, (p.x, p.y), radius=1, color=(255, 0, 0), thickness=3)
    # Remove this to stop animation
    # cv2.imshow('a', im)
    # cv2.waitKey(100)

bfm = MorphabelModel('models/BFM.mat')
X_ind = bfm.kpt_ind

fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t = bfm.fit(landmarks, X_ind, max_iter=3)

fitted_vertices = bfm.generate_vertices(fitted_sp, fitted_ep)
transformed_vertices = bfm.transform(fitted_vertices, fitted_s, fitted_angles, fitted_t)

image_vertices = mesh.transform.to_image(transformed_vertices, 512, 512)
fitted_image = mesh.render.render_colors(image_vertices, bfm.triangles, 255*np.ones((bfm.n_tex_para, 1)), 512, 512)

cv2.imshow('a', fitted_image)
cv2.waitKey(0)

pass


#
# im = cv2.rectangle(im, rects[0], rects[1], color=(0, 255, 0), thickness=2)
#
# cv2.imshow('a', im)
# cv2.waitKey(0)
# cv2.imwrite('results/landmarks.png', im)