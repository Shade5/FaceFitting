import cv2
import dlib
import numpy as np
from MorphabelModel import MorphabelModel
import mesh

im = cv2.imread('data/female.jpg')
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
h, w, c = im.shape

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

rects = detector(gray, 1)
shape = predictor(gray, rects[0])
rects = [(rects[0].tl_corner().x, rects[0].tl_corner().y), (rects[0].br_corner().x, rects[0].br_corner().y)]
landmarks = np.zeros((68, 2))

for i, p in enumerate(shape.parts()):
    landmarks[i] = [p.x, p.y]
    im = cv2.circle(im, (p.x, p.y), radius=3, color=(0, 0, 255), thickness=5)

bfm = MorphabelModel('models/BFM.mat')
x = mesh.transform.from_image(landmarks, h, w)
X_ind = bfm.kpt_ind


fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t = bfm.fit(x, X_ind, max_iter=200, isShow=False)
colors = bfm.generate_colors(np.random.rand(bfm.n_tex_para, 1))
colors = np.minimum(np.maximum(colors, 0), 1)

fitted_vertices = bfm.generate_vertices(fitted_sp, fitted_ep)
transformed_vertices = bfm.transform(fitted_vertices, fitted_s, fitted_angles, fitted_t)
image_vertices = mesh.transform.to_image(transformed_vertices, h, w)
fitted_image = mesh.render.render_colors(image_vertices, bfm.triangles, colors, h, w)
cv2.imwrite('results/female_ori.png', im)
cv2.imwrite('results/female_fit.png', 255*cv2.cvtColor(fitted_image, cv2.COLOR_BGR2RGB))
# cv2.imshow('inital', im)
# cv2.imshow('fit', cv2.cvtColor(fitted_image, cv2.COLOR_BGR2RGB))
# cv2.waitKey(0)
