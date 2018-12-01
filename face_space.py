import numpy as np
from MorphabelModel import MorphabelModel
import mesh
import imageio

bfm = MorphabelModel('models/BFM.mat')

shape_coefficients = np.zeros((bfm.n_shape_para, 1))
expr_coefficients = np.zeros((bfm.n_exp_para, 1))
color_coefficients = np.zeros((bfm.n_tex_para, 1))
images = []

for i in range(50):
    shape_coefficients += 0.2*np.vstack((np.random.randn(10, 1), np.zeros((bfm.n_shape_para - 10, 1))))
    expr_coefficients += 0.0001*np.vstack((np.random.randn(5, 1), np.zeros((bfm.n_exp_para - 5, 1))))
    color_coefficients += 0.2*np.vstack((np.random.randn(10, 1), np.zeros((bfm.n_tex_para - 10, 1))))
    vertices = bfm.generate_vertices(shape_coefficients, expr_coefficients)
    colors = bfm.generate_colors(color_coefficients)

    s = 2e-03
    angles = [0, 0, 0]
    t = [0, 0, 0]
    transformed_vertices = bfm.transform(vertices, s, angles, t)
    projected_vertices = transformed_vertices.copy()

    h = w = 512
    c = 3
    image_vertices = mesh.transform.to_image(projected_vertices, h, w)
    image = mesh.render.render_colors(image_vertices, bfm.triangles, colors, h, w)
    images.append(image)

images.extend(reversed(images))
imageio.mimsave('results/movie.gif', images)