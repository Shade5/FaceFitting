import numpy as np
import scipy.io
import mesh

class MorphabelModel():
    def __init__(self, model_path):
        self.model = None
        self.load(model_path)

        self.nver = self.model['shapePC'].shape[0] / 3
        self.ntri = self.model['tri'].shape[0]
        self.n_shape_para = self.model['shapePC'].shape[1]
        self.n_exp_para = self.model['expPC'].shape[1]
        self.n_tex_para = self.model['texMU'].shape[1]

        self.kpt_ind = self.model['kpt_ind']
        self.triangles = self.model['tri']
        self.full_triangles = np.vstack((self.model['tri'], self.model['tri_mouth']))

    def load(self, model_path):
        model = scipy.io.loadmat(model_path)['model'][0, 0]

        # change dtype from double(np.float64) to np.float32,
        # since big matrix process(espetially matrix dot) is too slow in python.
        model['shapeMU'] = (model['shapeMU'] + model['expMU']).astype(np.float32)
        model['shapePC'] = model['shapePC'].astype(np.float32)
        model['shapeEV'] = model['shapeEV'].astype(np.float32)
        model['expEV'] = model['expEV'].astype(np.float32)
        model['expPC'] = model['expPC'].astype(np.float32)

        # matlab start with 1. change to 0 in python.
        model['tri'] = model['tri'].T.copy(order='C').astype(np.int32) - 1
        model['tri_mouth'] = model['tri_mouth'].T.copy(order='C').astype(np.int32) - 1

        # kpt ind
        model['kpt_ind'] = (np.squeeze(model['kpt_ind']) - 1).astype(np.int32)

        self.model = model

    def generate_vertices(self, shape_para):
        vertices = self.model['shapeMU'] + self.model['shapePC']@(self.model['shapeEV'] * shape_para)
        vertices = np.reshape(vertices, [3, len(vertices)//3], 'F').T

        return vertices

    def generate_colors(self, tex_para):
        colors = self.model['texMU'] + self.model['texPC']@(self.model['texEV'] * tex_para)
        colors = np.reshape(colors, [3, len(colors) // 3], 'F').T / 255.

        return colors

    def transform(self, vertices, s, angles, t3d):
        R = mesh.transform.angle2matrix(angles)
        return mesh.transform.similarity_transform(vertices, s, R, t3d)
