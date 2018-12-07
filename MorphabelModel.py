import numpy as np
import scipy.io
import mesh
import fit


class MorphabelModel():
    def __init__(self, model_path):
        self.model = None
        self.load(model_path)

        self.nver = self.model['shapePC'].shape[0] / 3
        self.ntri = self.model['tri'].shape[0]
        self.n_shape_para = self.model['shapePC'].shape[1]
        self.n_exp_para = self.model['expPC'].shape[1]
        self.n_tex_para = self.model['texPC'].shape[1]

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

    def generate_vertices(self, shape_para, exp_para):
        '''
        Args:
            shape_para: (n_shape_para, 1)
            exp_para: (n_exp_para, 1)
        Returns:
            vertices: (nver, 3)
        '''
        vertices = self.model['shapeMU'] + self.model['shapePC'].dot(shape_para) + self.model['expPC'].dot(exp_para)
        vertices = np.reshape(vertices, [int(3), int(len(vertices)/3)], 'F').T

        return vertices

    def generate_colors(self, tex_para):
        colors = self.model['texMU'] + self.model['texPC']@(tex_para)
        colors = np.reshape(colors, [3, len(colors) // 3], 'F').T / 255.

        return colors

    def transform(self, vertices, s, angles, t3d):
        R = mesh.transform.angle2matrix(angles)
        return mesh.transform.similarity_transform(vertices, s, R, t3d)

    def fit(self, x, X_ind, max_iter=4, isShow = False):
        if isShow:
            fitted_sp, fitted_ep, s, R, t = fit.fit_points_for_show(x, X_ind, self.model, n_sp = self.n_shape_para, n_ep = self.n_exp_para, max_iter = max_iter)
            angles = np.zeros((R.shape[0], 3))
            for i in range(R.shape[0]):
                angles[i] = mesh.transform.matrix2angle(R[i])
        else:
            fitted_sp, fitted_ep, s, R, t = fit.fit_points(x, X_ind, self.model, n_sp = self.n_shape_para, n_ep = self.n_exp_para, max_iter = max_iter)
            angles = mesh.transform.matrix2angle(R)
        return fitted_sp, fitted_ep, s, angles, t
