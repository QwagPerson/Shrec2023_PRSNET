import numpy as np


def matmul(mats):
    out = mats[0]

    for i in range(1, len(mats)):
        out = np.matmul(out, mats[i])

    return out


# Returns the rotation matrix in X
def rotationX(theta):
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    return np.array([
        [1, 0, 0, 0],
        [0, cos_theta, -sin_theta, 0],
        [0, sin_theta, cos_theta, 0],
        [0, 0, 0, 1]], dtype=np.float32)


# Returns the rotation matrix in Y
def rotationY(theta):
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    return np.array([
        [cos_theta, 0, sin_theta, 0],
        [0, 1, 0, 0],
        [-sin_theta, 0, cos_theta, 0],
        [0, 0, 0, 1]], dtype=np.float32)


# Returns the rotation matrix in Y
def rotationZ(theta):
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    return np.array([
        [cos_theta, -sin_theta, 0, 0],
        [sin_theta, cos_theta, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]], dtype=np.float32)


# Returns the translation matrix
def translate(tx, ty, tz):
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]], dtype=np.float32)


# Stores the information of a symmetry plane
class SymmetryPlane:
    def __init__(self, point, normal):
        # 3D coords of a canonical plane (for drawing)
        self.coordsBase = np.array([[0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1]], dtype=np.float32)
        # Indices for the canonical plane
        self.trianglesBase = np.array([[0, 1, 3], [3, 1, 2]], dtype=np.int32)

        # The plane is determined by a normal vector and a point
        self.point = point.astype(np.float32)
        self.normal = normal
        self.normal = self.normal / (np.linalg.norm(self.normal) + 1e-5)

        self.compute_geometry()

    # Applies a rotation to the plane
    def apply_rotation(self, rot):
        transf = rot.copy()
        transf = transf[0:3, 0:3]
        transf = np.linalg.inv(transf).T

        self.normal = transf @ self.normal

        self.compute_geometry()

    def apply_traslation(self, x, y, z):
        self.point[0] = self.point[0] + x
        self.point[1] = self.point[1] + y
        self.point[2] = self.point[2] + z

        # print(self.point)

    # Transforms the canonical plane to be oriented wrt the normal
    def compute_geometry(self):
        # Be sure the vector is normal
        self.normal = self.normal / (np.linalg.norm(self.normal) + 1e-6)
        # print(f'First normal: {self.normal}')
        a, b, c = self.normal

        h = np.sqrt(a ** 2 + c ** 2)

        if h < 0.0000001:
            angle = np.pi / 2

            T = translate(self.point[0], self.point[1], self.point[2])
            Rz = rotationZ(angle)
            transform = matmul([T, Rz])
        else:

            Rzinv = np.array([
                [h, -b, 0, 0],
                [b, h, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)

            Ryinv = np.array([
                [a / h, 0, -c / h, 0],
                [0, 1, 0, 0],
                [c / h, 0, a / h, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)

            T = translate(self.point[0], self.point[1], self.point[2])

            transform = matmul([T, Ryinv, Rzinv])

        ones = np.ones((1, 4))
        self.coords = np.concatenate((self.coordsBase.T, ones))

        self.coords = transform @ self.coords

        self.coords = self.coords[0:3, :].T


