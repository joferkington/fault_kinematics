from inclined_shear import rotate
import numpy as np
import itertools

class TestRotate:
    def test_forward_equal_inverse(self):
        def check(theta, alpha, phi):
            x, y = np.mgrid[:10, :20]
            z = np.ones_like(x)
            xyz = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
            rxyz = rotate(xyz, theta, alpha, phi)
            ixyz = rotate(rxyz, theta, alpha, phi, inverse=True)
            assert np.allclose(ixyz, xyz)

        thetas = range(-180, 360, 30)
        alphas = range(-90, 100, 30)
        phis = range(-90, 100, 30)
        for theta, alpha, phi in itertools.product(thetas, alphas, phis):
            t, a, p = [np.radians(item) for item in [theta, alpha, phi]]
            check(t, a, p)


