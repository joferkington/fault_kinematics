import numpy as np
import scipy.interpolate as interpolate
import scipy.optimize

def invert_slip(faultxyz, horizonxyz, alpha=None, guess=(0,0), 
                return_metric=False, verbose=False, overlap_thresh=0.3, 
                **kwargs):
    """
    Given a fault, horizon, and optionally, a shear angle, and starting guess, 
    find the offset vector that best flattens the horizon using Powell's method.

    If the shear angle (`alpha`) is not specified, invert for an offset vector
    and a shear angle.

    Uses the variance as a metric of "flatness".  

    Parameters:
    -----------
        faultxyz : An Nx3 array of points making up the fault surface
        horxyz : An Mx3 array of points along the horizon surface
        alpha : A shear angle (in degrees) measured from vertical (i.e.
            vertical shear corresponds to alpha=0) through which the hanging
            wall deforms. This is constrained to lie in the vertical plane
            defined by the slip vector. If alpha is None, it will be solved for.
        guess : An initial displacement vector of (dx, dy) or (dx, dy, alpha)
            if alpha is not specified.
        return_metric : If True, return the minimized "roughness".
        overlap_thresh : If less than `overlap_thresh*100` percent of the 
            "moved" horizon's points are within the bounds of the fault, the 
            result is penalized.

        Additional keyword arguments are passed on to scipy.optimize.fmin_powell

    Returns:
    --------
        slip : A sequence of `(dx, dy)` or `(dx, dy, alpha)` if alpha is not 
            manually specified, defining the offset vector (and/or shear angle)
            that best flattens the horizon
        metric : (Only if return_metric is True) The resulting "flattness".
    """
    if (alpha is None) and (len(guess) == 2):
        guess = guess + (0,)

    func = _Shear(faultxyz, horizonxyz, alpha, overlap_thresh)

    # Set a few defaults...
    kwargs['disp'] = kwargs.get('disp', False)
    kwargs['full_output'] = True

    # Powell's method appears more reliable than CG for this problem...
    items = scipy.optimize.fmin_powell(func, guess, **kwargs)
    slip = items[0]

    if return_metric:
        return slip, items[1]
    else:
        return slip

def vertical_shear(faultxyz, horxyz, slip, remove_invalid=True):
    """
    Models vertical shear along a fault.  Uses Piecewise linear interpolation
    to define surfaces from the given, unordered, points.

    Parameters:
    -----------
        faultxyz : An Nx3 array of points making up the fault surface
        horxyz : An Mx3 array of points along the horizon surface
        slip : A displacement vector in 2 or 3 dimensions.  If 2D, the
            last element is assumed to be 0. (3D is allowed so that this 
            function can be used easily within the inclined_shear function.)
        remove_invalid : A boolean indicating whether points that have been
            moved off the fault's surface and have undefined values should be
            removed from the results. If True, only valid points will be 
            returned, if False, the result may have NaN's.

    Returns:
    --------
        movedxyz : An Mx3 array of points representing the "moved" horizon.
    """
    try:
        dx, dy = slip
        dz = 0
    except ValueError:
        dx, dy, dz = slip

    # Interpolate the fault's elevation values at the starting and ending
    # positions for the horizon's xy values.
    interp = interpolate.LinearNDInterpolator(faultxyz[:,:2], faultxyz[:,-1])
    zorig = interp(horxyz[:,:2])
    zfinal = interp(horxyz[:,:2] + [dx, dy])

    # Calculate the z-offset for the horizon by the difference in the _fault's_
    # elevation at the starting and ending horizon xy positions.
    dz = (zfinal - zorig) + dz

    # Remove points that have been moved off the fault, as their values are
    # undefined.
    if remove_invalid:
        mask = np.isfinite(dz)
        horxyz = horxyz[mask]
    else:
        mask = np.ones(dz.shape, dtype=np.bool)

    # Update the horizon's position
    horxyz[:,:2] += [dx, dy]
    horxyz[:,-1] += dz[mask]
    return horxyz

def inclined_shear(faultxyz, horxyz, slip, alpha, remove_invalid=True):
    """
    Models homogenous inclined shear along a fault.  This assumes that the
    shear angle lies in the plane of slip. Uses Piecewise linear interpolation
    to define surfaces from the given, unordered, points.

    Parameters:
    -----------
        faultxyz : An Nx3 array of points making up the fault surface
        horxyz : An Mx3 array of points along the horizon surface
        slip : A displacement vector in 2 dimensions.         
        alpha : A shear angle (in degrees) measured from vertical (i.e.
            vertical shear corresponds to alpha=0) through which the hanging
            wall deforms. This is constrained to lie in the vertical plane
            defined by the slip vector.
        remove_invalid : A boolean indicating whether points that have been
            moved off the fault's surface and have undefined values should be
            removed from the results. If True, only valid points will be 
            returned, if False, the result may have NaN's.

    Returns:
    --------
        movedxyz : An Mx3 array of points representing the "moved" horizon.
    """
    dx, dy = slip
    theta = np.arctan2(dy, dx)
    alpha = np.radians(alpha)

    # Rotate slip vector, horizon, and fault into a new reference frame such
    # that "down" is parallel to alpha. 
    slip = rotate([dx, dy, 0], theta, alpha)[0]
    rotated_horxyz = rotate(horxyz, theta, alpha)
    rotated_faultxyz = rotate(faultxyz, theta, alpha)

    # In the new reference frame, we can just use vertical shear...
    moved_xyz = vertical_shear(rotated_faultxyz, rotated_horxyz, slip[:2],
                               remove_invalid)

    # Then we rotate things back to the original reference frame.
    return rotate(moved_xyz, theta, alpha, inverse=True)

def rotate(xyz, theta, alpha, phi=0, inverse=False):
    """
    Rotates a point cloud `xyz` by the three Euler angles `theta`, `alpha`, and
    `phi` given in radians.  Preforms the inverse rotation if `inverse` is True.
    (Intended for internal use. Subject to "gimbal lock".)

    Rotations are preformed first around the "z" axis (by theta), then around
    the "y" axis (by alpha), then around the "x" axis (by phi).

    All angles are in radians

    Parameters:
    -----------
        xyz : An Nx3 array of points.
        theta : The rotation angle about the z axis (in the xy plane).
        alpha : The rotation angle about the y axis. (After being rotated about
            the z axis by theta.)
        phi : The rotation angle a about the x axis. (After being rotated by
            theta and alpha.)
        inverse : (boolean) If True, preform the inverse rotation.

    Returns:
    --------
        rotated_xyz : An Nx3 array of points rotated into the new coordinate
            system.
    """
    xyz = np.atleast_2d(xyz)
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]])
    Ry = np.array([[np.cos(phi), 0, np.sin(phi)],
                   [0, 1, 0],
                   [-np.sin(phi), 0, np.cos(phi)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha), np.cos(alpha)]])
    rot = Rz.dot(Ry).dot(Rx)
    if inverse:
        rot = np.linalg.inv(rot)
    return rot.dot(xyz.T).T

class _Shear(object):
    """
    A convience class to minimize "roughness" when inverting for the slip 
    and/or shear angle that best "flattens" the give horizon by moving it along
    the given fault.
    """
    def __init__(self, fault, horizon, alpha=None, overlap_thresh=0.3):
        """
        Parameters:
        -----------
            fault : An Nx3 array of points making up the fault surface
            hor : An Mx3 array of points along the horizon surface
            alpha : A shear angle (in degrees) measured from vertical (i.e.
                vertical shear corresponds to alpha=0) through which the
                hanging wall deforms. This is constrained to lie in the
                vertical plane defined by the slip vector. If alpha is None, it
                is assumed to be given when the _Shear instance is called.
            overlap_thresh : If less than `overlap_thresh*100` percent of the
                "moved" horizon's points are within the bounds of the fault,
                the result is penalized.
        """
        self.fault, self.horizon = fault, horizon
        self.alpha = alpha

        # Tracking these for non-overlap penalty
        self.starting_metric = self.horizon[:,-1].var()
        self.overlap_thresh = overlap_thresh
        self.numpoints = horizon.shape[0]

    def __call__(self, model):
        """
        Return the misfit ("roughness") metric for a given slip and/or shear
        angle.

        Parameters:
        -----------
            model : A displacement vector of (dx, dy) or (dx, dy, alpha)
                if alpha was not specified during initialization.
        """
        if self.alpha is None:
            slip = model[:2]
            alpha = model[-1]
        else:
            slip = model
            alpha = self.alpha
        hor = inclined_shear(self.fault, self.horizon, slip, alpha)
        metric = self.metric(hor, slip)
        return metric

    def metric(self, result, slip):
        """The "roughness" of the result."""
        if len(result) > 0:
            # Variance of the elevation values
            roughness = result[:,-1].var()
        else:
            roughness = self.starting_metric

        if result.shape[0] < self.overlap_thresh * self.numpoints:
            # If we're mostly off of the fault, penalize the result. 
            # We want to make sure it's never better than the roughness
            # of the "unmoved" horizon. (Also trying to avoid "hard" edges
            # here... If we just make it 1e10, it leads to instabilities.
            var = max(self.starting_metric, roughness)
            roughness = var * (1 + np.hypot(*slip))

        return roughness


