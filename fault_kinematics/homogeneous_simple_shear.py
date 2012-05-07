import numpy as np
import scipy.interpolate as interpolate
import scipy.optimize

def invert_slip(faultxyz, horizonxyz, alpha, guess=(0,0), return_metric=False, 
                verbose=False, overlap_thresh=0.3):
    """
    Given a fault, horizon, shear angle, and starting guess, find the offset
    vector that best flattens the horizon using Powell's method.

    Uses the variance as a metric of "flatness".  
    

    Parameters:
    -----------
        faultxyz : An Nx3 array of points making up the fault surface
        horxyz : An Mx3 array of points along the horizon surface
        alpha : A shear angle (in degrees) measured from vertical (i.e.
            vertical shear corresponds to alpha=0) through which the hanging
            wall deforms. This is constrained to lie in the vertical plane
            defined by the slip vector.
        guess : An initial displacement vector of (dx, dy).
        return_metric : If True, return the minimized "roughness".
        verbose : If True, print the slip and roughness at each iteration
        overlap_thresh : If less than `overlap_thresh*100` percent of the 
            "moved" horizon's points are within the bounds of the fault, the 
            result is penalized.

    Returns:
    --------
        slip : A sequence of `(dx, dy)` defining the offset vector that best
            flattens the horizon
        metric : (Only if return_metric is True) The resulting "flattness".
    """
    class Shear(object):
        def __init__(self, fault, horizon, alpha):
            self.fault, self.horizon = fault, horizon
            self.alpha = alpha

            # Tracking these for non-overlap penalty
            self.starting_metric = self.horizon[:,-1].var()
            self.overlap_thresh = overlap_thresh
            self.numpoints = horizon.shape[0]

        def __call__(self, slip):
            hor = inclined_shear(self.fault, self.horizon, slip, self.alpha)
            metric = self.metric(hor, slip)

            if verbose:
                print slip, metric

            return metric

        def metric(self, result, slip):
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

    func = Shear(faultxyz, horizonxyz, alpha)
    # Powell's method appears more reliable than CG for this problem...
    items = scipy.optimize.fmin_powell(func, guess, full_output=True, 
                                       disp=False)
    slip = items[0]

    if return_metric:
        return slip, items[1]
    else:
        return slip

def vertical_shear(faultxyz, horxyz, slip):
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
    mask = np.isfinite(dz)
    horxyz = horxyz[mask]

    # Update the horizon's position
    horxyz[:,:2] += [dx, dy]
    horxyz[:,-1] += dz[mask]
    return horxyz

def inclined_shear(faultxyz, horxyz, slip, alpha):
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
    moved_xyz = vertical_shear(rotated_faultxyz, rotated_horxyz, slip)

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

