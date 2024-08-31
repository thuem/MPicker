# Copyright (C) 2024  Xiaofeng Yan, Shudong Li
# Xueming Li Lab, Tsinghua University

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import numpy.linalg as la
from scipy.spatial import KDTree
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
from scipy.special import ellipeinc
import warnings


class my_ellipse(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        angs = np.linspace(0, 2*np.pi, 1000)
        arcs = self.ang2arc(angs)
        self.p = arcs[-1]
        self.invfun = interp1d(arcs, angs, kind='quadratic')

    def ang2arc(self, ang):
        """ang is a number or Array"""
        m = (self.b**2 - self.a**2) / self.b**2
        return self.b * ellipeinc(ang, m)
    
    def arc2ang(self, arc):
        """arc is a number or Array"""
        if type(arc) in (float, int):
            arc = np.array(arc, dtype=float)
            isnumber = True
        else:
            isnumber = False
        ang = 2 * np.pi * (arc // self.p) + self.invfun(arc % self.p)
        if isnumber:
            ang = float(ang)
        return ang
    
    def dist2dd(self, dist, ang):
        d0 = np.sqrt((self.a * np.cos(ang))**2 + (self.b * np.sin(ang))**2)
        return dist - d0
    
    def dd2dist(self, dd, ang):
        d0 = np.sqrt((self.a * np.cos(ang))**2 + (self.b * np.sin(ang))**2)
        return d0 + dd
    
    def ang_par2real(self, ang):
        x = self.a * np.cos(ang)
        y = self.b * np.sin(ang)
        return np.arctan2(y, x)
    
    def ang_real2par(self, ang):
        tan_ang = self.a / self.b * np.tan(ang)
        cos_ang = np.sqrt(1 / (1 + tan_ang**2)) * np.sign(np.cos(ang))
        sin_ang = np.sqrt(tan_ang**2 / (1 + tan_ang**2)) * np.sign(np.sin(ang))
        return np.arctan2(sin_ang, cos_ang)


def ellip_parameters(coefficients):
    """Edit from library lsq-ellipse, https://github.com/bdhammel/least-squares-ellipse-fitting, ellipse.py
    
    Returns the definition of the fitted ellipse as localized parameters

    Returns
    _______
    center : tuple
        (x0, y0)
    width : float
        semimajor axis
    height : float
        semiminor axis
    phi : float
        The counterclockwise angle [radians] of rotation from the x-axis to the semimajor axis
    """

    # Eigenvectors are the coefficients of an ellipse in general form
    # the division by 2 is required to account for a slight difference in
    # the equations between (*) and (**)
    # a*x^2 +   b*x*y + c*y^2 +   d*x +   e*y + f = 0  (*)  Eqn 1
    # a*x^2 + 2*b*x*y + c*y^2 + 2*d*x + 2*f*y + g = 0  (**) Eqn 15
    # We'll use (**) to follow their documentation
    a = coefficients[0]
    b = coefficients[1] / 2.
    c = coefficients[2]
    d = coefficients[3] / 2.
    f = coefficients[4] / 2.
    g = coefficients[5]

    # Finding center of ellipse [eqn.19 and 20] from (**)
    x0 = (c*d - b*f) / (b**2 - a*c)
    y0 = (a*f - b*d) / (b**2 - a*c)
    center = (x0, y0)

    # Find the semi-axes lengths [eqn. 21 and 22] from (**)
    numerator = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    denominator1 = (b**2 - a*c) * ( np.sqrt((a-c)**2+4*b**2) - (c+a))  # noqa: E201
    denominator2 = (b**2 - a*c) * (-np.sqrt((a-c)**2+4*b**2) - (c+a))
    height = np.sqrt(numerator / denominator1)
    width = np.sqrt(numerator / denominator2)

    # Angle of counterclockwise rotation of major-axis of ellipse to x-axis
    # [eqn. 23] from (**)
    # w/ trig identity eqn 9 form (***)
    if b == 0 and a > c:
        phi = 0.0
    elif b == 0 and a < c:
        phi = np.pi/2
    elif b != 0 and a > c:
        phi = 0.5 * np.arctan(2*b/(a-c))
    elif b != 0 and a < c:
        phi = 0.5 * (np.pi + np.arctan(2*b/(a-c)))
    elif a == c:
        phi = 0.0
    else:
        raise RuntimeError("Unreachable")

    return center, width, height, phi


def fit_ellip(dataxy):
    """Edit from library lsq-ellipse, https://github.com/bdhammel/least-squares-ellipse-fitting, ellipse.py
    
    Fit the data

    Parameters
    ----------
    X : array, shape (n_points, 2)
        Data values for the x-y data pairs to fit

    Returns
    -------
    coefficients, distances
    """

    # extract x-y pairs
    x, y = dataxy.T

    # Quadratic part of design matrix [eqn. 15] from (*)
    D1 = np.vstack([x**2, x*y, y**2]).T
    # Linear part of design matrix [eqn. 16] from (*)
    D2 = np.vstack([x, y, np.ones_like(x)]).T

    # Forming scatter matrix [eqn. 17] from (*)
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2

    # Constraint matrix [eqn. 18]
    C1 = np.array([[0., 0., 2.], [0., -1., 0.], [2., 0., 0.]])

    # Reduced scatter matrix [eqn. 29]
    M = la.inv(C1) @ (S1 - S2 @ la.inv(S3) @ S2.T)

    # M*|a b c >=l|a b c >. Find eigenvalues and eigenvectors from this
    # equation [eqn. 28]
    eigval, eigvec = np.linalg.eig(M)

    # Eigenvector must meet constraint 4ac - b^2 to be valid.
    cond = 4*np.multiply(eigvec[0, :], eigvec[2, :]) - np.power(eigvec[1, :], 2)
    a1 = eigvec[:, np.nonzero(cond > 0)[0]]

    # |d f g> = -S3^(-1) * S2^(T)*|a b c> [eqn. 24]
    a2 = la.inv(-S3) @ S2.T @ a1

    # Eigenvectors |a b c d f g>
    # list of the coefficients describing an ellipse [a,b,c,d,e,f]
    # corresponding to ax**2 + bxy + cy**2 + dx + ey + f from (*)
    coef = np.vstack([a1, a2]).flatten()

    # algebra distance
    a, b, c, d, e, f = coef
    coef = coef / np.sqrt((4*a*c - b**2)) # the paper says 4ac - b^2 should be 1 ??
    a, b, c, d, e, f = coef
    err = a*x**2 + b*x*y + c*y**2 + d*x + e*y + f
    return coef, err


def pad_linear(x):
    y_shape, x_shape = x.shape
    x_pad=np.zeros((y_shape+2, x_shape+2))
    x_pad[1:-1, 1:-1]=x.copy()
    x_pad[0,:]=2*x_pad[1,:]-x_pad[2,:]
    x_pad[-1,:]=2*x_pad[-2,:]-x_pad[-3,:]
    x_pad[:,0]=2*x_pad[:,1]-x_pad[:,2]
    x_pad[:,-1]=2*x_pad[:,-2]-x_pad[:,-3]
    return x_pad


def numerical_diff_uv(x_mgrid, y_mgrid, z_mgrid):
    """given 3 2dArray xyz mgrids, return norm vector zyx"""
    x, y, z = x_mgrid, y_mgrid, z_mgrid

    x_pad = pad_linear(x)
    y_pad = pad_linear(y)
    z_pad = pad_linear(z)
    dxdu = (x_pad[1:-1, 2:] - x_pad[1:-1, 0:-2]) / 2
    dydu = (y_pad[1:-1, 2:] - y_pad[1:-1, 0:-2]) / 2
    dzdu = (z_pad[1:-1, 2:] - z_pad[1:-1, 0:-2]) / 2
    dxdv = (x_pad[2:, 1:-1] - x_pad[0:-2, 1:-1]) / 2
    dydv = (y_pad[2:, 1:-1] - y_pad[0:-2, 1:-1]) / 2
    dzdv = (z_pad[2:, 1:-1] - z_pad[0:-2, 1:-1]) / 2

    # z=z(u,v), y=z(u,v), x=z(u,v) -> vz,vy,vx = [dx/du*dy/dv-dy/du*dx/dv,dz/du*dx/dv-dx/du*dz/dv,dy/du*dz/dv-dz/du*dy/dv]
    v_zyx = np.array([dxdu*dydv-dydu*dxdv, dzdu*dxdv-dxdu*dzdv, dydu*dzdv-dzdu*dydv])
    v_zyx = v_zyx / np.sqrt(v_zyx[0]**2 + v_zyx[1]**2 + v_zyx[2]**2)

    return v_zyx


def fibonacci_sample(N, half=True):
    # from mpicker_core
    gold=(np.sqrt(5)-1)/2
    if half:
        N*=2
        n=np.arange(int(N/2),N+1)
    else:
        n=np.arange(1,N+1)
    theta=np.arccos((2*n-1)/N - 1)
    phi=2*np.pi*gold*n
    return theta, phi


def down_simple(coords, dist=2.9, xy=False):
    if xy:
        coords = coords[:,:2] # 2d
    tree = KDTree(coords)
    pick_idx = np.ones(len(coords), dtype=bool)
    for i in range(len(coords)):
        if pick_idx[i]:
            pick_idx[tree.query_ball_point(coords[i], dist)] = False
            pick_idx[i] = True # dist from itself is always 0
    return pick_idx


def fit_cylinder_ellipse(dataxyz, num=400):
    "input dataxyz with shape (n, 3)"
    if len(dataxyz) > 100000:
        dataxyz = dataxyz[down_simple(dataxyz)]
    def fit_2d(angles):
        rot_matrix = Rotation.from_euler('ZY', angles).as_matrix().T
        x, y, z = np.dot(rot_matrix, dataxyz.T) # projection
        coeff, dist = fit_ellip(np.array([x,y]).T)
        center, width, height, angle = ellip_parameters(coeff)
        dist = dist / np.sqrt((width**2 + height**2) / 2) # Seems big ellipse will give bigger distance value ??
        return coeff, dist
    lsq_fun = lambda ang: fit_2d(ang)[1]
    # global search
    thetas, phis = fibonacci_sample(num)
    errs = np.zeros(len(thetas))
    for i in range(len(thetas)):
        angles = [phis[i], thetas[i]]
        err = lsq_fun(angles)
        errs[i] = sum(err**2)
    idx_best = np.argmin(errs)
    rmsd0 = np.sqrt(errs[idx_best] / len(dataxyz))
    # local search
    initial_angles = [phis[idx_best], thetas[idx_best]]
    try:
        res = least_squares(lsq_fun, initial_angles)
        angles = res.x
        coeff, err = fit_2d(angles)
        rmsd = np.sqrt( sum(err**2) / len(dataxyz) )
        if rmsd > rmsd0:
            raise Exception()
    except:
        warnings.warn("fine ellipse cylinder fitting failed.")
        coeff, err = fit_2d(initial_angles)
        rmsd = np.sqrt( sum(err**2) / len(dataxyz) )
    print("ellipse cylinder fitting rmsd:", rmsd)

    phi, theta = angles
    theta, phi = theta % (2 * np.pi), phi % (2 * np.pi)
    center, width, height, angle = ellip_parameters(coeff)
    return theta, phi, center, width, height, angle


def convert2ellipzyx(coords):
    "input datazyx with shape (n, 3), convert to new coordinate"
    dataxyz = coords[:, ::-1]
    theta, phi, center, width, height, angle = fit_cylinder_ellipse(dataxyz)
    cx, cy = center
    print("theta, phi, cx, cy, a, b, angle:")
    print(theta, phi, cx, cy, width, height, angle)
    ellipse = my_ellipse(width, height)
    rot_matrix = Rotation.from_euler('ZY', [phi, theta]).as_matrix().T
    x, y, z = np.dot(rot_matrix, dataxyz.T)
    x = x - cx
    y = y - cy
    ang_real = np.arctan2(y, x) - angle
    ang_par = ellipse.ang_real2par(ang_real)
    ang_par = ang_par % (2 * np.pi) # to [0, 2pi)
    arc = ellipse.ang2arc(ang_par)
    dist = np.sqrt(x**2 + y**2)
    dd = ellipse.dist2dd(dist, ang_par)
    arc, dd = -arc, -dd # make sure the positive norm vector is pointing to the outside
    new_x, new_y, new_z = z, arc, dd
    cylinder_par = (theta, phi, cx, cy, width, height, angle)
    return np.array([new_z, new_y, new_x]).T, cylinder_par


def convertback_ellipzyx(mgridzyx, cylinder_par):
    """input mgridzyx with shape (3,ny,nx) and cylinder_par,\n
       convert to real mgridzyx and return norm vector"""
    theta, phi, cx, cy, width, height, angle = cylinder_par
    _, shapey, shapex = mgridzyx.shape
    # back to xyz after rotate
    ellipse = my_ellipse(width, height)
    dd, arc, z = mgridzyx
    arc, dd = -arc, -dd # make sure the positive norm vector is pointing to the outside
    ang_par = ellipse.arc2ang(arc)
    dist = ellipse.dd2dist(dd, ang_par)
    ang_real = ellipse.ang_par2real(ang_par)
    x = dist * np.cos(ang_real + angle) + cx
    x = x.flatten()
    y = dist * np.sin(ang_real + angle) + cy
    y = y.flatten()
    z = z.flatten()
    # back to xyz before rotate
    rot_matrix = Rotation.from_euler('ZY', [phi, theta]).as_matrix() # no .T here
    x, y, z = np.dot(rot_matrix, np.array([x, y, z]))
    # norm vector
    mgridx = x.reshape((shapey, shapex))
    mgridy = y.reshape((shapey, shapex))
    mgridz = z.reshape((shapey, shapex))
    mgrid_new = np.array([mgridz, mgridy, mgridx])
    vector = numerical_diff_uv(mgridx, mgridy, mgridz)
    return mgrid_new, vector


def draw_cylinder(cylinder_par, shapezyx):
    """return cylinder coords in zyx"""
    theta, phi, cx, cy, width, height, angle = cylinder_par
    nz, ny, nx = shapezyx
    ellipse = my_ellipse(width, height)
    rot_matrix = Rotation.from_euler('ZY', [phi, theta]).as_matrix()

    x0, y0, z0 = rot_matrix @ np.array([cx,cy,0])
    x1, y1, z1 = rot_matrix @ np.array([cx,cy,1])
    k1, k2 = -10 * (nx + ny + nz), 10 * (nx + ny + nz) # big enough numbers
    def update_k(xyz0, xyz1, nxyz, k1, k2):
        # 0 < x0+k*(x1-x0) < nx
        if xyz1 != xyz0:
            kxyz1, kxyz2 = -xyz0/(xyz1-xyz0), (nxyz-xyz0)/(xyz1-xyz0)
            kxyz1, kxyz2 = min(kxyz1, kxyz2), max(kxyz1, kxyz2)
            k1, k2 = max(k1, kxyz1), min(k2, kxyz2)
        return k1, k2
    k1, k2 = update_k(x0, x1, nx, k1, k2)
    k1, k2 = update_k(y0, y1, ny, k1, k2)
    k1, k2 = update_k(z0, z1, nz, k1, k2)
    if k2 - k1 < 100:
        k1, k2 = k1 - 20, k2 + 20

    arc = np.linspace(0.05*ellipse.p, 0.95*ellipse.p, int(0.9*ellipse.p) + 1)
    ang = ellipse.arc2ang(arc)
    z = np.arange(k1, k2, dtype=float)
    ang, z = np.meshgrid(ang, z)
    dist = ellipse.dd2dist(np.zeros_like(ang), ang)
    ang_real = ellipse.ang_par2real(ang)
    x = dist * np.cos(ang_real + angle) + cx
    y = dist * np.sin(ang_real + angle) + cy
    x, y, z = np.dot(rot_matrix, np.array([x.flatten(), y.flatten(), z.flatten()]))
    return np.array([z, y, x]).T

    
    
