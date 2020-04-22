import numpy as np
from math import pi ,sin, cos
import itertools
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# from He et. al
def get_3d_bbox(scale, shift = 0):
    """
    Input:
        scale: [3] or scalar
        shift: [3] or scalar
    Return
        bbox_3d: [3, N]

    """
    if hasattr(scale, "__iter__"):
        bbox_3d = np.array([[scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2]]) + shift
    else:
        bbox_3d = np.array([[scale / 2, +scale / 2, scale / 2],
                  [scale / 2, +scale / 2, -scale / 2],
                  [-scale / 2, +scale / 2, scale / 2],
                  [-scale / 2, +scale / 2, -scale / 2],
                  [+scale / 2, -scale / 2, scale / 2],
                  [+scale / 2, -scale / 2, -scale / 2],
                  [-scale / 2, -scale / 2, scale / 2],
                  [-scale / 2, -scale / 2, -scale / 2]]) +shift

    bbox_3d = bbox_3d.transpose()
    return bbox_3d

def pts_inside_box(pts, bbox):
    # pts: N x 3
    # bbox: 8 x 3 (-1, 1, 1), (1, 1, 1), (1, -1, 1), (-1, -1, 1), (-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)
    u1 = bbox[5, :] - bbox[4, :]
    u2 = bbox[7, :] - bbox[4, :]
    u3 = bbox[0, :] - bbox[4, :]

    up = pts - np.reshape(bbox[4, :], (1, 3))
    p1 = np.matmul(up, u1.reshape((3, 1)))
    p2 = np.matmul(up, u2.reshape((3, 1)))
    p3 = np.matmul(up, u3.reshape((3, 1)))
    p1 = np.logical_and(p1>0, p1<np.dot(u1, u1))
    p2 = np.logical_and(p2>0, p2<np.dot(u2, u2))
    p3 = np.logical_and(p3>0, p3<np.dot(u3, u3))
    return np.logical_and(np.logical_and(p1, p2), p3)

def iou_3d(bbox1, bbox2, nres=50):
    bmin = np.min(np.concatenate((bbox1, bbox2), 0), 0)
    bmax = np.max(np.concatenate((bbox1, bbox2), 0), 0)
    xs = np.linspace(bmin[0], bmax[0], nres)
    ys = np.linspace(bmin[1], bmax[1], nres)
    zs = np.linspace(bmin[2], bmax[2], nres)
    pts = np.array([x for x in itertools.product(xs, ys, zs)])
    flag1 = pts_inside_box(pts, bbox1)
    flag2 = pts_inside_box(pts, bbox2)
    intersect = np.sum(np.logical_and(flag1, flag2))
    union = np.sum(np.logical_or(flag1, flag2))
    if union==0:
        return 1
    else:
        return intersect/float(union)

def transform_coordinates_3d(coordinates, RT):
    """
    Input:
        coordinates: [3, N]
        RT: [4, 4]
    Return
        new_coordinates: [3, N]

    """
    if coordinates.shape[0] != 3 and coordinates.shape[1]==3:
        # print('transpose box channels')
        coordinates = coordinates.transpose()
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = RT @ coordinates
    new_coordinates = new_coordinates[:3, :]/new_coordinates[3, :]
    return new_coordinates

def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Input:
        coordinates: [3, N]
        intrinsics: [3, 3]
    Return
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates


def compute_RT_distances(RT_1, RT_2):
    '''
    :param RT_1: [4, 4]. homogeneous affine transformation
    :param RT_2: [4, 4]. homogeneous affine transformation
    :return: theta: angle difference of R in degree, shift: l2 difference of T in centimeter
    '''
    #print(RT_1[3, :], RT_2[3, :])
    ## make sure the last row is [0, 0, 0, 1]
    if RT_1 is None or RT_2 is None:
        return -1

    try:
        assert np.array_equal(RT_1[3, :], RT_2[3, :])
        assert np.array_equal(RT_1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(RT_1[3, :], RT_2[3, :])

    R1 = RT_1[:3, :3]/np.cbrt(np.linalg.det(RT_1[:3, :3]))
    T1 = RT_1[:3, 3]

    R2 = RT_2[:3, :3]/np.cbrt(np.linalg.det(RT_2[:3, :3]))
    T2 = RT_2[:3, 3]

    R = R1 @ R2.transpose()
    theta = np.arccos((np.trace(R) - 1)/2) * 180/np.pi
    shift = np.linalg.norm(T1-T2) * 100
    # print(theta, shift)

    if theta < 5 and shift < 5:
        return 10 - theta - shift
    else:
        return -1

def axis_diff_degree(v1, v2):
    v1 = v1.reshape(-1)
    v2 = v2.reshape(-1)
    r_diff = np.arccos(np.sum(v1*v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))) * 180 / np.pi
    # print(r_diff)
    return min(r_diff, 180-r_diff)

def rot_diff_degree(rot1, rot2):
    return rot_diff_rad(rot1, rot2) / np.pi * 180

def rot_diff_rad(rot1, rot2):
    return np.arccos( ( np.trace(np.matmul(rot1, rot2.T)) - 1 ) / 2 ) % (2*np.pi)

def rotate_points_with_rotvec(points, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def dist_between_3d_lines(p1, e1, p2, e2):
    p1 = p1.reshape(-1)
    p2 = p2.reshape(-1)
    e1 = e1.reshape(-1)
    e2 = e2.reshape(-1)
    orth_vect = np.cross(e1, e2)
    product = np.sum(orth_vect * (p1 - p2))
    dist = product / np.linalg.norm(orth_vect)

    return np.abs(dist)

def project3d(pcloud_target, projMat, height=512, width=512):
    pcloud_projected = np.dot(pcloud_target, projMat.T)
    pcloud_projected_ndc = pcloud_projected/pcloud_projected[:, 3:4]
    img_coord = (pcloud_projected_ndc[:, 0:2] + 1)/(1/256)
    print('transformed image coordinates:\n', img_coord.shape)
    u = img_coord[:, 0]
    v = img_coord[:, 1]
    u = u.astype(np.int16)
    v = v.astype(np.int16)
    v = 512 - v
    print('u0, v0:\n', u[0], v[0])
    # rgb_raw[v, u]   = 250*np.array([0, 0, 1])              #rgb_raw[u, v] +

    return u, v # x, y in cv coords


def point_3d_offset_joint(joint, point):
    """
    joint: [x, y, z] or [[x, y, z] + [rx, ry, rz]]
    point: N * 3
    """
    if len(joint) == 2:
        P0 = np.array(joint[0])
        P  = np.array(point)
        l  = np.array(joint[1]).reshape(1, 3)
        P0P= P - P0
        PP = np.dot(P0P, l.T) * l / np.linalg.norm(l)**2  - P0P
    return PP


def rotate_pts(source, target):
    '''
    func: compute rotation between source: [N x 3], target: [N x 3]
    '''
    # pre-centering
    source = source - np.mean(source, 0, keepdims=True)
    target = target - np.mean(target, 0, keepdims=True)
    M = np.matmul(target.T, source)
    U, D, Vh = np.linalg.svd(M, full_matrices=True)
    d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
    if d:
        D[-1] = -D[-1]
        U[:, -1] = -U[:, -1]
    R = np.matmul(U, Vh)
    return R


def transform_pts(source, target):
    # source: [N x 3], target: [N x 3]
    # pre-centering and compute rotation
    source_centered = source - np.mean(source, 0, keepdims=True)
    target_centered = target - np.mean(target, 0, keepdims=True)
    rotation = rotate_pts(source_centered, target_centered)

    scale = scale_pts(source_centered, target_centered)

    # compute translation
    translation = np.mean(target.T-scale*np.matmul(rotation, source.T), 1)
    return rotation, scale, translation


def scale_pts(source, target):
    '''
    func: compute scaling factor between source: [N x 3], target: [N x 3]
    '''
    pdist_s = source.reshape(source.shape[0], 1, 3) - source.reshape(1, source.shape[0], 3)
    A = np.sqrt(np.sum(pdist_s**2, 2)).reshape(-1)
    pdist_t = target.reshape(target.shape[0], 1, 3) - target.reshape(1, target.shape[0], 3)
    b = np.sqrt(np.sum(pdist_t**2, 2)).reshape(-1)
    scale = np.dot(A, b) / (np.dot(A, A)+1e-6)
    return scale


def compute_3d_rotation_axis(pts_0, pts_1, rt, orientation=None, line_pts=None, methods='H-L', item='eyeglasses', viz=False):
    """
    pts_0: points in NOCS space of cannonical status(scaled)
    pts_1: points in camera space retrieved from depth image;
    rt: rotation + translation in 4 * 4
    """
    num_parts = len(rt)
    print('we have {} parts'.format(num_parts))

    chained_pts = [None] * num_parts
    delta_Ps = [None] * num_parts
    chained_pts[0] = np.dot( np.concatenate([ pts_0[0], np.ones((pts_0[0].shape[0], 1)) ], axis=1), rt[0].T )
    axis_list = []
    angle_list= []
    if item == 'eyeglasses':
        for j in range(1, num_parts):
            chained_pts[j] = np.dot(np.concatenate([ pts_0[j], np.ones((pts_0[j].shape[0], 1)) ], axis=1), rt[0].T)

            if methods == 'H-L':
                RandIdx = np.random.randint(chained_pts[j].shape[1], size=5)
                orient, position= estimate_joint_HL(chained_pts[j][RandIdx, 0:3], pts_1[j][RandIdx, 0:3])
                joint_axis = {}
                joint_axis['orient']   = orient
                joint_axis['position'] = position
                source_offset_arr= point_3d_offset_joint([position.reshape(1, 3), orient], chained_pts[j][RandIdx, 0:3])
                rotated_offset_arr= point_3d_offset_joint([position.reshape(1, 3), orient.reshape(1, 3)], pts_1[j][RandIdx, 0:3])
                angle = []
                for m in range(RandIdx.shape[0]):
                    modulus_0 = np.linalg.norm(source_offset_arr[m, :])
                    modulus_1 = np.linalg.norm(rotated_offset_arr[m, :])
                    cos_angle = np.dot(source_offset_arr[m, :].reshape(1, 3), rotated_offset_arr[m, :].reshape(3, 1))/(modulus_0 * modulus_1)
                    angle_per_pair = np.arccos(cos_angle)
                    angle.append(angle_per_pair)
                print('angle per pair from multiple pairs: {}', angle)
                angle_list.append(sum(angle)/len(angle))

            axis_list.append(joint_axis)
            angle_list.append(angle)

    return axis_list, angle_list

def point_rotate_about_axis(pts, anchor, unitvec, theta):
    a, b, c = anchor.reshape(3)
    u, v, w = unitvec.reshape(3)
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    ss =  u*x + v*y + w*z
    x_rotated = (a*(v**2 + w**2) - u*(b*v + c*w - ss)) * (1 - cos(theta)) + x * cos(theta) + (-c*v + b*w - w*y + v*z) * sin(theta)
    y_rotated = (b*(u**2 + w**2) - v*(a*u + c*w - ss)) * (1 - cos(theta)) + y * cos(theta) + (c*u - a*w + w*x - u*z) * sin(theta)
    z_rotated = (c*(u**2 + v**2) - w*(a*u + b*v - ss)) * (1 - cos(theta)) + z * cos(theta) + (-b*u + a*v - v*x + u*y) * sin(theta)
    rotated_pts = np.zeros_like(pts)
    rotated_pts[:, 0] = x_rotated
    rotated_pts[:, 1] = y_rotated
    rotated_pts[:, 2] = z_rotated

    return rotated_pts

def estimate_joint_HL(source_pts, rotated_pts):
    # estimate offsets
    delta_P = rotated_pts - source_pts
    assert delta_P.shape[1] == 3, 'points coordinates dimension is wrong, current is {}'.format(delta_P.shape)
    mid_pts = (source_pts + rotated_pts)/2
    CC      = np.zeros((3, 3), dtype=np.float32)
    BB      = np.zeros((delta_P.shape[0], 1), dtype=np.float32)
    for j in range(0, delta_P.shape[0]):
        CC += np.dot(delta_P[j, :].reshape(3, 1), delta_P[j, :].reshape(1, 3))
        BB[j] = np.dot(delta_P[j, :].reshape(1, 3), mid_pts[j, :].reshape((3, 1)))
    w, v   = np.linalg.eig(CC)
    print('eigen vectors are: \n', v)
    print('eigne values are: \n', w)
    orient = v[:, np.argmin(np.squeeze(w))].reshape(3, 1)

    # we already decouple the orient & position
    mat_1 = np.linalg.pinv( np.dot(delta_P.T, delta_P) )

    position = np.dot( np.dot(mat_1, delta_P.T), BB)
    print('orient has shape {}, position has shape {}'.format(orient.shape, position.shape))

    return orient, position


if __name__ == '__main__':
    #>>>>>>>>> 3D IOU compuatation
    from scipy.spatial.transform import Rotation
    bbox1 = np.array([[-1, 1, 1], [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, -1], [1, 1, -1], [1, -1, -1], [-1, -1, -1]])
    print('bbox1.shape: ', bbox1.shape)
    rotmatrix = Rotation.from_rotvec(np.pi/4 * np.array([np.sqrt(3)/3, np.sqrt(3)/3, np.sqrt(3)/3])).as_dcm()
    bbox2 = np.matmul(bbox1, rotmatrix.T)
    bbox3 = bbox1 + np.array([[1, 0, 0]])
    rotmatrix2 = Rotation.from_rotvec(np.pi/4 * np.array([0, 0, 1])).as_dcm()
    bbox4 = np.matmul(bbox1, rotmatrix2.T)
    bbox5 = bbox1 + np.array([[2, 0, 0]])
    print(iou_3d(bbox1, bbox1))
    print(iou_3d(bbox1, bbox2))
    print(iou_3d(bbox1, bbox3))
    print(iou_3d(bbox1, bbox4))
    print(iou_3d(bbox1, bbox5))
    #>>>>>>>>> test for joint parameters fitting 
    source_pts  = np.array([[5, 1, 5], [0, 0, 1], [0.5,0.5,0.5], [2, 0, 1], [3, 3, 5]])
    p1 = np.array([0,0,0])
    p2 = np.array([1,1,1])
    unitvec = (p2 - p1) / np.linalg.norm(p2 - p1)
    anchor  = p1
    rotated_pts = point_rotate_about_axis(source_pts, anchor, unitvec, pi)
    joint_axis, position = estimate_joint_HL(source_pts, rotated_pts)
    print(joint_axis, position)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(source_pts[:, 0], source_pts[:, 1], source_pts[:, 2], c='r',
               marker='o', label='source pts')
    ax.scatter(rotated_pts[:, 0], rotated_pts[:, 1], rotated_pts[:, 2], c='b',
               marker='o', label='rotated pts')
    linepts = unitvec * np.mgrid[-5:5:2j][:, np.newaxis] + np.array(p1).reshape(1, 3)
    ax.plot3D(*linepts.T, linewidth=5, c='green')
    ax.legend(loc='lower left')
    plt.show()
