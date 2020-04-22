import numpy as np
import os
import sys
import time
import json
import h5py
import pickle
import argparse
import platform

from scipy.optimize import linear_sum_assignment
DIVISION_EPS = 1e-10
from scipy.spatial.transform import Rotation as srot
from scipy.optimize import least_squares
import _init_paths
from global_info import global_info
from lib.data_utils import get_model_pts, get_pose, get_part_bounding_box, get_sampled_model_pts, get_test_group, get_pickle
from lib.iou_3d import iou_3d

def hungarian_matching(W_pred, I):
    # This non-tf function does not backprob gradient, only output matching indices
    # W_pred - BxNxK
    # I - BxN, may contain -1's
    # Output: matching_indices - BxK, where (b,k)th ground truth primitive is matched with (b, matching_indices[b, k])
    #   where only n_labels entries on each row have meaning. The matching does not include gt background part
    batch_size = I.shape[0]
    n_points = I.shape[1]
    n_max_labels = W_pred.shape[2]

    matching_indices = np.zeros([batch_size, n_max_labels], dtype=np.int32)
    for b in range(batch_size):
        # assuming I[b] does not have gap
        n_labels = np.max(I[b]) + 1 # this is K'
        # print('Type: ', type(n_points), type(n_max_labels))
        W = np.zeros([n_points, n_labels + 1]) # HACK: add an extra column to contain -1's
        W[np.arange(n_points), I[b]] = 1.0 # NxK'

        dot = np.sum(np.expand_dims(W, axis=2) * np.expand_dims(W_pred[b], axis=1), axis=0) # K'xK
        denominator = np.expand_dims(np.sum(W, axis=0), axis=1) + np.expand_dims(np.sum(W_pred[b], axis=0), axis=0) - dot
        cost = dot / np.maximum(denominator, DIVISION_EPS) # K'xK
        cost = cost[:n_labels, :] # remove last row, corresponding to matching gt background part

        _, col_ind = linear_sum_assignment(-cost) # want max solution
        # print('finishing linear_sum_assignment')
        matching_indices[b, :n_labels] = col_ind

    return matching_indices

def rotate_pts(source, target):
    # compute rotation between source: [N x 3], target: [N x 3]
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

def scale_pts(source, target):
    # compute scaling factor between source: [N x 3], target: [N x 3]
    pdist_s = source.reshape(source.shape[0], 1, 3) - source.reshape(1, source.shape[0], 3)
    A = np.sqrt(np.sum(pdist_s**2, 2)).reshape(-1)
    pdist_t = target.reshape(target.shape[0], 1, 3) - target.reshape(1, target.shape[0], 3)
    b = np.sqrt(np.sum(pdist_t**2, 2)).reshape(-1)
    scale = np.dot(A, b) / (np.dot(A, A)+1e-6)
    return scale

def transform_pts(source, target):
    # source: [N x 3], target: [N x 3]
    # pre-centering and compute rotation
    source_centered = source - np.mean(source, 0, keepdims=True)
    target_centered = target - np.mean(target, 0, keepdims=True)
    rotation = rotate_pts(source_centered, target_centered)

    # compute scale
#     A = np.matmul(rotation, source_centered.T).reshape(-1)
#     b = target_centered.T.reshape(-1)
#     scale = np.dot(A, b) / (np.dot(A, A)+1e-6)
    scale = scale_pts(source_centered, target_centered)

    # compute translation
    translation = np.mean(target.T-scale*np.matmul(rotation, source.T), 1)
    return rotation, scale, translation

def rot_diff_rad(rot1, rot2):
    return np.arccos( ( np.trace(np.matmul(rot1, rot2.T)) - 1 ) / 2 ) % (2*np.pi)

def rot_diff_degree(rot1, rot2):
    return rot_diff_rad(rot1, rot2) / np.pi * 180

def ransac(dataset, model_estimator, model_verifier, inlier_th, niter=10000):
    best_model = None
    best_score = -np.inf
    best_inliers = None
    for i in range(niter):
        cur_model = model_estimator(dataset)
        cur_score, cur_inliers = model_verifier(dataset, cur_model, inlier_th)
        if cur_score > best_score:
            best_model = cur_model
            best_inliers = cur_inliers
    best_model = model_estimator(dataset, best_inliers)
    return best_model, best_inliers

def single_transformation_estimator(dataset, best_inliers = None):
    # dataset: dict, fields include source, target, nsource
    if best_inliers is None:
        sample_idx = np.random.randint(dataset['nsource'], size=3)
    else:
        sample_idx = best_inliers
    rotation, scale, translation = transform_pts(dataset['source'][sample_idx,:], dataset['target'][sample_idx,:])
    strans = dict()
    strans['rotation'] = rotation
    strans['scale'] = scale
    strans['translation'] = translation
    return strans

def single_transformation_verifier(dataset, model, inlier_th):
    # dataset: dict, fields include source, target, nsource, ntarget
    # model: dict, fields include rotation, scale, translation
    res = dataset['target'].T - model['scale'] * np.matmul( model['rotation'], dataset['source'].T ) - model['translation'].reshape((3, 1))
    inliers = np.sqrt(np.sum(res**2, 0)) < inlier_th
    score = np.sum(inliers)
    return score, inliers

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

def objective_eval(params, x0, y0, x1, y1, joints, isweight=True):
    # params: [:3] R0, [3:] R1
    # x0: N x 3, y0: N x 3, x1: M x 3, y1: M x 3, R0: 1 x 3, R1: 1 x 3, joints: K x 3
    rotvec0 = params[:3].reshape((1,3))
    rotvec1 = params[3:].reshape((1,3))
    res0 = y0 - rotate_points_with_rotvec(x0, rotvec0)
    res1 = y1 - rotate_points_with_rotvec(x1, rotvec1)
    res_joint = rotate_points_with_rotvec(joints, rotvec0) - rotate_points_with_rotvec(joints, rotvec1)
    if isweight:
        res0 /= x0.shape[0]
        res1 /= x1.shape[0]
        res_joint /= joints.shape[0]
    return np.concatenate((res0, res1, res_joint), 0).ravel()

def objective_eval_r(params, x0, y0, x1, y1, joints, isweight=True, joint_type='prismatic'):
    # params: [:3] R0, [3:] R1
    # x0: N x 3, y0: N x 3, x1: M x 3, y1: M x 3, R0: 1 x 3, R1: 1 x 3, joints: K x 3
    rotvec0 = params[:3].reshape((1,3))
    rotvec1 = params[3:].reshape((1,3))
    res0 = y0 - rotate_points_with_rotvec(x0, rotvec0)
    res1 = y1 - rotate_points_with_rotvec(x1, rotvec1)
    res_R= rotvec0 - rotvec1
    if isweight:
        res0 /= x0.shape[0]
        res1 /= x1.shape[0]
    return np.concatenate((res0, res1, res_R), 0).ravel()

def objective_eval_t(params, x0, y0, x1, y1, joints, R0, R1, scale0, scale1, isweight=True):
    # params: [0:3] t0, [3:6] t1;
    # joints: K * 3
    # rotvec0, rotvec1, scale0, scale1 solved from previous steps
    R   = R0
    transvec0 = params[0:3].reshape((1, 3))
    transvec1 = params[3:6].reshape((1, 3))
    res0 = y0 - scale0 * np.matmul(x0, R0.T) - transvec0
    res1 = y1 - scale1 * np.matmul(x1, R1.T) - transvec1
    rot_u= np.matmul(joints, R.T)[0]
    delta_trans = transvec0 - transvec1
    cross_mat  = np.array([[0, -rot_u[2], rot_u[1]],
                           [rot_u[2], 0, -rot_u[0]],
                           [-rot_u[1], rot_u[0], 0]])
    res2 = np.matmul(delta_trans, cross_mat.T).reshape(1, 3)
    # np.linspace(0, 1, num = np.min((x0.shape[0], x1.shape[0]))+1 )[1:].reshape((-1, 1))
    res2 = np.ones((np.min((x0.shape[0], x1.shape[0])), 1)) * res2
    if isweight:
        res0 /= x0.shape[0]
        res1 /= x1.shape[0]
        res2 /= res2.shape[0]
    return np.concatenate((res0, res1, res2), 0).ravel()

def joint_transformation_estimator(dataset, best_inliers = None, joint_type='revolute'):
    # dataset: dict, fields include source0, target0, nsource0,
    #     source1, target1, nsource1, joint_direction
    if best_inliers is None:
        sample_idx0 = np.random.randint(dataset['nsource0'], size=3)
        sample_idx1 = np.random.randint(dataset['nsource1'], size=3)
    else:
        sample_idx0 = best_inliers[0]
        sample_idx1 = best_inliers[1]

    source0 = dataset['source0'][sample_idx0, :]
    target0 = dataset['target0'][sample_idx0, :]
    source1 = dataset['source1'][sample_idx1, :]
    target1 = dataset['target1'][sample_idx1, :]
    # prescaling and centering
    scale0 = scale_pts(source0, target0)
    scale1 = scale_pts(source1, target1)
    scale0_inv = scale_pts(target0, source0) # check if could simply take reciprocal
    scale1_inv = scale_pts(target1, source1)

    target0_scaled_centered = scale0_inv*target0
    target0_scaled_centered -= np.mean(target0_scaled_centered, 0, keepdims=True)
    source0_centered = source0 - np.mean(source0, 0, keepdims=True)

    target1_scaled_centered = scale1_inv*target1
    target1_scaled_centered -= np.mean(target1_scaled_centered, 0, keepdims=True)
    source1_centered = source1 - np.mean(source1, 0, keepdims=True)

    # joint optimization
    #     joint_points0 = np.linspace(0, 1, num = np.min((source0.shape[0], source1.shape[0]))+1 )[1:].reshape((-1, 1))*dataset['joint_direction'].reshape((1, 3))
    #     joint_points1 = np.linspace(0, 1, num = np.min((source0.shape[0], source1.shape[0]))+1 )[1:].reshape((-1, 1))*dataset['joint_direction'].reshape((1, 3))
    joint_points0 = np.ones_like(np.linspace(0, 1, num = np.min((source0.shape[0], source1.shape[0]))+1 )[1:].reshape((-1, 1)))*dataset['joint_direction'].reshape((1, 3))
    joint_points1 = np.ones_like(np.linspace(0, 1, num = np.min((source0.shape[0], source1.shape[0]))+1 )[1:].reshape((-1, 1)))*dataset['joint_direction'].reshape((1, 3))
    joint_axis    = dataset['joint_direction'].reshape((1, 3))
    #     joint_points0 = np.linspace(0, 1, num = source1.shape[0]+1 )[1:].reshape((-1, 1))*dataset['joint_direction'].reshape((1, 3))
    #     joint_points1 = np.linspace(0, 1, num = source0.shape[0]+1 )[1:].reshape((-1, 1))*dataset['joint_direction'].reshape((1, 3))
    R0 = rotate_pts(source0_centered, target0_scaled_centered)
    R1 = rotate_pts(source1_centered, target1_scaled_centered)
    rdiff0 = np.inf
    rdiff1 = np.inf
    niter  = 100
    degree_th   = 0.1
    isalternate = False
    isdirect    = False
    if not isalternate:
        rotvec0 = srot.from_dcm(R0).as_rotvec()
        rotvec1 = srot.from_dcm(R1).as_rotvec()
        # print('initialize rotvec0 vs rotvec1: \n', rotvec0, rotvec1)
        if joint_type == 'prismatic':
            res = least_squares(objective_eval_r, np.hstack((rotvec0, rotvec1)), verbose=0, ftol=1e-4, method='lm',
                            args=(source0_centered, target0_scaled_centered, source1_centered, target1_scaled_centered, joint_points0, False))
        elif joint_type == 'revolute':
            res = least_squares(objective_eval, np.hstack((rotvec0, rotvec1)), verbose=0, ftol=1e-4, method='lm',
                            args=(source0_centered, target0_scaled_centered, source1_centered, target1_scaled_centered, joint_points0, False))
        R0 = srot.from_rotvec(res.x[:3]).as_dcm()
        R1 = srot.from_rotvec(res.x[3:]).as_dcm()
    else:
        for i in range(niter):
            if rdiff0<=degree_th and rdiff1<=degree_th:
                break
            newsrc0 = np.concatenate( (source0_centered, joint_points0), 0 )
            newtgt0 = np.concatenate( (target0_scaled_centered, np.matmul( joint_points0, R1.T ) ), 0 )
            newR0 = rotate_pts( newsrc0, newtgt0 )
            rdiff0 = rot_diff_degree(R0, newR0)
            R0 = newR0

            newsrc1 = np.concatenate( (source1_centered, joint_points1), 0 )
            newtgt1 = np.concatenate( (target1_scaled_centered, np.matmul( joint_points1, R0.T ) ), 0 )
            newR1 = rotate_pts( newsrc1, newtgt1 )
            rdiff1 = rot_diff_degree(R1, newR1)
            R1 = newR1

    translation0 = np.mean(target0.T-scale0*np.matmul(R0, source0.T), 1)
    translation1 = np.mean(target1.T-scale1*np.matmul(R1, source1.T), 1)

    # if joint_type == 'prismatic': # todo best_inliers is not None and
    #     res = least_squares(objective_eval_t, np.hstack((translation0, translation1)), verbose=0, ftol=1e-4, method='lm',
    #                 args=(source0, target0, source1, target1, joint_axis, R0, R1, scale0, scale1, False))
    #     translation0 = res.x[:3]
    #     translation1 = res.x[3:]

    jtrans = dict()
    jtrans['rotation0'] = R0
    jtrans['scale0'] = scale0
    jtrans['translation0'] = translation0
    jtrans['rotation1'] = R1
    jtrans['scale1'] = scale1
    jtrans['translation1'] = translation1
    return jtrans

def joint_transformation_verifier(dataset, model, inlier_th):
    # dataset: dict, fields include source, target, nsource, ntarget
    # model: dict, fields include rotation, scale, translation
    res0 = dataset['target0'].T - model['scale0'] * np.matmul( model['rotation0'], dataset['source0'].T ) - model['translation0'].reshape((3, 1))
    inliers0 = np.sqrt(np.sum(res0**2, 0)) < inlier_th
    res1 = dataset['target1'].T - model['scale1'] * np.matmul( model['rotation1'], dataset['source1'].T ) - model['translation1'].reshape((3, 1))
    inliers1 = np.sqrt(np.sum(res1**2, 0)) < inlier_th
    score = ( np.sum(inliers0)/res0.shape[0] + np.sum(inliers1)/res1.shape[0] ) / 2
    return score, [inliers0, inliers1]

def parallel_eval_part(s_ind, e_ind, test_exp, baseline_exp, choose_threshold, num_parts, test_group, problem_ins, rts_all, file_name):
    all_rts   = {}
    mean_err  = {'baseline': [], 'nonlinear': []}
    if num_parts == 2:
        r_raw_err   = {'baseline': [[], []], 'nonlinear': [[], []]}
        t_raw_err   = {'baseline': [[], []], 'nonlinear': [[], []]}
        s_raw_err   = {'baseline': [[], []], 'nonlinear': [[], []]}
    elif num_parts == 3:
        r_raw_err   = {'baseline': [[], [], []], 'nonlinear': [[], [], []]}
        t_raw_err   = {'baseline': [[], [], []], 'nonlinear': [[], [], []]}
        s_raw_err   = {'baseline': [[], [], []], 'nonlinear': [[], [], []]}
    elif num_parts == 4:
        r_raw_err   = {'baseline': [[], [], [], []], 'nonlinear': [[], [], [], []]}
        t_raw_err   = {'baseline': [[], [], [], []], 'nonlinear': [[], [], [], []]}
        s_raw_err   = {'baseline': [[], [], [], []], 'nonlinear': [[], [], [], []]}
    print('working on ', '/work/cascades/lxiaol9/6DPOSE/results/test_pred/{}/ with {} data'.format(test_exp, len(test_group)))

    for i in range(s_ind, e_ind):
        # try:
        print('\n Checking {}th data point: {}'.format(i, test_group[i]))
        if test_group[i].split('_')[0] in problem_ins:
            continue
        basename = test_group[i].split('.')[0]
        rts_dict      = rts_all[basename]
        scale_gt = rts_dict['scale']['gt'] # list of 2, for part 0 and part 1
        rt_gt    = rts_dict['rt']['gt']    # list of 2, each is 4*4 Hom transformation mat, [:3, :3] is rotation
        nocs_err_pn   = rts_dict['nocs_err']

        fb = h5py.File('/work/cascades/lxiaol9/6DPOSE/results/test_pred/{}/{}.h5'.format(baseline_exp, basename), 'r')
        # for name in list(f.keys()):
        #     print(name, f[name].shape)

        print('using baseline part NOCS')
        nocs_pred = fb['nocs_per_point']
        nocs_gt   = fb['nocs_gt']
        mask_pred = fb['instance_per_point'][()]
        mask_gt   = fb['cls_gt'][()]
        # matching_indices = hungarian_matching(mask_pred[np.newaxis, : ,:], mask_gt[np.newaxis, :].astype(np.int32))
        # mask_pred = mask_pred[:, matching_indices[0, :]]
        cls_per_pt_pred  = np.argmax(mask_pred, axis=1)
        partidx = []
        for j in range(num_parts):
            partidx.append(np.where(cls_per_pt_pred==j)[0])

        f = h5py.File('/work/cascades/lxiaol9/6DPOSE/results/test_pred/{}/{}.h5'.format(test_exp, basename), 'r')
        joint_cls_pred = f['index_per_point'][()]
        joint_cls_pred = np.argmax(joint_cls_pred, axis=1)
        joint_idx = []
        for j in range(1, num_parts):
            joint_idx.append(np.where(joint_cls_pred==j)[0])

        mask_pred = f['instance_per_point'][()]
        cls_per_pt_pred_ours  = np.argmax(mask_pred, axis=1)
        partidx_ours = []
        for j in range(num_parts):
            partidx_ours.append(np.where(cls_per_pt_pred_ours==j)[0])

        scale_dict = {'gt': [], 'baseline': [], 'nonlinear': []}
        r_dict = {'gt': [], 'baseline': [], 'nonlinear': []}
        t_dict = {'gt': [], 'baseline': [], 'nonlinear': []}
        xyz_err = {'baseline': [], 'nonlinear': []}
        rpy_err = {'baseline': [], 'nonlinear': []}
        scale_err= {'baseline': [], 'nonlinear': []}
        jts_axis = []
        for j in range(1, num_parts):
            niter = 200
            inlier_th = choose_threshold
            source0 = nocs_pred[partidx[0], :3]
            target0 = fb['P'][partidx[0], :3]
            source1 = nocs_pred[partidx[j], 3*j:3*(j+1)]
            target1 = fb['P'][partidx[j], :3]

            jt_axis = np.median(f['joint_axis_per_point'][joint_idx[j-1], :], 0)
            print('jt_axis', jt_axis)
            jts_axis.append(jts_axis)
            dataset = dict()
            dataset['source0'] = source0
            dataset['target0'] = target0
            dataset['nsource0'] = source0.shape[0]
            dataset['source1'] = source1
            dataset['target1'] = target1
            dataset['nsource1'] = source1.shape[0]
            dataset['joint_direction'] = jt_axis

            best_model, best_inliers = ransac(dataset, joint_transformation_estimator, joint_transformation_verifier, inlier_th, niter)
            rdiff0 = rot_diff_degree(best_model['rotation0'], rt_gt[0][:3, :3])
            tdiff0 = np.linalg.norm(best_model['translation0']-rt_gt[0][:3, 3])
            sdiff0 = np.linalg.norm(best_model['scale0']-scale_gt[0][0])
            if j == 1:
                print('part0 -- rdiff: %f degree, tdiff: %f, sdiff %f, ninliers: %f, npoint: %f' % (rdiff0, tdiff0, sdiff0, np.sum(best_inliers[0]), best_inliers[0].shape[0]))
                rpy_err['nonlinear'].append(rdiff0)
                xyz_err['nonlinear'].append(tdiff0)
                scale_err['nonlinear'].append(sdiff0)
                r_raw_err['nonlinear'][0].append(rdiff0)
                t_raw_err['nonlinear'][0].append(tdiff0)
                s_raw_err['nonlinear'][0].append(sdiff0)

            rdiff1 = rot_diff_degree(best_model['rotation1'], rt_gt[j][:3, :3])
            tdiff1 = np.linalg.norm(best_model['translation1']-rt_gt[j][:3, 3])
            sdiff1 = np.linalg.norm(best_model['scale1']-scale_gt[j][0])
            print('part%d -- rdiff: %f degree, tdiff: %f, sdiff %f, ninliers: %f, npoint: %f' % (j, rdiff1, tdiff1, sdiff1, np.sum(best_inliers[1]), best_inliers[1].shape[0]))
            rpy_err['nonlinear'].append(rdiff1)
            xyz_err['nonlinear'].append(tdiff1)
            scale_err['nonlinear'].append(sdiff1)
            r_raw_err['nonlinear'][j].append(rdiff1)
            t_raw_err['nonlinear'][j].append(tdiff1)
            s_raw_err['nonlinear'][j].append(sdiff1)
            # save
            rts_dict = {}
            if j == 1:
                scale_dict['gt'].append(scale_gt[0][0])
                scale_dict['nonlinear'].append(best_model['scale0'])
                r_dict['gt'].append(rt_gt[0][:3, :3])
                r_dict['nonlinear'].append(best_model['rotation0'])
                t_dict['gt'].append(rt_gt[0][:3, 3])
                t_dict['nonlinear'].append(best_model['translation0'])

            scale_dict['gt'].append(scale_gt[j][0])
            scale_dict['nonlinear'].append(best_model['scale1'])
            r_dict['gt'].append(rt_gt[j][:3, :3])
            r_dict['nonlinear'].append(best_model['rotation1'])
            t_dict['gt'].append(rt_gt[j][:3, 3])
            t_dict['nonlinear'].append(best_model['translation1'])

        rts_dict['scale']   = scale_dict
        rts_dict['axis']    = jts_axis
        rts_dict['rotation']      = r_dict
        rts_dict['translation']   = t_dict
        rts_dict['xyz_err'] = xyz_err
        rts_dict['rpy_err'] = rpy_err
        rts_dict['scale_err'] = scale_err
        all_rts[basename]   = rts_dict
        # except:
        #     print('Something wrong happens!!')

    with open(file_name, 'wb') as f:
        pickle.dump(all_rts, f)

    for j in range(num_parts):
        r_err_base = np.array(r_raw_err['nonlinear'][j])
        t_err_base = np.array(t_raw_err['nonlinear'][j])
        t_err_base[np.where(np.isnan(t_err_base))] = 0
        print('mean rotation err of part {}: \n'.format(j), 'nonlinear: {}'.format(r_err_base.mean())) #
        print('mean translation err of part {}: \n'.format(j), 'nonlinear: {}'.format(t_err_base.mean())) #
    end_time = time.time()
    print('saving to ', file_name)

if platform.uname()[1] == 'viz1':
    my_dir       = '/home/xiaolong/ARCwork/6DPOSE'
else:
    my_dir       = '/work/cascades/lxiaol9/6DPOSE'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', default='unseen', help='which sub test set to choose')
    parser.add_argument('--nocs', default='NAOCS', help='which sub test set to choose')
    parser.add_argument('--item', default='oven', help='object category for benchmarking')
    parser.add_argument('--save', action='store_true', help='save err to pickles')
    parser.add_argument('--viz', action='store_true', help='whether to viz')
    args = parser.parse_args()

    base_path    = my_dir + '/results/test_pred'
    infos           = global_info()
    dset_info       = infos.datasets[args.item]
    num_parts       = dset_info.num_parts
    num_ins         = dset_info.num_object
    unseen_instances= dset_info.test_list
    special_ins     = dset_info.spec_list
    main_exp        = dset_info.exp
    baseline_exp    = dset_info.baseline
    test_exp        = main_exp
    choose_threshold = 0.1

    # testing
    test_h5_path    = base_path + '/{}'.format(test_exp)
    all_test_h5     = os.listdir(test_h5_path)
    test_group      = get_test_group(all_test_h5, unseen_instances, domain=args.domain, spec_instances=special_ins)

    all_bad          = []
    print('we have {} testing data for {} {}'.format(len(test_group), args.domain, args.item))

    if args.item == 'washing_machine':
        problem_ins = ['0016']
    elif args.item == 'drawer':
        problem_ins = ['45841']
    else:
        problem_ins = []
    start_time = time.time()
    rts_all = pickle.load( open('/work/cascades/lxiaol9/6DPOSE/results/test_pred/pickle/{}/{}_{}_{}_rt.pkl'.format(main_exp, args.domain, args.nocs, args.item), 'rb' ))

    directory = '/work/cascades/lxiaol9/6DPOSE/results/test_pred/pickle/{}'.format(main_exp)
    file_name = directory + '/{}_{}_{}_{}_rt_ours_{}.pkl'.format(baseline_exp, args.domain, args.nocs, args.item, choose_threshold)

    s_ind = 0
    e_ind = 10
    parallel_eval_part(s_ind, e_ind, test_exp, baseline_exp, choose_threshold, num_parts, test_group, problem_ins, rts_all, file_name)
