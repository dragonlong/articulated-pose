'''
Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation
RANSAC for Similarity Transformation Estimation

Written by Srinath Sridhar
'''

import numpy as np
import cv2
import itertools
import _init_paths
from lib.vis_utils import plot3d_pts, plot_arrows
from lib.data_utils import get_pickle, load_pickle
from lib.transformations import quaternion_matrix, quaternion_from_matrix
import matplotlib.pyplot as plt

def estimateSimilarityTransform(source: np.array, target: np.array, rt_pre=None, verbose=False):
    nIter = 100
    # [4, N], [4, N]
    SourceHom, TargetHom, TargetNorm, SourceNorm, RatioTS, RatioST, PassT, StopT = set_config(source, target, verbose)
    inliers, transform_results = getRANSACInliers(SourceHom, TargetHom, rt_pre=rt_pre, MaxIterations=nIter, PassThreshold=PassT, StopThreshold=StopT)
    SourceInliersHom, TargetInliersHom, BestInlierRatio = inliers
    # Scales, Rotation, Translation, OutTransform = transform_results
    if(BestInlierRatio < 0.1):
        print('[ WARN ] - Something is wrong. Small BestInlierRatio: ', BestInlierRatio)
        return None, None, None, None

    Scales, Rotation, Translation, OutTransform = estimateSimilarityUmeyama(SourceInliersHom, TargetInliersHom, rt_pre=rt_pre)

    if verbose:
        print('BestInlierRatio:', BestInlierRatio)
    return Scales, Rotation, Translation, OutTransform

def estimateSimilarityTransformCoords(source: np.array, target: np.array, source1=None, target1=None, joints=None, rt_ref=[None, None], rt_pre=[None, None],\
            viz=False, viz_ransac=False, viz_sample=False, viz_normal=False, use_jt_pts=False, eval_rts=False, use_ext_rot=False, verbose=False, index=0):
    nIter = 100
    # [4, N], [4, N]
    SourceHom, TargetHom, TargetNorm, SourceNorm, RatioTS, RatioST, PassT, StopT = set_config(source, target, verbose)
    SourceHom1, TargetHom1, TargetNorm1, SourceNorm1, RatioTS1, RatioST1, PassT1, StopT1 = set_config(source1, target1, verbose)

    # 1. find inliers
    inliers, records = getRANSACInliersCoords(SourceHom, TargetHom, SourceHom1, TargetHom1, joints=joints, rt_ref=rt_ref, rt_pre=rt_pre, \
                     MaxIterations=nIter, PassThreshold=[PassT, PassT1], StopThreshold=[StopT, StopT1], \
                     viz=viz, viz_ransac=viz_ransac, viz_sample=viz_sample, viz_normal=viz_normal, use_jt_pts=use_jt_pts, eval_rts=eval_rts, use_ext_rot=use_ext_rot, verbose=verbose)

    SourceInliersHom, TargetInliersHom, BestInlierRatio0, SourceInliersHom1, TargetInliersHom1, BestInlierRatio1 = inliers
    ang_dis_list, inliers_ratio, select_index = records

    if(BestInlierRatio0 < 0.05) or (BestInlierRatio1 < 0.05):
        print('[ WARN ] - Something is wrong. Small BestInlierRatio: ', [BestInlierRatio0, BestInlierRatio1])
        return None, None, None, None

    # 2. further use inlier points and joints to decide the final pose
    position, joint_axis, joint_pts = get_joint_features(joints)
    assert joint_pts.shape[0] == 4
    Scale, Rotations, Translations, OutTransforms = estimateSimilarityUmeyamaCoords(SourceInliersHom, TargetInliersHom, SourceInliersHom1, TargetInliersHom1, joint_axis, rt_ref=rt_ref, joint_pts=joint_pts, \
         viz=viz, viz_ransac=viz_ransac, viz_sample=viz_sample, use_jt_pts=use_jt_pts, use_ext_rot=use_ext_rot, verbose=verbose)

    if verbose:
        print('BestInlierRatio:', BestInlierRatio0)

    if viz_ransac:
        fig = plt.figure(dpi=200)
        for j in range(2):
            q_gt          = quaternion_from_matrix(rt_ref[j][:3, :3])
            q_iter        = quaternion_from_matrix(Rotations[j].T)
            ang_dis       = 2 * np.arccos(sum(q_iter * q_gt)) * 180 / np.pi
            if ang_dis > 180:
                ang_dis = 360 - ang_dis
            ax = plt.subplot(1, 2, j+1)
            plt.plot(range(len(ang_dis_list[j])), ang_dis_list[j], label='rotation err')
            plt.plot(range(len(inliers_ratio[j])), inliers_ratio[j], label='inliers ratio')
            plt.plot([select_index[j]], [ang_dis_list[j][select_index[j]]], 'bo')
            plt.plot([select_index[0]], [ang_dis_list[j][select_index[0]]], 'ro')
            plt.plot([select_index[j]], [ang_dis], 'yo', label='final rotation error')
            plt.xlabel('Ransac sampling order')
            plt.ylabel('value')
            ax.text(0.55, 0.80, 'Select {0}th inliers with {1:0.4f} rotation error'.format(select_index[j], ang_dis_list[j][select_index[j]]), transform=ax.transAxes, color='blue', fontsize=6)
            plt.grid(True)
            plt.legend()
            plt.title('part {}'.format(j))
        plt.show()
        save_path = '/home/lxiaol9/Downloads/ARCwork/6DPOSE/results/test_pred/images'
        fig.savefig('{}/{}_{}.png'.format(save_path, index, 'coord_descent' ), pad_inches=0)

    return Scale, Rotations, Translations, OutTransforms

def set_config(source, target, verbose):
    SourceHom = np.transpose(np.hstack([source, np.ones([source.shape[0], 1])]))
    TargetHom = np.transpose(np.hstack([target, np.ones([target.shape[0], 1])]))

    # Auto-parameter selection based on source-target heuristics
    TargetNorm = np.mean(np.linalg.norm(target, axis=1))
    SourceNorm = np.mean(np.linalg.norm(source, axis=1))
    RatioTS = (TargetNorm / SourceNorm)
    RatioST = (SourceNorm / TargetNorm)
    PassT = RatioST if(RatioST>RatioTS) else RatioTS
    StopT = PassT / 100
    # if verbose:
    #     print('Pass threshold: ', PassT)
    #     print('Stop threshold: ', StopT)

    return SourceHom, TargetHom, TargetNorm, SourceNorm, RatioTS, RatioST, PassT, StopT

def compose_rt(rotation, translation):
    aligned_RT = np.zeros((4, 4), dtype=np.float32)
    aligned_RT[:3, :3] = rotation.transpose()
    aligned_RT[:3, 3]  = translation
    aligned_RT[3, 3]   = 1
    return aligned_RT

def getRANSACInliersCoords(SourceHom0, TargetHom0, \
                    SourceHom1, TargetHom1, joints=None, rt_ref=[None, None], rt_pre=[None, None], MaxIterations=100, PassThreshold=[200, 200], StopThreshold=[1, 1], \
                       viz=False, viz_ransac=False, viz_sample=False, viz_normal=False, verbose=False, \
                       use_jt_pts=False, use_ext_rot=False, \
                       eval_rts=False):
    """
    joints: [position, axis, pts]
            position: [1, 3]
            axis : 3
            pts  : [N, 3]
    """
    BestResidual0 = 1e10
    BestResidual1 = 1e10
    BestInlierRatio0 = 0
    BestInlierRatio1 = 0
    BestInlierIdx0 = np.arange(SourceHom0.shape[1])
    BestInlierIdx1 = np.arange(SourceHom1.shape[1])

    # if viz_ransac: # todo
    #     plot3d_pts([[SourceHom0[:3].transpose(), SourceHom1[:3].transpose(), TargetHom0[:3].transpose(), TargetHom1[:3].transpose()]], [['source0', 'source1', 'target0', 'target1']], s=5, title_name=['points to ransac'], color_channel=None, save_fig=False, sub_name='default')

    position, joint_axis, joint_pts = get_joint_features(joints)
    assert joint_pts.shape[0] == 4
    ang_dis_list = [[], []]
    inliers_ratio= [[], []]
    select_index = [0] * 2
    for i in range(0, MaxIterations):
        if i >5:
            verbose = False
        RandIdx0 = np.random.randint(SourceHom0.shape[1], size=5)
        RandIdx1 = np.random.randint(SourceHom1.shape[1], size=5)

        scale, Rs, Ts, OutTrans = estimateSimilarityUmeyamaCoords(SourceHom0[:, RandIdx0], TargetHom0[:, RandIdx0],\
                         SourceHom1[:, RandIdx1], TargetHom1[:, RandIdx1], joint_axis, joint_pts=joint_pts, rt_ref=rt_ref, rt_pre=rt_pre, \
                         viz=viz, viz_ransac=viz_ransac, viz_sample=viz_sample, use_jt_pts=use_jt_pts, use_ext_rot=use_ext_rot, verbose=verbose, index=i+1)

        # evaluate per part pts
        if eval_rts:
            # print('evaluating inliers using rts for pair 0')
            Residual0, InlierRatio0, InlierIdx0 = evaluateModel(OutTrans[0], SourceHom0, TargetHom0, PassThreshold[0])
        else:
            Residual0, InlierRatio0, InlierIdx0 = evaluateModelRotation(Rs[0].T, SourceHom0, TargetHom0, 0.05 * PassThreshold[0], rt_ref=rt_ref[0], viz_normal=viz_normal)

        # if Residual0 < BestResidual0: # todo
        # if InlierRatio0 > BestInlierRatio0 and Residual0 < BestResidual0:

        if eval_rts:
            # print('evaluating inliers using rts for pair 1')
            Residual1, InlierRatio1, InlierIdx1 = evaluateModel(OutTrans[1], SourceHom1, TargetHom1, PassThreshold[1])
        else:
            Residual1, InlierRatio1, InlierIdx1 = evaluateModelRotation(Rs[1].T, SourceHom1, TargetHom1, 0.05 * PassThreshold[1],  rt_ref=rt_ref[1], viz_normal=viz_normal)

        if viz_ransac:
            inliers_ratio[0].append(InlierRatio0)
            inliers_ratio[1].append(InlierRatio1)
            for j in range(2):
                q_gt          = quaternion_from_matrix(rt_ref[j][:3, :3])
                q_iter        = quaternion_from_matrix(Rs[j].T)
                ang_dis       = 2 * np.arccos(sum(q_iter * q_gt)) * 180 / np.pi
                if ang_dis > 180:
                    ang_dis = 360 - ang_dis
                ang_dis_list[j].append(ang_dis)

        if InlierRatio0 > BestInlierRatio0:
            select_index[0] = i
            BestResidual0 = Residual0
            BestInlierRatio0 = InlierRatio0
            BestInlierIdx0   = InlierIdx0

        # if Residual1 < BestResidual1: # todo
        # if InlierRatio1 > BestInlierRatio1 and Residual1 < BestResidual1:
        if InlierRatio1 > BestInlierRatio1:
            select_index[1]  = i
            BestResidual1    = Residual1
            BestInlierRatio1 = InlierRatio1
            BestInlierIdx1   = InlierIdx1
        # print('Iteration: ', i, '\n Residual: ', [Residual0, Residual1], 'Inlier ratio: ', [InlierRatio0, InlierRatio1])

        if BestResidual0 < StopThreshold[0] and BestResidual1 < StopThreshold[1]:
            break

    # if viz_ransac:
    #     fig = plt.figure(dpi=200)
    #     for j in range(2):
    #         ax = plt.subplot(1, 2, j+1)
    #         plt.plot(range(len(ang_dis_list[j])), ang_dis_list[j], label='rotation err')
    #         plt.plot(range(len(inliers_ratio[j])), inliers_ratio[j], label='inliers ratio')
    #         plt.plot([select_index[j]], [ang_dis_list[j][select_index[j]]], 'bo')
    #         plt.plot([select_index[0]], [ang_dis_list[j][select_index[0]]], 'ro')
    #         plt.xlabel('Ransac sampling order')
    #         plt.ylabel('value')
    #         ax.text(0.55, 0.80, 'Select {0}th inliers with {1:0.4f} rotation error'.format(select_index[j], ang_dis_list[j][select_index[j]]), transform=ax.transAxes, color='blue', fontsize=6)
    #         plt.grid(True)
    #         plt.legend()
    #         plt.title('part {}'.format(j))
    #     plt.show()
    inliers = [SourceHom0[:, BestInlierIdx0], TargetHom0[:, BestInlierIdx0], BestInlierRatio0, SourceHom1[:, BestInlierIdx1], TargetHom1[:, BestInlierIdx1], BestInlierRatio1]

    return inliers, [ang_dis_list, inliers_ratio, select_index]


# compute transform after every sampling
def estimateSimilarityUmeyamaCoords(SourceHom0, TargetHom0, SourceHom1, TargetHom1, joint_axis, joint_pts=None, rt_ref=[None, None], rt_pre=[None, None], \
                 viz=False, viz_ransac=False, viz_sample=False, use_jt_pts=False, use_ext_rot=False, verbose=False, index=0):
    """
    SourceHom0: [4, 5]
    joint_pts : [4, 5]
    joint_axis: [4, 1]
    """
    U, D0, Vh  = svd_pts(SourceHom0, TargetHom0) #
    R0        = np.matmul(U, Vh).T # Transpose is the one that works
    U, D1, Vh  = svd_pts(SourceHom1, TargetHom1) #
    R1        = np.matmul(U, Vh).T #
    # begin EM
    max_iter = 100
    # max_iter = 1 # todo
    StopThreshold = 2 * np.cos(0.5/180*np.pi)
    if viz_sample:
        plot3d_pts([[SourceHom0[:3].transpose(), SourceHom1[:3].transpose(), TargetHom0[:3].transpose(), TargetHom1[:3].transpose(), joint_pts[:3].transpose()]], [['source0', 'source1', 'target0', 'target1', 'joint_points']], s=100, title_name=['sampled points'], color_channel=None, save_fig=False, sub_name='default')
    joint_axis_tiled0 = np.tile(joint_axis, (1, int(SourceHom0.shape[1]/5)))
    joint_axis_tiled1 = np.tile(joint_axis, (1, int(SourceHom1.shape[1]/5)))
    # joint_axis_tiled0 = np.tile(joint_axis, (1, int(SourceHom0.shape[1])))
    # joint_axis_tiled1 = np.tile(joint_axis, (1, int(SourceHom1.shape[1])))
    if use_ext_rot and rt_pre[0] is not None:
        # print('using external rotation')
        R0 = rt_pre[0][:3, :3].T
        R1 = rt_pre[1][:3, :3].T
    else:
        r_list = [[R0], [R1]]
        for i in range(max_iter):
            rotated_axis = np.matmul(R0.T, joint_axis_tiled1[:3]) # [3, 1]
            U, D1, Vh    = svd_pts(SourceHom1, TargetHom1, joint_axis_tiled1, rotated_axis, viz_sample=viz_sample, index=2*i)
            R1_new       = np.matmul(U, Vh).T
            rotated_axis = np.matmul(R1_new.T, joint_axis_tiled0[:3])
            U, D0, Vh    = svd_pts(SourceHom0, TargetHom0, joint_axis_tiled0, rotated_axis, viz_sample=viz_sample, index=2*i + 1)
            R0_new       = np.matmul(U, Vh).T
            eigen_sum0   = np.trace(np.matmul(R0_new.T, R0)) -1
            eigen_sum1   = np.trace(np.matmul(R1_new.T, R1)) -1
            R0 = R0_new
            R1 = R1_new
            r_list[0].append(R0)
            r_list[1].append(R1)
            if eigen_sum0 > StopThreshold and eigen_sum1 > StopThreshold:
                # if verbose:
                #     print('Algorithm converges at {}th iteration for Coordinate Descent'.format(i))
                break
    if viz_ransac and index<10:# and SourceHom0.shape[1]>5:
        ang_dis_list = [[], []]
        for j in range(2):
            q_gt          = quaternion_from_matrix(rt_ref[j][:3, :3])
            for rot_iter in r_list[j]:
                q_iter        = quaternion_from_matrix(rot_iter.T)
                ang_dis       = 2 * np.arccos(sum(q_iter * q_gt)) * 180 / np.pi
                if ang_dis > 180:
                    ang_dis = 360 - ang_dis
                ang_dis_list[j].append(ang_dis)
        fig = plt.figure(dpi=200)
        for j in range(2):
            ax = plt.subplot(1, 2, j+1)
            plt.plot(range(len(ang_dis_list[j])), ang_dis_list[j])
            plt.xlabel('iteration')
            plt.ylabel('rotation error')
            plt.title('{}th sampling part {}'.format(index, j))
        plt.show()
    Rs = [R0, R1]

    if use_jt_pts:
        if viz_sample:
            plot3d_pts([[SourceHom0[:3].transpose(), SourceHom1[:3].transpose(), TargetHom0[:3].transpose(), TargetHom1[:3].transpose(), joint_pts[:3].transpose()]], [['source0', 'source1', 'target0', 'target1', 'joint_points']], s=100, title_name=['sampled points'], color_channel=None, save_fig=False, sub_name='default')
        final_scale, Ts, OutTrans = compute_scale_translation([SourceHom0,  SourceHom1], [TargetHom0, TargetHom1], Rs, joint_pts)
        if verbose:
            print("scale by adding joints are \n: {}".format(final_scale))
    else:
        if viz_sample:
            plot3d_pts([[SourceHom0[:3].transpose(), SourceHom1[:3].transpose(), TargetHom0[:3].transpose(), TargetHom1[:3].transpose()]], [['source0', 'source1', 'target0', 'target1']], s=100, title_name=['points after sampling'], color_channel=None, save_fig=False, sub_name='default')
        final_scale0, T0, OutTrans0 = est_ST(SourceHom0, TargetHom0, D0, Rs[0])
        final_scale1, T1, OutTrans1 = est_ST(SourceHom1, TargetHom1, D1, Rs[1])
        final_scale = [final_scale0, final_scale1]
        Ts = [T0, T1]
        OutTrans = [OutTrans0, OutTrans1]
        if verbose:
            print("scale by direct solving per part are \n: {}".format(final_scale))

    return final_scale, Rs, Ts, OutTrans

def svd_pts(SourceHom, TargetHom, raw_axis=None, rotated_axis=None, viz_sample=False, index=0):
        # Copy of original paper is at: http://web.stanford.edu/class/cs273/refs/umeyama.pdf
    SourceCentroid = np.mean(SourceHom[:3, :], axis=1)
    TargetCentroid = np.mean(TargetHom[:3, :], axis=1)
    nPoints = SourceHom.shape[1]

    # pre-centering
    CenteredSource = SourceHom[:3, :] - np.tile(SourceCentroid, (nPoints, 1)).transpose()
    CenteredTarget = TargetHom[:3, :] - np.tile(TargetCentroid, (nPoints, 1)).transpose()

    # pre-scaling
    stdSource  = np.sqrt(np.var(CenteredSource[:3, :], axis=1).sum())
    stdTarget  = np.sqrt(np.var(CenteredTarget[:3, :], axis=1).sum())
    try:
        CenteredSource = CenteredSource/stdSource
        CenteredTarget = CenteredTarget/stdTarget
    except:
        CenteredSource = CenteredSource
        CenteredTarget = CenteredTarget
    origin = np.zeros((1, 3))
    if viz_sample:
        if raw_axis is not None:
            # raw_axis[:3, 0:1].transpose(), rotated_axis[:3, 0:1].transpose(), 'axis', 'rotated axis',
            plot3d_pts([[CenteredSource[:3].transpose(), CenteredTarget[:3].transpose(), raw_axis.transpose(), rotated_axis.transpose(), origin]], [['source', 'target', 'raw_axis_point', 'rotated_axis_point', 'origin']], s=100, title_name=['{}th iteration points for coords descent'.format(index)], color_channel=None, save_fig=False, sub_name='default')

    if raw_axis is not None:
        CenteredSource = np.concatenate([CenteredSource, raw_axis[:3]], axis=1)
        CenteredTarget = np.concatenate([CenteredTarget, rotated_axis[:3]], axis=1)
        nPoints = nPoints + raw_axis.shape[1]

    CovMatrix = np.matmul(CenteredTarget, np.transpose(CenteredSource)) / nPoints

    if np.isnan(CovMatrix).any():
        print('nPoints:', nPoints)
        print(SourceHom.shape)
        print(TargetHom.shape)
        return None, None, None

    U, D, Vh = np.linalg.svd(CovMatrix, full_matrices=True)
    d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
    if d:
        D[-1] = -D[-1]
        U[:, -1] = -U[:, -1]

    return U, D, Vh

def compute_scale_translation(SourceHoms, TargetHoms, Rotation, Z, verbose=False):
    """
    for global nocs only
    SourceHoms: [4, N] * n, n is the number of parts
    TargetHoms: [4, N] * n, n is the number of parts
    Rotations : [3, 3] * n
    Z: [4, N] or [4, 1], points sitting on
    """
    num_parts     = len(SourceHoms)
    num_joint_pt  = 5
    # # implement 1
    # offsetsTarget = TargetHoms[1][:3] - TargetHoms[0][:3]# [3,]
    # offsetsRotate = np.matmul(Rotation[0].T, Z[:3, 0:num_joint_pt]) + np.matmul(Rotation[1].T, SourceHoms[1][:3]) \
    #      - np.matmul(Rotation[0].T, SourceHoms[0][:3]) - np.matmul(Rotation[1].T, Z[:3, 0:num_joint_pt])  # [3,]
    # if normalize together, we might be able to find better scale

    # scale = np.sqrt(offsetsTarget**2/offsetsRotate**2).mean()
    # scale = np.sqrt(np.abs(offsetsTarget).mean()/np.abs(offsetsRotate).mean())

    # # implement 2
    # varP = np.var(SourceHom[:3, :], axis=1).sum()
    # varT = np.var(TargetHom[:3, :], axis=1).sum()
    # ScaleFact = np.sqrt(varT/varP)
    # Scales = np.array([ScaleFact, ScaleFact, ScaleFact])
    # ScaleMatrix = np.diag(Scales)
    # final_scale = scale
    # Scales      = np.array([final_scale, final_scale, final_scale])
    # ScaleMatrix = np.diag(Scales)
    # for i in range(num_parts):
    #     Translation = TargetHoms[i][:3] - final_scale * np.matmul(Rotation[i].T,  SourceHoms[i][:3])
    #     Ts[i] = Translation.mean(axis=1)
    #     OutTransform = np.identity(4)
    #     OutTransform[:3, :3] = ScaleMatrix @ Rotation[i].T
    #     OutTransform[:3, 3] = Ts[i]
    #     OutTrans[i]  = OutTransform
    # implementation 3
    # sR x + T = y

    centered_Source= [ None ] * num_parts
    centered_Target= [ None ] * num_parts
    OutTrans = [None] * num_parts

    for i in range(num_parts):
        SourceCentroid = np.mean(SourceHoms[i][:3, :], axis=1)
        TargetCentroid = np.mean(TargetHoms[i][:3, :], axis=1)
        nPoints = SourceHoms[i].shape[1]

        # pre-centering
        CenteredSource = SourceHoms[i][:3, :] - np.tile(SourceCentroid, (nPoints, 1)).transpose()
        CenteredTarget = TargetHoms[i][:3, :] - np.tile(TargetCentroid, (nPoints, 1)).transpose()

        centered_Source[i] = np.matmul(Rotation[0].T, CenteredSource[i][:3]) # [4, N]
        centered_Target[i] = CenteredTarget

    SourceHoms_concat = np.concatenate(centered_Source, axis=1)
    TargetHoms_concat = np.concatenate(centered_Target, axis=1)

    varP = np.var(SourceHoms_concat[:3, :], axis=1).sum()
    varT = np.var(TargetHoms_concat[:3, :], axis=1).sum()
    ScaleFact = np.sqrt(varT/varP) # even it is aligned very well, we still believe only the normal is reliable
    Scales = np.array([ScaleFact, ScaleFact, ScaleFact])
    ScaleMatrix = np.diag(Scales)

    # assume we only have two parts and one joint
    A = np.zeros((SourceHoms_concat.shape[1]+ num_joint_pt, num_parts)) # [N, 3]
    Translations = [None] * num_parts
    Z_offsets    = [None] * (num_parts - 1)
    for i in range(num_parts):
        A[SourceHoms[i].shape[1], i] = 1
        Translations[i] = TargetHoms[i][:3] - final_scale * np.matmul(Rotation[i].T,  SourceHoms[i][:3])
        Translations[i].reshape(-1, 1)
    A[-(num_joint_pt):, :] = np.array([-1, 1]).reshape(1, -1)
    if verbose:
        print('A is : \n', A)

    for i in range(num_parts - 1):
        Z_offsets[i] = ScaleFact * np.matmul(Rotation[i].T,  Z[:3, :num_joint_pt]) - ScaleFact * np.matmul(Rotation[i+1].T,  Z[:3, :num_joint_pt])

    B = np.concatenate(Translations + Z_offsets, axis=0) # [N, 3]

    # solve T1, ..., Tn with least-square, Ax = B, A: [N*2], B[N, 3]
    Ts = np.linalg.lstsq(A, B) # return 2, 3
    for i in range(num_parts):
        OutTransform = np.identity(4)
        OutTransform[:3, :3] = ScaleMatrix @ Rotation[i].T
        OutTransform[:3, 3] = Ts[i]
        OutTrans[i]  = OutTransform
    final_scale = [ScaleFact, ScaleFact] # for both parts

    return final_scale, Ts, OutTrans

def estimateRestrictedAffineTransform(source: np.array, target: np.array, verbose=False):
    SourceHom = np.transpose(np.hstack([source, np.ones([source.shape[0], 1])]))
    TargetHom = np.transpose(np.hstack([target, np.ones([source.shape[0], 1])]))

    RetVal, AffineTrans, Inliers = cv2.estimateAffine3D(source, target)
    # We assume no shear in the affine matrix and decompose into rotation, non-uniform scales, and translation
    Translation = AffineTrans[:3, 3]
    NUScaleRotMat = AffineTrans[:3, :3]
    # NUScaleRotMat should be the matrix SR, where S is a diagonal scale matrix and R is the rotation matrix (equivalently RS)
    # Let us do the SVD of NUScaleRotMat to obtain R1*S*R2 and then R = R1 * R2
    R1, ScalesSorted, R2 = np.linalg.svd(NUScaleRotMat, full_matrices=True)

    if verbose:
        print('-----------------------------------------------------------------------')
    # Now, the scales are sort in ascending order which is painful because we don't know the x, y, z scales
    # Let's figure that out by evaluating all 6 possible permutations of the scales
    ScalePermutations = list(itertools.permutations(ScalesSorted))
    MinResidual = 1e8
    Scales = ScalePermutations[0]
    OutTransform = np.identity(4)
    Rotation = np.identity(3)
    for ScaleCand in ScalePermutations:
        CurrScale = np.asarray(ScaleCand)
        CurrTransform = np.identity(4)
        CurrRotation = (np.diag(1 / CurrScale) @ NUScaleRotMat).transpose()
        CurrTransform[:3, :3] = np.diag(CurrScale) @ CurrRotation
        CurrTransform[:3, 3] = Translation
        # Residual = evaluateModel(CurrTransform, SourceHom, TargetHom)
        Residual = evaluateModelNonHom(source, target, CurrScale,CurrRotation, Translation)
        if verbose:
            # print('CurrTransform:\n', CurrTransform)
            print('CurrScale:', CurrScale)
            print('Residual:', Residual)
            print('AltRes:', evaluateModelNoThresh(CurrTransform, SourceHom, TargetHom))
        if Residual < MinResidual:
            MinResidual = Residual
            Scales = CurrScale
            Rotation = CurrRotation
            OutTransform = CurrTransform

    if verbose:
        print('Best Scale:', Scales)

    if verbose:
        print('Affine Scales:', Scales)
        print('Affine Translation:', Translation)
        print('Affine Rotation:\n', Rotation)
        print('-----------------------------------------------------------------------')

    return Scales, Rotation, Translation, OutTransform

def getRANSACInliers(SourceHom, TargetHom, rt_pre=None, MaxIterations=100, PassThreshold=200, StopThreshold=1):
    BestResidual = 1e10
    BestInlierRatio = 0
    BestInlierIdx = np.arange(SourceHom.shape[1])
    for i in range(0, MaxIterations):
        RandIdx = np.random.randint(SourceHom.shape[1], size=5)
        Scales, Rotation, Translation, OutTransform = estimateSimilarityUmeyama(SourceHom[:, RandIdx], TargetHom[:, RandIdx], rt_pre=rt_pre)
        Residual, InlierRatio, InlierIdx = evaluateModel(OutTransform, SourceHom, TargetHom, PassThreshold)
        # Residual, InlierRatio, InlierIdx = evaluateModelRotation(Rotation.T, SourceHom, TargetHom, 0.05 * PassThreshold)
        # if Residual < BestResidual:
        if InlierRatio > BestInlierRatio:
            BestResidual = Residual
            BestInlierRatio = InlierRatio
            BestInlierIdx = InlierIdx
        if BestResidual < StopThreshold:
            break
        # print('Iteration: ', i)
        # print('Residual: ', Residual)
        # print('Inlier ratio: ', InlierRatio)
    inliers = [SourceHom[:, BestInlierIdx], TargetHom[:, BestInlierIdx], BestInlierRatio]
    transform_results = [Scales, Rotation, Translation, OutTransform]

    return inliers, transform_results

def evaluateModelRotation(Rotation, SourceHom, TargetHom, PassThreshold, rt_ref=None, viz_normal=False):
    SourceCentroid = np.mean(SourceHom[:3, :], axis=1)
    TargetCentroid = np.mean(TargetHom[:3, :], axis=1)
    nPoints = SourceHom.shape[1]

    CenteredSource = SourceHom[:3, :] - np.tile(SourceCentroid, (nPoints, 1)).transpose()
    CenteredTarget = TargetHom[:3, :] - np.tile(TargetCentroid, (nPoints, 1)).transpose()

    stdSource  = np.sqrt(np.var(CenteredSource[:3, :], axis=1).sum())
    stdTarget  = np.sqrt(np.var(CenteredTarget[:3, :], axis=1).sum())

    CenteredSource = CenteredSource/stdSource
    CenteredTarget = CenteredTarget/stdTarget
    origin = np.zeros((1, 3))
    if viz_normal:
        if rt_ref is not None:
            RotatedSource = np.matmul(rt_ref[:3, :3], CenteredSource)
            plot3d_pts([[RotatedSource[:3].transpose(), CenteredTarget[:3].transpose(), origin]], [['GT rotated source', 'target', 'origin']], s=100, title_name=['normalized source and target pts'], color_channel=None, save_fig=False, sub_name='default')
        plot3d_pts([[CenteredSource[:3].transpose(), CenteredTarget[:3].transpose(), origin]], [['source', 'target', 'origin']], s=100, title_name=['normalized source and target pts'], color_channel=None, save_fig=False, sub_name='default')
    RotatedSource = np.matmul(Rotation, CenteredSource)
    if viz_normal:
        plot3d_pts([[RotatedSource[:3].transpose(), CenteredTarget[:3].transpose(), origin]], [['Pred rotated source', 'target', 'origin']], s=100, title_name=['normalized source and target pts'], color_channel=None, save_fig=False, sub_name='default')
    Diff = CenteredTarget - RotatedSource
    ResidualVec = np.linalg.norm(Diff[:3, :], axis=0)
    Residual = np.linalg.norm(ResidualVec)
    InlierIdx = np.where(ResidualVec < PassThreshold)
    nInliers = np.count_nonzero(InlierIdx)
    # Residual = np.linalg.norm(ResidualVec[InlierIdx[0]]) # todo
    InlierRatio = nInliers / SourceHom.shape[1]
    return Residual, InlierRatio, InlierIdx[0]

def evaluateModel(OutTransform, SourceHom, TargetHom, PassThreshold):
    Diff = TargetHom - np.matmul(OutTransform, SourceHom)
    ResidualVec = np.linalg.norm(Diff[:3, :], axis=0)
    Residual = np.linalg.norm(ResidualVec)
    InlierIdx = np.where(ResidualVec < PassThreshold)
    nInliers = np.count_nonzero(InlierIdx)
    InlierRatio = nInliers / SourceHom.shape[1]
    return Residual, InlierRatio, InlierIdx[0]

def evaluateModelNoThresh(OutTransform, SourceHom, TargetHom):
    Diff = TargetHom - np.matmul(OutTransform, SourceHom)
    ResidualVec = np.linalg.norm(Diff[:3, :], axis=0)
    Residual = np.linalg.norm(ResidualVec)
    return Residual

def evaluateModelNonHom(source, target, Scales, Rotation, Translation):
    RepTrans = np.tile(Translation, (source.shape[0], 1))
    TransSource = (np.diag(Scales) @ Rotation @ source.transpose() + RepTrans.transpose()).transpose()
    Diff = target - TransSource
    ResidualVec = np.linalg.norm(Diff, axis=0)
    Residual = np.linalg.norm(ResidualVec)
    return Residual

def testNonUniformScale(SourceHom, TargetHom):
    OutTransform = np.matmul(TargetHom, np.linalg.pinv(SourceHom))
    ScaledRotation = OutTransform[:3, :3]
    Translation = OutTransform[:3, 3]
    Sx = np.linalg.norm(ScaledRotation[0, :])
    Sy = np.linalg.norm(ScaledRotation[1, :])
    Sz = np.linalg.norm(ScaledRotation[2, :])
    Rotation = np.vstack([ScaledRotation[0, :] / Sx, ScaledRotation[1, :] / Sy, ScaledRotation[2, :] / Sz])
    print('Rotation matrix norm:', np.linalg.norm(Rotation))
    Scales = np.array([Sx, Sy, Sz])

    # # Check
    # Diff = TargetHom - np.matmul(OutTransform, SourceHom)
    # Residual = np.linalg.norm(Diff[:3, :], axis=0)
    return Scales, Rotation, Translation, OutTransform


def estimateSimilarityUmeyama(SourceHom, TargetHom, rt_pre=None):
    # Copy of original paper is at: http://web.stanford.edu/class/cs273/refs/umeyama.pdf
    SourceCentroid = np.mean(SourceHom[:3, :], axis=1)
    TargetCentroid = np.mean(TargetHom[:3, :], axis=1)
    nPoints = SourceHom.shape[1]

    CenteredSource = SourceHom[:3, :] - np.tile(SourceCentroid, (nPoints, 1)).transpose()
    CenteredTarget = TargetHom[:3, :] - np.tile(TargetCentroid, (nPoints, 1)).transpose()

    CovMatrix = np.matmul(CenteredTarget, np.transpose(CenteredSource)) / nPoints

    if np.isnan(CovMatrix).any():
        print('nPoints:', nPoints)
        print(SourceHom.shape)
        print(TargetHom.shape)
        raise RuntimeError('There are NANs in the input.')

    U, D, Vh = np.linalg.svd(CovMatrix, full_matrices=True)
    d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
    if d:
        D[-1] = -D[-1]
        U[:, -1] = -U[:, -1]
    if rt_pre is not None:
        # print('using external rotation')
        Rotation = rt_pre[:3, :3].T
    else:
        Rotation = np.matmul(U, Vh).T # Transpose is the one that works

    varP = np.var(SourceHom[:3, :], axis=1).sum()
    ScaleFact = 1/varP * np.sum(D) # scale factor
    Scales = np.array([ScaleFact, ScaleFact, ScaleFact])
    ScaleMatrix = np.diag(Scales)

    Translation  = TargetHom[:3, :].mean(axis=1) - SourceHom[:3, :].mean(axis=1).dot(ScaleFact*Rotation)

    OutTransform = np.identity(4)
    OutTransform[:3, :3] = ScaleMatrix @ Rotation.T
    OutTransform[:3, 3]  = Translation

    # # Check
    # Diff = TargetHom - np.matmul(OutTransform, SourceHom)
    # Residual = np.linalg.norm(Diff[:3, :], axis=0)
    return Scales, Rotation, Translation, OutTransform

def get_joint_features(joints):
    # get joint pts & axis
    lower_bound, upper_bound = -0.5, 0.5
    position       = joints[0].reshape(1, 3)
    joint_axis     = joints[1].reshape(1, 3)
    if len(joints)>2:
        joint_pts = joints[2]
    else:
        # joint_pts = position.reshape(1, 3)
        joint_pts = position + joint_axis * np.linspace(lower_bound, upper_bound, num=20).reshape(-1, 1)

    rand_idx   = np.random.randint(joint_pts.shape[0], size=5)
    joint_pts  = np.transpose(np.hstack([joint_pts[rand_idx, :3], np.ones([joint_pts[rand_idx, :3].shape[0], 1])]))
    joint_axis = np.array([joint_axis[0, 0], joint_axis[0, 1], joint_axis[0, 2], 1]).reshape(4, 1)

    return position, joint_axis, joint_pts # [1, 3], [4, 1], [4, 20]

def est_ST(SourceHom, TargetHom, D, Rotation):
    varP = np.var(SourceHom[:3, :], axis=1).sum()
    varT = np.var(TargetHom[:3, :], axis=1).sum()
    ScaleFact = np.sqrt(varT/varP)
    Scales = np.array([ScaleFact, ScaleFact, ScaleFact])
    ScaleMatrix = np.diag(Scales)

    Translation = TargetHom[:3, :].mean(axis=1) - SourceHom[:3, :].mean(axis=1).dot(ScaleFact*Rotation)

    OutTransform = np.identity(4)
    OutTransform[:3, :3] = ScaleMatrix @ Rotation.T
    OutTransform[:3, 3] = Translation

    return Scales, Translation, OutTransform

# s = (Y_2 - Y_1)/(R_1 * Z_1 + R_2 * X_2 - R_1 * X_1 - R_2 * Z_1)
# T_1 = Y_1 - s * R_1 * X_1
# T_2 = Y_2 - s * R_2 * X_2

if __name__ == '__main__':
    import argparse
    import platform
    import time
    from lib.transformations import euler_from_matrix, euler_matrix, quaternion_matrix, quaternion_from_matrix
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action='store_true', help='whether to viz')
    parser.add_argument('--verbose', action='store_true', help='whether to verbose')
    parser.add_argument('--viz_err', action='store_true', help='whether to viz error')
    parser.add_argument('--viz_joint', action='store_true', help='whether to viz joints')
    parser.add_argument('--viz_ransac', action='store_true', help='whether to viz all ransac points')
    parser.add_argument('--viz_sample', action='store_true', help='whether to viz sampled points')
    parser.add_argument('--viz_normal', action='store_true', help='whether to viz normalized points')
    args = parser.parse_args()
    if platform.uname()[1] == 'viz1':
        my_dir       = '/home/xiaolong/homeARC/3DGenNet2019/Art6D/baselines/spfn/experiments/results'
    else:
        my_dir       = '/home/lxiaol9/3DGenNet2019/Art6D/baselines/spfn/experiments/results'
    num_parts = 3
    rt_dict         = {} # gt，pred1, 2 --> [] * num_parts
    rt_dict['gt']   = [None] * num_parts
    rt_dict['pred'] = [None] * num_parts
    rt_dict['pred_it'] = [None] * num_parts # predictions from ransac-iterative

    scale_dict      = {}
    scale_dict ['gt']      = [None] * num_parts    # same for every part
    scale_dict ['pred']    = [None] * num_parts    # slightly different
    scale_dict ['pred_it'] = [None] * num_parts # slightly different for every part

    SourceHoms = [np.random.rand(4, 5), np.random.rand(4, 5)]
    TargetHoms = [np.random.rand(4, 5), np.random.rand(4, 5)]
    Rotations  = [np.ones((3, 3)), np.ones((3, 3))]
    Z          = np.ones((3, 1))

    for index in range(0, 5):
        file_name  = my_dir  + '/pickle/datapoint_{}.pkl'.format(index)
        data = load_pickle(file_name)
        # for key, value in data.items():
        #     if isinstance(value, dict):
        #         for k, v in value.items():
        #             if isinstance(v, list):
        #                 print(k, len(v), v)
        #             else:
        #                 print(k, v.shape)
        #     else:
        #         print(key, value.shape)
        tstart  = time.time()
        data_pts   = data['pts_pair']['pred'] # or we choose gt points?
        source_pts = data_pts['source0']
        target_pts = data_pts['target0']
        source_pts1= data_pts['source1']
        target_pts1= data_pts['target1']
        joints     = [data['joint']['position'], data['joint']['axis'], data['joint']['points']]
        rt_dict    = data['rt_dict']
        scale_dict = data['scale_dict']
        scales, rotation, translation, outtransform = estimateSimilarityTransform(source_pts, target_pts, source_pts1, target_pts1, joints=joints, rt_ref=rt_dict['gt'][0:2], \
                    viz=args.viz, viz_ransac=args.viz_ransac, viz_sample=args.viz_sample, viz_normal=args.viz_normal, verbose=args.verbose, index=index)

        tend    = time.time()
        if args.verbose:
            print('scales, translation for part 0 and part {} is {}, {}'.format(1, scales, translation))
            print('ransac with with coordinate descent takes {} seconds for part 0, {}'.format(tend- tstart, 1))
        aligned_RT               = compose_rt(rotation[0], translation[0])
        rt_dict['pred_it'][0]    = aligned_RT
        scale_dict['pred_it'][0] = scales
        aligned_RT               = compose_rt(rotation[1], translation[1])
        rt_dict['pred_it'][1]    = aligned_RT
        scale_dict['pred_it'][1] = scales

        # final evaluation per part
        for j in range(num_parts-1):
            q_pred        = quaternion_from_matrix(rt_dict['pred'][j][:3, :3])
            q_pred_it     = quaternion_from_matrix(rt_dict['pred_it'][j][:3, :3])
            q_gt          = quaternion_from_matrix(rt_dict['gt'][j][:3, :3])
            q_pred_list   = [q_pred, q_pred_it, q_gt]

            # # how to deal with err
            rt_pred_list  = [rt_dict['pred'][j], rt_dict['pred_it'][j]]
            methods       = ['vanilla SVD', 'coords descent']
            for m in range(2):
                ang_dis           = 2 * np.arccos(sum(q_pred_list[m] * q_gt)) * 180 / np.pi
                xyz_dis           = np.linalg.norm(rt_pred_list[m][:3, 3] - rt_dict['gt'][j][:3, 3])
                if args.verbose:
                    print('Angular distance is : {} for part {} with {}'.format(ang_dis, j, methods[m]))
                    # print('Euclid distance is : {} for part {} with {}'.format(xyz_dis, j, methods[m]))
