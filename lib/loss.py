import os, sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..'))
from lib.tf_wrapper import batched_gather
from constants import DIVISION_EPS

import tensorflow as tf
import keras
import keras.backend as K
import numpy as np
from scipy.optimize import linear_sum_assignment

def hungarian_matching(cost, n_instance_gt):
    # cost is BxNxM
    B, N, M = cost.shape
    matching_indices = np.zeros([B, N], dtype=np.int32)
    for b in range(B):
        # limit to first n_instance_gt[b]
        _, matching_indices[b, :n_instance_gt[b]] = linear_sum_assignment(cost[b, :n_instance_gt[b], :])
    return matching_indices
    
def aggregate_loss_from_stacked(loss_stacked, T_gt):
    # loss_stacked - BxKxT, T_gt - BxK
    # out[b, k] = loss_stacked[b, k, T_gt[b, k]]
    B = tf.shape(loss_stacked)[0]
    K = tf.shape(loss_stacked)[1]
    indices_0 = tf.tile(tf.expand_dims(tf.range(B), axis=1), multiples=[1, K]) # BxK
    indices_1 = tf.tile(tf.expand_dims(tf.range(K), axis=0), multiples=[B, 1]) # BxK
    indices = tf.stack([indices_0, indices_1, T_gt], axis=2) # BxKx3
    return tf.gather_nd(loss_stacked, indices=indices)

def aggregate_per_point_loss_from_stacked(loss_stacked, T_gt):
    # loss_stacked - BxKxN'xT, T_gt - BxK
    # out[b, k, n'] = loss_stacked[b, k, n', T_gt[b, k]]
    B = tf.shape(loss_stacked)[0]
    K = tf.shape(loss_stacked)[1]
    N_p = tf.shape(loss_stacked)[2]

    indices_0 = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(B), axis=1), axis=2), multiples=[1, K, N_p]) # BxKxN'
    indices_1 = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(K), axis=0), axis=2), multiples=[B, 1, N_p]) # BxKxN'
    indices_2 = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(N_p), axis=0), axis=0), multiples=[B, K, 1]) # BxKxN'
    indices_3 = tf.tile(tf.expand_dims(T_gt, axis=2), multiples=[1, 1, N_p]) # BxKxN'
    indices = tf.stack([indices_0, indices_1, indices_2, indices_3], axis=3)
    return tf.gather_nd(loss_stacked, indices=indices) # BxKxN'

def reduce_mean_masked_part(loss, mask_gt):
    # loss: BxK
    loss = tf.where(mask_gt, loss, tf.zeros_like(loss))
    reduced_loss = tf.reduce_sum(loss, axis=1) # B
    denom = tf.reduce_sum(tf.to_float(mask_gt), axis=1) # B
    return tf.where(denom > 0, reduced_loss / denom, tf.zeros_like(reduced_loss)) # B

def compute_nocs_loss(nocs, nocs_gt, confidence, \
                        num_parts=2, mask_array=None, \
                        TYPE_L='L2', MULTI_HEAD=False, \
                        SELF_SU=False):
    # nocs, nocs_gt: BxNx3
    # Assume nocss are unoriented * L1
    if MULTI_HEAD:
        loss_nocs   = 0
        nocs_splits = tf.split(nocs, num_or_size_splits=num_parts, axis=2)
        mask_splits = tf.split(mask_array, num_or_size_splits=num_parts, axis=2)
        for i in range(num_parts):
            diff_l2 = tf.norm(nocs_splits[i] - nocs_gt, axis=2) # BxN
            diff_abs= tf.reduce_sum(tf.abs(nocs_splits[i] - nocs_gt), axis=2)
            if not SELF_SU:
                if TYPE_L=='Soft_L1':
                    loss_nocs += tf.reduce_mean(mask_splits[i][:, :, 0]  * smooth_l1_diff(diff_l2), axis=1)
                elif TYPE_L=='L2':
                    loss_nocs += tf.reduce_mean(mask_splits[i][:, :, 0]  * diff_l2, axis=1)
                else:
                    loss_nocs += tf.reduce_mean(mask_splits[i][:, :, 0]  * diff_abs, axis=1)
            else:
                if TYPE_L=='Soft_L1':
                    loss_nocs += tf.reduce_mean(mask_splits[i][:, :, 0]  * smooth_l1_diff(diff_l2) * confidence[:, :, 0] , axis=1) # B
                elif TYPE_L=='L2':
                    loss_nocs += tf.reduce_mean(mask_splits[i][:, :, 0]  * diff_l2  * confidence[:, :, 0], axis=1)
                else:
                    loss_nocs += tf.reduce_mean(mask_splits[i][:, :, 0]  * diff_abs * confidence[:, :, 0], axis=1)

            if SELF_SU:
                loss_nocs += - 0.1 * tf.reduce_mean(tf.log(confidence[:, :, 0]), axis=1)
        return loss_nocs

    else:
        diff_l2 = tf.norm(nocs - nocs_gt, axis=2) # BxN
        diff_abs= tf.reduce_sum(tf.abs(nocs - nocs_gt), axis=2) # BxN
        if not SELF_SU:
            if TYPE_L=='L2':
                return tf.reduce_mean(diff_l2, axis=1) # B
            elif TYPE_L=='Soft_L1':
                return tf.reduce_mean(smooth_l1_diff(diff_l2), axis=1) # B
            else:
                return tf.reduce_mean(diff_abs, axis=1) # B
        else:
            if TYPE_L=='L2':
                return tf.reduce_mean(diff_l2 * confidence[:, :, 0] - 0.1 * tf.log(confidence[:, :, 0]), axis=1) # B
            elif TYPE_L=='Soft_L1':
                return tf.reduce_mean(confidence[:, :, 0] * smooth_l1_diff(diff_l2) - 0.1 * tf.log(confidence[:, :, 0]), axis=1) # B
            else:
                return tf.reduce_mean(confidence[:, :, 0] * diff_abs - 0.1 * tf.log(confidence[:, :, 0]) , axis=1) # B

def compute_vect_loss(vect, vect_gt, confidence=None, \
                        num_parts=2, mask_array=None, \
                        TYPE_L='L2', MULTI_HEAD=False, \
                        SELF_SU=False):
    # nocs, nocs_gt: BxNx3
    # Assume nocss are unoriented * L1
    if MULTI_HEAD:
        loss_vect   = 0
        vect_splits = tf.split(vect, num_or_size_splits=num_parts, axis=2)
        mask_splits = tf.split(mask_array, num_or_size_splits=num_parts, axis=2)
        for i in range(num_parts):
            diff_l2 = tf.norm(vect_splits[i] - vect_gt, axis=2) # BxN
            diff_abs= tf.reduce_sum(tf.abs(vect_splits[i] - vect_gt), axis=2)
            if not SELF_SU:
                if TYPE_L=='Soft_L1':
                    loss_vect += tf.reduce_mean(mask_splits[i][:, :, 0]  * smooth_l1_diff(diff_l2), axis=1)
                elif TYPE_L=='L2':
                    loss_vect += tf.reduce_mean(mask_splits[i][:, :, 0]  * diff_l2, axis=1)
                else:
                    loss_vect += tf.reduce_mean(mask_splits[i][:, :, 0]  * diff_abs, axis=1)
            else:
                if TYPE_L=='Soft_L1':
                    loss_vect += tf.reduce_mean(mask_splits[i][:, :, 0]  * smooth_l1_diff(diff_l2) * confidence[:, :, 0] , axis=1) # B
                elif TYPE_L=='L2':
                    loss_vect += tf.reduce_mean(mask_splits[i][:, :, 0]  * diff_l2  * confidence[:, :, 0], axis=1)
                else:
                    loss_vect += tf.reduce_mean(mask_splits[i][:, :, 0]  * diff_abs * confidence[:, :, 0], axis=1)

            if SELF_SU:
                loss_vect += - 0.01 * tf.reduce_mean(tf.log(confidence[:, :, 0]), axis=1)
        return loss_vect

    else:
        if vect.shape[2]==1:
            vect = tf.squeeze(vect, axis=2)
            if confidence is not None:
                diff_l2 = tf.abs(vect - vect_gt) * confidence # BxN
                diff_abs= tf.abs(vect - vect_gt) * confidence # BxN
            else:
                diff_l2 = tf.abs(vect - vect_gt) * confidence # BxN
                diff_abs= tf.abs(vect - vect_gt) * confidence # BxN
        else:
            if confidence is not None:
                diff_l2 = tf.norm(vect - vect_gt, axis=2) * confidence # BxN
                diff_abs= tf.reduce_sum(tf.abs(vect - vect_gt), axis=2) * confidence # BxN
            else:
                diff_l2 = tf.norm(vect - vect_gt, axis=2) # BxN
                diff_abs= tf.reduce_sum(tf.abs(vect - vect_gt), axis=2) # BxN

        if not SELF_SU:
            if TYPE_L=='L2':
                return tf.reduce_mean(diff_l2, axis=1) # B
            elif TYPE_L=='Soft_L1':
                return tf.reduce_mean(smooth_l1_diff(diff_l2), axis=1) # B
            else:
                return tf.reduce_mean(diff_abs, axis=1) # B
        else:
            if TYPE_L=='L2':
                return tf.reduce_mean(diff_l2 * confidence[:, :, 0] - 0.01 * tf.log(confidence[:, :, 0]), axis=1) # B
            elif TYPE_L=='Soft_L1':
                return tf.reduce_mean(confidence[:, :, 0] * smooth_l1_diff(diff_l2) - 0.01 * tf.log(confidence[:, :, 0]), axis=1) # B
            else:
                return tf.reduce_mean(confidence[:, :, 0] * diff_abs - 0.01 * tf.log(confidence[:, :, 0]) , axis=1) # B


def compute_miou_loss(W, I_gt, matching_indices=None):
    # W - BxNxK
    # I_gt - BxN
    if matching_indices is not None:
        W_reordered = batched_gather(W, indices=matching_indices, axis=2) # BxNxK
    else:
        W_reordered = W
    depth = tf.shape(W)[2]
    # notice in tf.one_hot, -1 will result in a zero row, which is what we want
    W_gt = tf.one_hot(I_gt, depth=depth, dtype=tf.float32) # BxNxK
    dot = tf.reduce_sum(W_gt * W_reordered, axis=1) # BxK
    denominator = tf.reduce_sum(W_gt, axis=1) + tf.reduce_sum(W_reordered, axis=1) - dot
    mIoU = dot / (denominator + DIVISION_EPS) # BxK
    return 1.0 - mIoU

def compute_per_point_type_loss(per_point_type, I_gt, T_gt, is_eval):
    # For training, per_point_type is BxNxQ, where Q = n_registered_primitives
    # For test, per_point_type is BxN
    # I_gt - BxN, allow -1
    # T_gt - BxK
    batch_size= tf.shape(I_gt)[0]
    n_points  = tf.shape(I_gt)[1]
    indices_0 = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, n_points]) # BxN
    indices = tf.stack([indices_0, tf.maximum(0, I_gt)], axis=2)
    per_point_type_gt = tf.gather_nd(T_gt, indices=indices) # BxN
    if is_eval:
        type_loss = 1.0 - tf.to_float(tf.equal(per_point_type, per_point_type_gt))
    else:
        type_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=per_point_type, labels=per_point_type_gt) # BxN

    # do not add loss to background points in gt
    type_loss = tf.where(tf.equal(I_gt, -1), tf.zeros_like(type_loss), type_loss)
    return tf.reduce_sum(type_loss, axis=1) / tf.to_float(tf.count_nonzero(tf.not_equal(I_gt, -1), axis=1)) # B

def compute_joint_residual_loss(all_results, joint_params_gt, line_space='orthogonal'):
    """
    all_results: [[[B, 3], [B, 3], [B, 1]], [[B, 3], [B, 3], [B, 1]]]
    joint_params_gt: [N, 2, 7]

    """

    axis_loss = []
    orth_loss = []
    if line_space == 'orthogonal':
        dist_loss = []
    for i, joint_param in enumerate(all_results):
        axis_loss.append(tf.norm(joint_param[0] - joint_params_gt[:, i, 0:3], axis=1))
        orth_loss.append(tf.norm(joint_param[1] - joint_params_gt[:, i, 3:6], axis=1))
        if line_space == 'orthogonal':
            dist_loss.append(tf.squeeze( tf.abs( joint_param[2] - joint_params_gt[:, i, 6:7] ), axis=1 ))
    axis_loss_avg = tf.reduce_mean(tf.stack(axis_loss, axis=1), axis=1)
    orth_loss_avg = tf.reduce_mean(tf.stack(orth_loss, axis=1), axis=1)

    fitter_loss = {}
    fitter_loss['axis_loss'] = axis_loss_avg
    fitter_loss['orth_loss'] = orth_loss_avg
    if line_space == 'orthogonal':
        dist_loss_avg = tf.reduce_mean(tf.stack(dist_loss, axis=1), axis=1)
        fitter_loss['dist_loss'] = dist_loss_avg

    return fitter_loss

def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typicallly: [N, 4], but could be any shape.
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1-less_than_one) * (diff - 0.5)

    return loss

def smooth_l1_diff(diff, threshold = 0.1):
    coefficient = 1 / (2 * threshold)
    #coefficient = tf.Print(coefficient, [coefficient], message='coefficient', summarize=15)

    less_than_threshold = K.cast(K.less(diff, threshold), "float32")
    #less_than_threshold = tf.Print(less_than_threshold, [less_than_threshold], message='less_than_threshold', summarize=15)

    loss = (less_than_threshold * coefficient * diff ** 2) + (1 - less_than_threshold) * (diff - threshold / 2)
    #loss = tf.Print(loss, [loss], message='loss',
    #                              summarize=15)

    return loss
