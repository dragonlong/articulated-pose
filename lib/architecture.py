import os, sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..'))

from pointnet_plusplus.architectures import build_pointnet2_seg, build_pointnet2_cls, build_pointnet2_shared
from lib.tf_wrapper import batched_gather
from pointnet_plusplus.utils import tf_util
from lib import loss

import tensorflow as tf
import numpy as np
from scipy.optimize import linear_sum_assignment

epsilon = 10e-7

def get_per_point_model(scope, P, n_max_parts, is_training, bn_decay, early_split=False, early_split_nocs=False, mixed_pred=False, pred_joint=False, pred_joint_ind=False):
    '''
        Inputs:
            - P: BxNx3 tensor, the input point cloud
            - K := n_max_parts
        Outputs: a dict, containing
            - W: BxNxK, segmentation instances, fractional
            - nocs_per_points: BxNx3, nocs per point
            - confi_per_points: BxNx1,
            - parameters - a dict, each entry is a BxKx... tensor, not using here
    '''
    with tf.variable_scope(scope):
        out_dims=[n_max_parts, 3*n_max_parts]
        if mixed_pred:
            out_dims.append(3)
        out_dims.append(1)
        net = build_pointnet2_shared('est_net', X=P, out_dims=out_dims, is_training=is_training, bn_decay=bn_decay)
        # early_split by default
        if early_split_nocs:
            print('Now we are using early_split_nocs')
            with tf.variable_scope('nocs_net'):
                net_results = []
                for idx, out_dim in enumerate(out_dims):
                    if idx > 0:
                        net_shared = net
                        net_shared = tf_util.conv1d(net_shared, 128, 1, padding='VALID', activation_fn=None, scope='fc11_{}'.format(idx))
                        current_result = tf_util.conv1d(net_shared, out_dim, 1, padding='VALID', activation_fn=None, scope='fc2_{}'.format(idx))
                    else:
                        current_result = tf_util.conv1d(net, out_dim, 1, padding='VALID', activation_fn=None, scope='fc2_{}'.format(idx))
                    net_results.append(current_result)
        else:
            with tf.variable_scope('nocs_net'):
                net_results = []
                for idx, out_dim in enumerate(out_dims):
                    current_result = tf_util.conv1d(net, out_dim, 1, padding='VALID', activation_fn=None, scope='fc2_{}'.format(idx))
                    net_results.append(current_result)

        if mixed_pred:
            W, nocs_per_points, gocs_per_points, confi_per_points  = net_results
        else:
            W, nocs_per_points, confi_per_points = net_results
        # by default we predict joints
        joint_axis, unitvec, heatmap, joint_cls = joint_est_model('joint_net', X=net, is_training=is_training, bn_decay=bn_decay, pred_joint_ind=pred_joint_ind)

    W = tf.nn.softmax(W, axis=2) # BxNxK # maximum
    confi_per_points = tf.nn.sigmoid(confi_per_points)
    nocs_per_points  = tf.nn.sigmoid(nocs_per_points)   # BxNx3

    heatmap = tf.nn.sigmoid(heatmap)
    unitvec = tf.nn.tanh(unitvec)
    joint_axis  = tf.nn.tanh(joint_axis)
    joint_cls = tf.nn.softmax(joint_cls, axis=2)

    pred = {
        'W': W,
        'nocs_per_point' : nocs_per_points,
        'confi_per_point': confi_per_points,
        'heatmap_per_point':  heatmap,
        'unitvec_per_point':  unitvec,
        'joint_axis_per_point':  joint_axis,
        'index_per_point'     : joint_cls,
    }

    if mixed_pred:
        pred['gocs_per_point'] = gocs_per_points

    return pred


def get_per_point_model_new(scope, P, n_max_parts, is_training, bn_decay, early_split=False, early_split_nocs=False, mixed_pred=False, pred_joint=False, pred_joint_ind=False):
    '''
        Inputs:
            - P: BxNx3 tensor, the input point cloud
            - K := n_max_parts
        Outputs: a dict, containing
            - W: BxNxK, segmentation instances, fractional
            - nocs_per_points: BxNx3, nocs per point
            - confi_per_points: BxNx1,
            - parameters - a dict, each entry is a BxKx... tensor, not using here
    '''
    with tf.variable_scope(scope):
        out_dims=[n_max_parts, 3*n_max_parts] # seg + part NOCS
        if mixed_pred:
            out_dims.append(1*n_max_parts) # scale
            out_dims.append(3*n_max_parts) # translation
        out_dims.append(1)
        net = build_pointnet2_shared('est_net', X=P, out_dims=out_dims, is_training=is_training, bn_decay=bn_decay)

        if early_split_nocs:
            print('Now we are using early_split_nocs')
            with tf.variable_scope('nocs_net'):
                net_results = []
                for idx, out_dim in enumerate(out_dims):
                    net_shared = net
                    if idx == 1:
                        net_shared = tf_util.conv1d(net_shared, 128, 1, padding='VALID', activation_fn=None, scope='fc11_{}'.format(idx))
                    current_result = tf_util.conv1d(net_shared, out_dim, 1, padding='VALID', activation_fn=None, scope='fc2_{}'.format(idx))
                    net_results.append(current_result)
        else:
            with tf.variable_scope('nocs_net'):
                net_results = []
                for idx, out_dim in enumerate(out_dims):
                    current_result = tf_util.conv1d(net, out_dim, 1, padding='VALID', activation_fn=None, scope='fc2_{}'.format(idx))
                    net_results.append(current_result)

        if mixed_pred:
            W, nocs_per_points, scale_per_points, trans_per_points, confi_per_points  = net_results
            scale_per_points = tf.nn.sigmoid(scale_per_points)
            trans_per_points = tf.nn.tanh(trans_per_points)
        else:
            W, nocs_per_points, confi_per_points = net_results

        joint_axis, unitvec, heatmap, joint_cls = joint_est_model('joint_net', X=net, is_training=is_training, bn_decay=bn_decay, pred_joint_ind=pred_joint_ind)

    W = tf.nn.softmax(W, axis=2) # BxNxK # maximum
    confi_per_points = tf.nn.sigmoid(confi_per_points)
    nocs_per_points  = tf.nn.sigmoid(nocs_per_points)   # BxNx3


    heatmap = tf.nn.sigmoid(heatmap)
    unitvec = tf.nn.tanh(unitvec)
    joint_axis  = tf.nn.tanh(joint_axis)
    joint_cls = tf.nn.softmax(joint_cls, axis=2)

    pred = {
        'W': W,
        'nocs_per_point' : nocs_per_points,
        'confi_per_point': confi_per_points,
        'heatmap_per_point':  heatmap,
        'unitvec_per_point':  unitvec,
        'joint_axis_per_point': joint_axis,
        'index_per_point'     : joint_cls
    }

    if mixed_pred:
        # scale_per_points_tiled = tf.tile(scale_per_points, [1, 1, 3*n_max_parts])
        # trans_per_points_tiled = tf.tile(trans_per_points, [1, 1, n_max_parts])
        scale_per_points_tiled = tf.reshape(tf.tile(tf.expand_dims(scale_per_points, -1), [1, 1, 1, 3]), [tf.shape(scale_per_points)[0], tf.shape(scale_per_points)[1], 3*n_max_parts])
        trans_per_points_tiled = trans_per_points   
        assert trans_per_points_tiled.get_shape().as_list()[2] ==  scale_per_points_tiled.get_shape().as_list()[2] == 3* n_max_parts, print(scale_per_points_tiled.get_shape().as_list()[2],  trans_per_points_tiled.get_shape().as_list()[2])
        pred['gocs_per_point'] = nocs_per_points  * scale_per_points_tiled + trans_per_points_tiled
        pred['global_scale']   = scale_per_points
        pred['global_translation'] = trans_per_points

    return pred

def get_direct_regression_model_baseline(scope, P, n_max_parts, gt_dict, is_training, bn_decay, line_space='orthogonal'):
    """
    P: input pts, [N, 3];
    output: [K, 7(split into 3, 3, 1)], K is the number of joints, is a kind of direct regression
    """
    # check dict keys and items
    if line_space == 'orthogonal':
        unit_param = [3, 3, 1]
    else:
        unit_param = [3, 3]

    param_dim_list  = unit_param * ( n_max_parts - 1 )
    param_pair_list = []
    reg_result = build_pointnet2_cls('direct_reg_net', point_cloud=P, out_dims=param_dim_list, is_training=is_training, bn_decay=bn_decay)
    for j in range(n_max_parts - 1):
        if line_space == 'orthogonal':
            axis, orth, dist = reg_result[j*3:(j+1) * 3]
            direct_axis     = tf.nn.tanh(axis)
            direct_orth     = tf.nn.tanh(orth)
            direct_dist     = tf.nn.sigmoid(dist)
            joint_params    = [direct_axis, direct_orth, direct_dist]
        else:
            axis, orth = reg_result[j*2: (j+1) * 2]
            direct_axis     = tf.nn.tanh(axis)
            direct_orth     = tf.nn.tanh(orth)
            joint_params    = [direct_axis, direct_orth]
        param_pair_list.append(joint_params)
    pred = {'joint_params': param_pair_list}

    return pred

# what would best architecture for joints property?
def joint_est_model(scope, X, is_training, bn_decay, n_max_parts=3, pred_joint_ind=False):
    layer_dims = [128, 128]
    with tf.variable_scope(scope):
        for j, dim in enumerate(layer_dims):
            X = tf_util.conv1d(X, dim, 1, padding='VALID', bn=True,
                is_training=is_training, scope='fc3_{}'.format(j), bn_decay=bn_decay)
            X = tf_util.dropout(X, keep_prob=0.5, is_training=is_training,
                scope='dp1')
        joint_axis  = tf_util.conv1d(X, 3, 1, padding='VALID', activation_fn=None, scope='fc4_0') # default is relu, as we have extra activation function
        univect = tf_util.conv1d(X, 3, 1, padding='VALID', activation_fn=None, scope='fc4_1')
        heatmap = tf_util.conv1d(X, 1, 1, padding='VALID', activation_fn=None, scope='fc4_2')
        joint_cls = tf_util.conv1d(X, n_max_parts, 1, padding='VALID', activation_fn=None, scope='fc4_3')

    return joint_axis, univect, heatmap, joint_cls

def get_batch_norm_decay(global_step, batch_size, bn_decay_step):
    BN_INIT_DECAY = 0.5
    BN_DECAY_RATE = 0.5
    BN_DECAY_CLIP = 0.99

    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        global_step*batch_size,
        bn_decay_step,
        BN_DECAY_RATE,
        staircase=True)

    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

if __name__ == '__main__':

    sess = tf.InteractiveSession()
    a = tf.constant(5.0)
    b = tf.constant(6.0)
    c = a * b
    # We can just use 'c.eval()' without passing 'sess'
    print(c.eval())
    N = 512
    batch_size = 8
    num_parts  = 3
    bn_decay_step = 1000
    joint_cls  = tf.random_uniform([batch_size, N], minval=0, maxval=3, dtype=tf.int32)
    joint_cls  = tf.one_hot(joint_cls, depth=num_parts, axis=-1)

    P          = tf.random_uniform([batch_size, N, 3], minval=0, maxval=5, dtype=tf.float32)
    unitvec    = tf.ones([batch_size, N, 3])
    heatmap    = tf.random_uniform([batch_size, N, 1], minval=0, maxval=1, dtype=tf.float32)
    joint_axis = tf.ones([batch_size, N, 3])
    joint_params_gt= tf.ones([batch_size, num_parts-1, 7]) #
    nocs_pred  = tf.random_uniform(P.shape)

    keys_list  = ['joint_cls', 'unitvec', 'heatmap', 'joint_axis', 'nocs_pred', 'joint_params_gt']
    value_list = [joint_cls, unitvec, heatmap, joint_axis, nocs_pred, joint_params_gt]
    represent_dict = dict(zip(keys_list, value_list))
    print(list(represent_dict.keys()))
    print(joint_cls.eval().shape)
    print(joint_cls.eval()[0, 0:10, :])
    global_step = tf.Variable(0)
    is_training = tf.constant(True, dtype=tf.bool)
    bn_decay = get_batch_norm_decay(global_step, batch_size, bn_decay_step)
    # result, loss   = get_direct_regression_model('magic_pred', P, represent_dict, joint_params_gt, num_parts,  is_training, bn_decay, line_space='orthogonal')
    # print(result.shape)
    sess.close()
