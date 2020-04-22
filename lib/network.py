import os, sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..'))

import architecture
import loss
from loss import hungarian_matching # , calculate_eval_stats
# import fitter_factory
import prediction_io

import time
import numpy as np
import tensorflow as tf
import re
import subprocess

"""
- initialization;
- index selecton + 2-stage graph building;
- loss calculation and synthesis;
- graident slection;
- back-propagation and model saving, do we need to save all the weights?
"""
class Network(object):
    def __init__(self, n_max_parts, config, is_new_training):
        self.n_max_parts= n_max_parts
        self.config     = config
        self.graph      = tf.Graph()

        self.pred_joint      =  config.pred_joint # control both sub model architecture and loss
        self.pred_joint_ind  =  config.pred_joint_ind# control both sub model architecture and loss
        self.early_split     =  config.early_split# control sub model architecture
        self.early_split_nocs=  config.early_split_nocs# control early split for part & global NOCS
        self.is_mixed        =  False # control whether we want to consider part & global NOCS together
        if config.get_nocs_type() == 'ancsh':
            self.is_mixed = True
            print('We use mixed NOCS type...')

        with self.graph.as_default():
            self.global_step = tf.Variable(0)

            self.is_training = tf.placeholder(dtype=tf.bool, shape=[])

            self.P = tf.placeholder(dtype=tf.float32, shape=[None, None, 3]) # B, N, 3
            self.batch_size = tf.shape(self.P)[0]

            if config.get_bn_decay_step() < 0:
                self.bn_decay = None
            else:
                self.bn_decay = self.get_batch_norm_decay(self.global_step, self.batch_size, config.get_bn_decay_step())
                tf.summary.scalar('bn_decay', self.bn_decay)

            self.gt_dict = self.create_gt_dict(n_max_parts)

            # todo
            self.pred_dict = architecture.get_per_point_model_new(
                scope='SPFN',
                P=self.P,
                n_max_parts=n_max_parts,
                is_training=self.is_training,
                bn_decay=self.bn_decay,
                mixed_pred=self.is_mixed,
                pred_joint=self.pred_joint,
                pred_joint_ind=self.pred_joint_ind,
                early_split=self.early_split,
                early_split_nocs=self.early_split_nocs
            )
            # here we'll bring all the preditions into loss module
            eval_dict = self.compute_loss(
                self.pred_dict,
                self.gt_dict,
                config,
                is_eval=False,
                is_nn=True
            )
            self.collect_losses(eval_dict['loss_dict'])

            learning_rate = self.get_learning_rate(
                config.get_init_learning_rate(),
                self.global_step,
                self.batch_size,
                config.get_decay_step(),
                config.get_decay_rate())

            tf.summary.scalar('learning_rate', learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = self.create_train_op(learning_rate, self.total_loss)

            self.summary = tf.summary.merge_all()
            self.saver = tf.train.Saver(max_to_keep=3)

    def create_train_op(self, learning_rate, total_loss):
        # Skip gradient update if any gradient is infinite. This should not happen and is for debug only.
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.optimizer = optimizer
        grads_and_vars = optimizer.compute_gradients(total_loss)
        grads = [g for g, v in grads_and_vars]
        varnames = [v for g, v in grads_and_vars]
        is_finite = tf.ones(dtype=tf.bool, shape=[])
        for g, v in grads_and_vars:
            if g is not None:
                g_is_finite = tf.reduce_any(tf.is_finite(g))
                g_is_finite_cond = tf.cond(g_is_finite, tf.no_op, lambda: tf.Print(g_is_finite, [g], '{} is not finite:'.format(str(g))))
                with tf.control_dependencies([g_is_finite_cond]):
                    is_finite = tf.logical_and(is_finite, g_is_finite)
        train_op = tf.cond(
            is_finite,
            lambda: optimizer.apply_gradients(zip(grads, varnames), global_step=self.global_step),
            lambda: tf.Print(is_finite, [is_finite], 'Some gradients are not finite! Skipping gradient backprop.')
        )
        return train_op

    # as we have multiple losses funcs, here we try to combine each one
    def collect_losses(self, loss_dict):
        """
        confidence map is B*N*1
        """
        self.total_loss = tf.zeros(shape=[], dtype=tf.float32)
        self.nocs_loss_per_part = loss_dict['nocs_loss']
        self.total_nocs_loss = tf.reduce_mean(self.nocs_loss_per_part)
        nocs_loss_multiplier = self.config.get_nocs_loss_multiplier()
        if nocs_loss_multiplier > 0:
            tf.summary.scalar('total_nocs_loss', self.total_nocs_loss)

        if self.is_mixed:
            self.gocs_loss_per_part = loss_dict['gocs_loss']
            self.total_gocs_loss = tf.reduce_mean(self.gocs_loss_per_part)
            gocs_loss_multiplier = self.config.get_gocs_loss_multiplier()
            if gocs_loss_multiplier > 0:
                tf.summary.scalar('total_gocs_loss', self.total_gocs_loss)

        # loss from heatmap estimation & offset estimation
        self.total_heatmap_loss = tf.reduce_mean(loss_dict['heatmap_loss'])
        self.total_unitvec_loss = tf.reduce_mean(loss_dict['unitvec_loss'])
        self.total_orient_loss  = tf.reduce_mean(loss_dict['orient_loss'])
        heatmap_loss_multiplier = self.config.get_offset_loss_multiplier()
        unitvec_loss_multiplier = self.config.get_offset_loss_multiplier()
        orient_loss_multiplier  = self.config.get_orient_loss_multiplier()
        if heatmap_loss_multiplier > 0: # add loss profile on
            tf.summary.scalar('total_heatmap_loss', self.total_heatmap_loss)
            tf.summary.scalar('total_unitvec_loss', self.total_unitvec_loss)
            tf.summary.scalar('total_orient_loss', self.total_orient_loss)

        self.total_index_loss = tf.reduce_mean(loss_dict['index_loss'])
        index_loss_multiplier = self.config.get_index_loss_multiplier()
        if index_loss_multiplier > 0:
            tf.summary.scalar('total_index_loss', self.total_index_loss)

        self.miou_loss_per_part = loss_dict['miou_loss']
        self.total_miou_loss = tf.reduce_mean(self.miou_loss_per_part)
        miou_loss_multiplier = self.config.get_miou_loss_multiplier()
        tf.summary.scalar('total_miou_loss', self.total_miou_loss)

        self.total_loss += nocs_loss_multiplier * self.total_nocs_loss
        self.total_loss += miou_loss_multiplier * self.total_miou_loss
        if self.is_mixed:
            self.total_loss += gocs_loss_multiplier * self.total_gocs_loss

        if self.pred_joint: # todo
            if self.is_mixed: # only use it in part + global NOCS
                self.total_loss += heatmap_loss_multiplier * self.total_heatmap_loss
                self.total_loss += unitvec_loss_multiplier * self.total_unitvec_loss
            self.total_loss += orient_loss_multiplier * self.total_orient_loss
            if self.pred_joint_ind: # no joint points association
                self.total_loss += index_loss_multiplier * self.total_index_loss

        self.total_loss *= self.config.get_total_loss_multiplier()
        tf.summary.scalar('total_loss', self.total_loss)

    def train(self, sess, train_data, vals_data, n_epochs, val_interval, snapshot_interval, model_dir, log_dir):
        assert n_epochs > 0
        train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'), sess.graph)
        val_writer1 = tf.summary.FileWriter(os.path.join(log_dir, 'val1'), sess.graph)
        val_writer2 = tf.summary.FileWriter(os.path.join(log_dir, 'val2'), sess.graph)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(self.config.get_val_prediction_dir()):
            os.makedirs(self.config.get_val_prediction_dir())
        print('Training started.')

        start_time = time.time()
        for epoch in range(1, n_epochs + 1):
            for batch in train_data.create_iterator():
                feed_dict = self.create_feed_dict(batch, is_training=True)
                step, _, summary, loss = sess.run([self.global_step, self.train_op, self.summary, self.total_loss], feed_dict=feed_dict)

                elapsed_min = (time.time() - start_time) / 60
                print('Epoch: {:d} | Step: {:d} | Batch Loss: {:6f} | Elapsed: {:.2f}m'.format(epoch, step, loss, elapsed_min))

                if step >= self.config.get_writer_start_step():
                    train_writer.add_summary(summary, step)

                if step % val_interval == 0:
                    print('Start validating...')
                    msg = 'Epoch: {:d} | Step: {:d}'.format(epoch, step)
                    remain_min = (n_epochs * train_data.n_data - step) * elapsed_min / step
                    for i, val_data in enumerate(vals_data):
                        predict_result = self.predict_and_save(sess, val_data, save_dir=os.path.join(self.config.get_val_prediction_dir(), 'step{}'.format(step)))
                        msg = predict_result['msg']
                        msg = 'Validation: ' + msg + ' | Elapsed: {:.2f}m, Remaining: {:.2f}m'.format(elapsed_min, remain_min)
                        print(msg)
                        # clean up old predictions
                        prediction_n_keep = self.config.get_val_prediction_n_keep()
                        if prediction_n_keep != -1:
                            self.clean_predictions_earlier_than(step=step, prediction_dir=self.config.get_val_prediction_dir(), n_keep=prediction_n_keep)
                        if step >= self.config.get_writer_start_step():
                            if i == 0:
                                val_writer1.add_summary(predict_result['summary'], step)
                            else:
                                val_writer2.add_summary(predict_result['summary'], step)

                if step % snapshot_interval == 0:
                    print('Saving snapshot at step {:d}...'.format(step))
                    self.saver.save(sess, os.path.join(model_dir, 'tf_model.ckpt'), global_step=step)
                    print('Done saving model at step {:d}.'.format(step))

        train_writer.close()
        val1_writer.close()
        val2_writer.close()
        elapsed_min = (time.time() - start_time) / 60
        print('Training finished.')
        print('Elapsed: {:.2f}m.'.format(elapsed_min))
        print('Saved {}.'.format(self.saver.save(sess, os.path.join(model_dir, 'tf_model.ckpt'), global_step=step)))

    def format_loss_result(self, losses):
        msg = ''
        msg += 'Total Loss: {:6f}'.format(losses['total_loss'])
        msg += ', MIoU Loss: {:6f}'.format(losses['total_miou_loss'])
        msg += ', nocs Loss: {:6f}'.format(losses['total_nocs_loss'])
        if self.is_mixed:
            msg += ', gocs Loss: {:6f}'.format(losses['total_gocs_loss'])
        if self.pred_joint:
            msg += ', heatmap Loss: {:6f}'.format(losses['total_heatmap_loss'])
            msg += ', unitvec Loss: {:6f}'.format(losses['total_unitvec_loss'])
            if self.early_split:
                msg += ', orient Loss: {:6f}'.format(losses['total_orient_loss'])
        if self.pred_joint_ind:
            msg += ', index Loss: {:6f}'.format(losses['total_index_loss'])

        return msg

    def clean_predictions_earlier_than(self, step, prediction_dir, n_keep):
        prog = re.compile('step([0-9]+)')
        arr = []
        for f in os.listdir(prediction_dir):
            if os.path.isdir(os.path.join(prediction_dir, f)):
                m = prog.match(f)
                if m is not None:
                    arr.append((int(m.group(1)), f))
        arr.sort(key=lambda pr: pr[0])
        for pr in arr[:-n_keep]:
            subprocess.run(['rm', '-r', os.path.join(prediction_dir, pr[1])])

    def predict_and_save(self, sess, dset, save_dir):
        print('Predicting and saving predictions to {}...'.format(save_dir))
        losses = {
                'total_loss': 0.0,
                'total_miou_loss': 0.0,
                'total_nocs_loss': 0.0,
            }
        if self.is_mixed:
            losses['total_gocs_loss'] = 0.0

        if self.pred_joint:
            losses['total_heatmap_loss'] = 0.0
            losses['total_unitvec_loss'] = 0.0
            if self.early_split:
                losses['total_orient_loss']  = 0.0

        if self.pred_joint_ind:
            losses['total_index_loss']  = 0.0

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for batch in dset.create_iterator():
            feed_dict = self.create_feed_dict(batch, is_training=False)
            loss_dict = {
                'total_loss': self.total_loss,
                'total_miou_loss': self.total_miou_loss,
                'total_nocs_loss': self.total_nocs_loss,
                'total_heatmap_loss': self.total_heatmap_loss,
                'total_unitvec_loss': self.total_unitvec_loss,
                'total_orient_loss' : self.total_orient_loss,
                'total_index_loss'  : self.total_index_loss,
            }
            if self.is_mixed:
                loss_dict['total_gocs_loss'] = self.total_gocs_loss

            pred_result, loss_result = sess.run([self.pred_dict, loss_dict], feed_dict=feed_dict)

            for key in losses.keys():
                losses[key] += loss_result[key] * dset.last_step_size

            prediction_io.save_batch_nn(
                nn_name=self.config.get_nn_name(),
                pred_result=pred_result,
                input_batch=batch,
                basename_list=dset.get_last_batch_basename_list(),
                save_dir=save_dir,
                is_mixed=self.is_mixed,
                W_reduced=False
            )
            print('Finished {}/{}'.format(dset.get_last_batch_range()[1], dset.n_data), end='\r')
        losses.update((x, y / dset.n_data) for x, y in losses.items())
        msg = self.format_loss_result(losses)
        open(os.path.join(save_dir, 'test_loss.txt'), 'w').write(msg)
        summary = tf.Summary()
        for x, y in losses.items():
            summary.value.add(tag=x, simple_value=y)
        return {
            'msg': msg,
            'summary': summary,
        }

    def simple_predict_and_save(self, sess, pc, pred_h5_file):
        feed_dict = {
            self.P: np.expand_dims(pc, axis=0), # 1xNx3
            self.is_training: False
        }
        pred_result = sess.run(self.pred_dict, feed_dict=feed_dict)
        prediction_io.save_single_nn(
            nn_name=self.config.get_nn_name(),
            pred_result=pred_result,
            pred_h5_file=pred_h5_file,
            W_reduced=False,
        )

    def create_feed_dict(self, batch, is_training):
        feed_dict = {
            self.P : batch['P'],
            self.is_training: is_training,
        }
        self.fill_gt_dict_with_batch_data(feed_dict, self.gt_dict, batch)

        return feed_dict

    def create_gt_dict(self, n_max_parts):
        '''
            Returns gt_dict containing:
                - cls_per_point: BxN
                - nocs_per_point: BxNx3
                # - type_per_part: BxK
                - points_per_part: BxKxN'x3, sampled points on each part
                - parameters: a dict, each entry is a BxKx... tensor
        '''
        gt_dict = {}
        gt_dict['nocs_per_point']       = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
        gt_dict['cls_per_point']        = tf.placeholder(dtype=tf.int32, shape=[None, None])
        gt_dict['mask_array_per_point'] = tf.placeholder(dtype=tf.float32, shape=[None, None, n_max_parts])
        if self.is_mixed:
            gt_dict['gocs_per_point']   = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
        #
        gt_dict['heatmap_per_point']= tf.placeholder(dtype=tf.float32, shape=[None, None])
        gt_dict['unitvec_per_point']= tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
        gt_dict['orient_per_point'] = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
        gt_dict['index_per_point']  = tf.placeholder(dtype=tf.int32, shape=[None, None])
        gt_dict['joint_cls_mask']   = tf.placeholder(dtype=tf.float32, shape=[None, None])
        gt_dict['joint_params_gt']  = tf.placeholder(dtype=tf.float32, shape=[None, None, 7])

        return gt_dict

    def fill_gt_dict_with_batch_data(self, feed_dict, gt_dict, batch):
        """
        feed dict update the results
        """
        feed_dict.update({
                gt_dict['nocs_per_point']: batch['nocs_gt'],     # input NOCS
                gt_dict['cls_per_point'] : batch['cls_gt'],       # part cls: 0-9
                gt_dict['mask_array_per_point']: batch['mask_array'],
                gt_dict['heatmap_per_point']: batch['heatmap_gt'],   # input offset scalar
                gt_dict['unitvec_per_point']: batch['unitvec_gt'],   # input offset scalar
                gt_dict['orient_per_point'] : batch['orient_gt'],
                gt_dict['index_per_point']: batch['joint_cls_gt'],
                gt_dict['joint_cls_mask']: batch['joint_cls_mask'],
                gt_dict['joint_params_gt']: batch['joint_params_gt'],
            })
        if self.is_mixed:
            feed_dict.update({
                gt_dict['gocs_per_point']: batch['nocs_gt_g'],   # input NOCS global
                })

    def get_batch_norm_decay(self, global_step, batch_size, bn_decay_step):
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

    def get_learning_rate(self, init_learning_rate, global_step, batch_size, decay_step, decay_rate):
        learning_rate = tf.train.exponential_decay(
            init_learning_rate,
            global_step*batch_size,
            decay_step,
            decay_rate,
            staircase=True)
        return learning_rate

    def load_ckpt(self, sess, pretrained_model_path=None):
        """
            Load a model checkpoint
            In train mode, load the latest checkpoint from the checkpoint folder if it exists; otherwise, run initializer.
            In other modes, load from the specified checkpoint file.
        """
        sess.run(tf.global_variables_initializer())
        var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='SPFN')
        saver= tf.train.Saver({v.op.name: v for v in var})
        print('Restoring from {}'.format(pretrained_model_path))
        saver.restore(sess, pretrained_model_path)

    def compute_loss(self, pred_dict, gt_dict, config, is_eval, is_nn, P_in=None):
        '''
            Input:
                pred_dict should contain:
                    - W: BxNxK, segmentation parts. Allow zero rows to indicate unassigned points.
                    - nocs_per_point: BxNx3, nocs per point
                    - confi_per_point: type per points
                        - This should be logit of shape BxNxT if is_eval=False, and actual value of shape BxN otherwise
                        - can contain -1
                    - parameters - a dict, each entry is a BxKx... tensor
                gt_dict should be obtained from calling create_gt_dict
                P_in - BxNx3 is the input point cloud, used only when is_eval=True

            Returns: {loss_dict, matching_indices} + stats from calculate_eval_stats(), where
                - loss_dict contains:
                    - nocs_loss: B, averaged over all N points
                    - type_loss: B, averaged over all N points.
                        - This is cross entropy loss during training, and accuracy during test time
                    - miou_loss: BxK, mean IoU loss for each matched parts
                    - residue_loss: BxK, residue loss for each part
                    - parameter_loss: BxK, parameter loss for each part
                    - avg_miou_loss: B
                    - avg_residue_loss: B
                    - avg_parameter_loss: B
                - matching_indices: BxK, where (b,k)th ground truth primitive is matched with (b, matching_indices[b, k])
        '''
        # dimension tensors
        W          = pred_dict['W']
        batch_size = tf.shape(W)[0]   # B*N*K(k parts)
        n_points   = tf.shape(W)[1]
        n_max_parts= W.get_shape()[2] # n_max_parts should not be dynamic, fixed number of parts
        # n_registered_primitives = fitter_factory.get_n_registered_primitives()

        if is_eval and is_nn:
            # at loss, want W to be binary and filtered (if is from nn)
            W = nn_filter_W(W)

        # note that I_gt can contain -1, indicating part of unknown primitive type
        I_gt = gt_dict['cls_per_point'] # BxN
        n_parts_gt = tf.reduce_max(I_gt, axis=1) + 1 # only count known primitive type parts, as -1 will be ignored
        mask_gt = tf.sequence_mask(n_parts_gt, maxlen=n_max_parts) # BxK, mask_gt[b, k] = 1 iff instace k is present in the ground truth batch b

        matching_indices = tf.stop_gradient(tf.py_func(hungarian_matching, [W, I_gt], Tout=tf.int32)) # BxK into K parts
        # miou_loss = loss.compute_miou_loss(W, I_gt, matching_indices) # losses all have dimension BxK, here is for segmentation
        miou_loss = loss.compute_miou_loss(W, I_gt)
        nocs_loss = loss.compute_nocs_loss(pred_dict['nocs_per_point'], gt_dict['nocs_per_point'], pred_dict['confi_per_point'], \
                                        num_parts=n_max_parts, mask_array=gt_dict['mask_array_per_point'],  \
                                        TYPE_L=config.get_nocs_loss(), MULTI_HEAD=True, SELF_SU=False) # todo

        if self.is_mixed:
            gocs_loss = loss.compute_nocs_loss(pred_dict['gocs_per_point'], gt_dict['gocs_per_point'], pred_dict['confi_per_point'], \
                                        num_parts=n_max_parts, mask_array=gt_dict['mask_array_per_point'],  \
                                        TYPE_L=config.get_nocs_loss(), MULTI_HEAD=True, SELF_SU=False) # todo

        heatmap_loss = loss.compute_vect_loss(pred_dict['heatmap_per_point'], gt_dict['heatmap_per_point'], confidence=gt_dict['joint_cls_mask'],\
                                    TYPE_L=config.get_nocs_loss())
        unitvec_loss = loss.compute_vect_loss(pred_dict['unitvec_per_point'], gt_dict['unitvec_per_point'], confidence=gt_dict['joint_cls_mask'],\
                                    TYPE_L=config.get_nocs_loss())
        orient_loss  = loss.compute_vect_loss(pred_dict['joint_axis_per_point'], gt_dict['orient_per_point'], confidence=gt_dict['joint_cls_mask'],\
                                TYPE_L=config.get_nocs_loss())

        J_gt = gt_dict['index_per_point'] # BxN
        inds_pred = pred_dict['index_per_point']
        miou_joint_loss = loss.compute_miou_loss(inds_pred, J_gt) # losses all have dimension BxK, here is for segmentation
        # here we need to add input GT masks for different array

        loss_dict = {
            'nocs_loss': nocs_loss,
            'miou_loss': miou_loss,
            'heatmap_loss': heatmap_loss,
            'unitvec_loss': unitvec_loss,
            'orient_loss' : orient_loss,
            'index_loss'  : miou_joint_loss
            }

        if self.is_mixed:
            loss_dict['gocs_loss'] = gocs_loss

        result = {'loss_dict': loss_dict, 'matching_indices': matching_indices}
        """
        if is_eval:
            result.update(
                calculate_eval_stats(
                    W=W,
                    matching_indices=matching_indices,
                    mask_gt=mask_gt,
                    P_in=P_in,
                    confi_per_point=pred_dict['confi_per_point'],
                )
            )
        """
        return result
