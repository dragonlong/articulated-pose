"""
evaluate prediction from global NOCS and part NOCS
"""
import numpy as np
import random
import platform
import os
import h5py
import csv
import pickle
import yaml
import json
import os.path
import sys
import argparse
import matplotlib.pyplot as plt  # matplotlib.use('Agg') # TkAgg
from mpl_toolkits.mplot3d import Axes3D

import _init_paths
from lib.vis_utils import plot3d_pts, plot2d_img, plot_arrows, plot_arrows_list
from lib.d3_utils import axis_diff_degree, dist_between_3d_lines, rot_diff_rad, rot_diff_degree
from global_info import global_info

def breakpoint():
    import pdb; pdb.set_trace()

def compose_rt(rotation, translation):
    aligned_RT = np.zeros((4, 4), dtype=np.float32)
    aligned_RT[:3, :3] = rotation[:3, :3]
    aligned_RT[:3, 3]  = translation
    aligned_RT[3, 3]   = 1
    return aligned_RT

# def rot_diff_rad(rot1, rot2):
#     return np.arccos( ( np.trace(np.matmul(rot1, rot2.T)) - 1 ) / 2 ) % (2*np.pi)

# def rot_diff_degree(rot1, rot2):
#     return rot_diff_rad(rot1, rot2) / np.pi * 180

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', default='unseen', help='which sub test set to choose')
    parser.add_argument('--nocs', default='ANCSH', help='which sub test set to choose')
    parser.add_argument('--item', default='eyeglasses', help='object category for benchmarking')
    parser.add_argument('--save', action='store_true', help='save err to pickles')
    parser.add_argument('--viz', action='store_true', help='whether to viz')
    args = parser.parse_args()

    infos           = global_info()
    my_dir          = infos.base_path
    dset_info       = infos.datasets[args.item]
    num_parts       = dset_info.num_parts
    num_ins         = dset_info.num_object
    name_dset       = dset_info.dataset_name
    unseen_instances= dset_info.test_list
    test_ins        = dset_info.test_list
    special_ins     = dset_info.spec_list

    main_exp        = dset_info.exp
    baseline_exp    = dset_info.baseline #
    test_exp        = main_exp # we may choose a differnt one to get the real input

    choose_threshold= 0.1

    pn_gt_file    = my_dir + '/results/pickle/{}/{}_{}_{}_rt.pkl'.format(main_exp, args.domain, 'ANCSH', args.item)
    gn_gt_file    = my_dir + '/results/pickle/{}/{}_{}_{}_rt.pkl'.format(main_exp, args.domain, 'NAOCS', args.item)
    directory_subs= my_dir + '/results/pickle/{}/subs'.format(main_exp)
    all_files = os.listdir(directory_subs)
    valid_files = []
    for k in range(30):
        curr_file = '{}_{}_{}_{}_rt_ours_{}_{}.pkl'.format(baseline_exp, args.domain, args.nocs, args.item, choose_threshold, k)
        if curr_file in all_files:
            valid_files.append(directory_subs + '/' + curr_file)
    valid_files.sort()
    print(valid_files)
    result_files = {'pn_gt': pn_gt_file, 'gn_gt': gn_gt_file, 'nonlinear': valid_files}
    test_items  = list(result_files.keys())[-2:] # ['baseline', 'nonlinear']
    datas       = {}
    basenames   = {}
    for key, file_name in result_files.items():
        if key == 'nonlinear':
            datas[key] = {}
            if isinstance(file_name, list):
                for cur_file in file_name:
                    curr_rts =  pickle.load( open(cur_file, 'rb'))
                    for sub_key, value in curr_rts.items():
                        datas[key][sub_key] = value
            else:
                datas[key] = pickle.load( open(cur_file, 'rb'))
            basenames[key] = list(datas[key].keys())
        else:
            with open(file_name, 'rb') as f:
                print(file_name)
                datas[key] = pickle.load(f)
                basenames[key] = list(datas[key].keys())
                print('number of data for {} : {}'.format(key, len(basenames[key])))

    #
    # breakpoint()
    # GT gn RTS, GT pn + gn, pred RTS in pn,
    test_h5_path = my_dir + '/results/test_pred/{}'.format(test_exp)
    all_st = {}
    angle_err_all= []
    dist_err_all = []
    for i in range(len(basenames['nonlinear'])):
    # for i in range(5):
        try:
            basename = basenames['nonlinear'][i]
            print('\n Checking {}th data point: {}'.format(i, basename))
            # get global NOCS and joints offsets prediction
            h5_file        =  test_h5_path + '/{}.h5'.format(basename)
            hf             =  h5py.File(h5_file, 'r')
            input_pts      =  hf['P'][()]
            mask_gt        =  hf['cls_gt'][()]
            nocs_gt        =  {}
            nocs_pred      =  {}
            nocs_gt['gn']  =  hf['nocs_gt_g'][()]
            nocs_pred['gn']=  hf['gocs_per_point'][()]
            nocs_gt['pn']  =  hf['nocs_gt'][()]
            nocs_pred['pn']=  hf['nocs_per_point'][()]

            mask_pred      =  hf['instance_per_point'][()]

            # for joints estimation
            heatmap_pred   =  hf['heatmap_per_point'][()]
            heatmap_gt     =  hf['heatmap_gt'][()]
            unitvec_pred   =  hf['unitvec_per_point'][()]
            unitvec_gt     =  hf['unitvec_gt'][()]

            # for joints axis
            orient_pred    = hf['joint_axis_per_point'][()]
            orient_gt      = hf['joint_axis_gt'][()]
            joint_cls_pred = hf['index_per_point'][()]
            joint_cls_pred = np.argmax(joint_cls_pred, axis=1)
            joint_cls_gt   = hf['joint_cls_gt'][()]

            name_info      = basename.split('_')
            item           = name_info[0]
            art_index      = name_info[1]
            frame_order    = name_info[2]
            instance       = item

            part_idx_list_gt   = []
            part_idx_list_pred = []
            joint_idx_list_gt  = []

            cls_per_pt_pred  = np.argmax(mask_pred, axis=1)
            for j in range(num_parts):
                if j > 0:
                    joint_idx_list_gt.append(np.where(joint_cls_gt==j)[0])
                part_idx_list_gt.append(np.where(mask_gt==j)[0])
                part_idx_list_pred.append(np.where(cls_per_pt_pred==j)[0])

            nocs_pred_final = {}
            nocs_pred_final['gn'] = np.zeros_like(nocs_gt['gn'])
            nocs_pred_final['pn'] = np.zeros_like(nocs_gt['pn'])
            st_dict = {}
            scale_list = []
            translation_list = []
            for j in range(num_parts):
                nocs_pred_final['pn'][part_idx_list_pred[j], :] = nocs_pred['pn'][part_idx_list_pred[j], j*3:j*3+3]
                if nocs_pred['gn'].shape[1] == 3:
                    nocs_pred_final['gn'][part_idx_list_pred[j], :] = nocs_pred['gn'][part_idx_list_pred[j], :3]
                else:
                    nocs_pred_final['gn'][part_idx_list_pred[j], :] = nocs_pred['gn'][part_idx_list_pred[j], j*3:j*3+3]
                x = nocs_pred_final['gn'][part_idx_list_pred[j], :] # N * 3
                y = nocs_pred_final['pn'][part_idx_list_pred[j], :] # N * 3
                scale = np.std(np.mean(y, axis=1))/np.std(np.mean(x, axis=1))
                translation = np.mean(y - scale*x, axis=0)
                print('scale, translation from global NOCS to part NOCS is: ', scale, translation)
                scale_list.append(scale)
                translation_list.append(translation)
            st_dict['scale'] = scale_list
            st_dict['translation'] = translation_list

            joints = {'gt': [], 'pred':[]}
            print('\n gn space')
            for j in range(1, num_parts):
                thres_r       = 0.2
                offset        = unitvec_pred * (1- heatmap_pred.reshape(-1, 1)) * thres_r
                nocs          = nocs_pred_final['gn']
                joint_pts     = nocs + offset
                idx           = np.where(joint_cls_pred == j)[0]
                joint_axis    = np.median(orient_pred[idx], axis=0)
                joint_pt      = np.median(joint_pts[idx], axis=0)
                print('pred joint pt: ',joint_pt, 'joint axis: ', joint_axis)
                joint = {}
                joint['l'] = joint_axis
                joint['p'] = joint_pt
                joints['pred'].append(joint)
                # plot_arrows(nocs_gt['gn'][idx], [offset[idx]], [[joint_pt.reshape(1,3), joint_axis.reshape(1, 3)]], whole_pts=nocs, title_name='pred joint {}'.format(j))

            for j in range(1, num_parts):
                thres_r       = 0.2
                offset        = unitvec_gt * (1- heatmap_gt.reshape(-1, 1)) * thres_r
                nocs          = nocs_gt['gn']
                joint_pts     = nocs + offset
                idx           = np.where(joint_cls_gt == j)[0]

                joint_axis    = np.mean(orient_gt[idx], axis=0)
                joint_pt      = np.median(joint_pts[idx], axis=0)
                print('gt joint pt: ',joint_pt, 'joint axis: ', joint_axis)
                joint      = {}
                joint['l'] = joint_axis
                joint['p'] = joint_pt
                joints['gt'].append(joint)
                # plot_arrows(nocs[idx], [offset[idx]], [[joint_pt.reshape(1,3), joint_axis.reshape(1, 3)]], whole_pts=nocs, title_name='gt joint {}'.format(j))

            rt_gt    = datas['pn_gt'][ basename ]['rt']['gt']
            s_gt     = datas['pn_gt'][ basename ]['scale']['gt']

            rt_g     = datas['gn_gt'][ basename ]['rt']['gt']
            s_g      = datas['gn_gt'][ basename ]['scale']['gt']

            r        = datas['nonlinear'][ basename ]['rotation']['nonlinear']
            t        = datas['nonlinear'][ basename ]['translation']['nonlinear']
            s        = datas['nonlinear'][ basename ]['scale']['nonlinear']

            # plot_arrows(input_pts[idx], [offset[idx]], [[joint_pt.reshape(1,3), joint_axis.reshape(1, 3)]], whole_pts=input_pts, title_name='nearby pts to joint {}'.format(j))
            # transform joint pts to part NOCS space, we use base as the platform
            t_joints = {'gt': [], 'pred':[]}
            print('\n camera space')
            for j in range(1, num_parts):
                s2 = st_dict['scale'][0]
                t2 = st_dict['translation'][0]
                t_joint_pt = joints['pred'][j-1]['p'] * s2 + t2
                joint = {}
                joint['p'] = np.dot(s[0] * t_joint_pt.reshape(1, 3), r[0].T) + t[0]
                joint['l'] = np.dot(joints['pred'][j-1]['l'].reshape(1, 3), r[0].T)
                print('pred joint pt: ',joint['p'], 'joint axis: ', joint['l'])
                t_joints['pred'].append(joint)
                # plot_arrows(nocs_pred_final['gn'][idx], [offset[idx]], [[joint['p'].reshape(1,3), joint['l'].reshape(1, 3)]], whole_pts=input_pts, title_name='camera space: pred joint {}'.format(j))

            for j in range(1, num_parts):
                t_joint_pt = joints['gt'][j-1]['p']
                joint = {}
                joint['p'] = np.dot(s_g[0] * t_joint_pt.reshape(1, 3), rt_g[0][:3, :3].T) + rt_g[0][:3, 3]
                joint['l'] = np.dot(joints['gt'][j-1]['l'].reshape(1, 3), rt_g[0][:3, :3].T)
                print('gt joint pt: ',joint['p'], 'joint axis: ', joint['l'])
                t_joints['gt'].append(joint)
                # plot_arrows(nocs_gt['gn'][idx], [offset[idx]], [[joint['p'].reshape(1,3), joint['l'].reshape(1, 3)]], whole_pts=input_pts, title_name='camera space: gt joint {}'.format(j))

            # visualize joints
            joints_list = []
            joints_list.append([t_joints['pred'][0]['p'], t_joints['pred'][0]['l']])
            joints_list.append([t_joints['gt'][0]['p'], t_joints['gt'][0]['l']])
            # plot_arrows(nocs_gt['gn'][idx], [offset[idx]], joints_list, whole_pts=input_pts, title_name='camera space: gt joint {}'.format(j))
            # start evaluation
            angle_err = []
            dist_err  = []
            for j in range(1, num_parts):
                r_diff = axis_diff_degree(t_joints['gt'][j-1]['l'], t_joints['pred'][j-1]['l'])
                t_diff = dist_between_3d_lines(t_joints['gt'][j-1]['p'], t_joints['gt'][j-1]['l'], t_joints['pred'][j-1]['p'], t_joints['pred'][j-1]['l'])
                print('joint {} --- r_diff: {}, t_diff: {}'.format(j, r_diff, t_diff))
                angle_err.append(r_diff)
                dist_err.append(t_diff)
            if len(angle_err) == num_parts - 1:
                angle_err_all.append(angle_err)
                dist_err_all.append(dist_err)
        except:
            pass
    r_diff_arr = np.array(angle_err_all) # num * k
    t_diff_arr = np.array(dist_err_all)  # num * k
    r_diff_arr[np.where(np.isnan(r_diff_arr))] = 0
    t_diff_arr[np.where(np.isnan(t_diff_arr))] = 0
    print(r_diff_arr.shape, t_diff_arr.shape, num_parts)
    for k in range(num_parts - 1):
        print('joint {} with mean angle error {} degrees, mean dist {}'.format(k, np.mean(np.abs(r_diff_arr[:, k])), np.mean(np.abs(t_diff_arr[:, k]))))
        print(np.mean(np.abs(r_diff_arr[:, k])), np.mean(np.abs(t_diff_arr[:, k])))
