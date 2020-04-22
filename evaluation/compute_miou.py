
"""
evaluate prediction from A-NCSH and baselines
"""
import numpy as np
import os
import h5py
import pickle
import argparse
from tqdm import tqdm

import _init_paths
from global_info import global_info
from lib.transformations import euler_matrix
from lib.d3_utils import get_3d_bbox, rot_diff_rad, rot_diff_degree, iou_3d
from lib.vis_utils import plot3d_pts
from lib.data_utils import get_urdf_mobility

def compose_rt(rotation, translation):
    aligned_RT = np.zeros((4, 4), dtype=np.float32)
    aligned_RT[:3, :3] = rotation[:3, :3]
    aligned_RT[:3, 3]  = translation
    aligned_RT[3, 3]   = 1
    return aligned_RT

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', default='unseen', help='which sub test set to choose')
    parser.add_argument('--nocs', default='ANCSH', help='which sub test set to choose')
    parser.add_argument('--item', default='eyeglasses', help='object category for benchmarking')
    args = parser.parse_args()

    infos           = global_info()
    dset_info       = infos.datasets[args.item]
    num_parts       = dset_info.num_parts
    num_ins         = dset_info.num_object
    name_dset       = dset_info.dataset_name
    unseen_instances= dset_info.test_list
    test_ins        = dset_info.test_list
    special_ins     = dset_info.spec_list
    main_exp        = dset_info.exp
    npcs_exp        = dset_info.baseline
    test_exp        = main_exp
    choose_threshold= 0.1


    my_dir          = infos.base_path
    group_dir       = infos.group_path
    base_path       = my_dir + '/results/test_pred'
    root_dset       = my_dir + '/' + name_dset
    if args.item in ['drawer']:
        root_dset   = group_dir + '/' + name_dset

    baseline_file = my_dir + '/results/pickle/{}/{}_{}_{}_rt_pn.pkl'.format(test_exp, args.domain, 'ANCSH', args.item)
    pn_gt_file    = my_dir + '/results/pickle/{}/{}_{}_{}_rt.pkl'.format(test_exp, args.domain, 'ANCSH', args.item)
    gn_gt_file    = my_dir + '/results/pickle/{}/{}_{}_{}_rt.pkl'.format(test_exp, args.domain, 'NAOCS', args.item)

    directory_subs = my_dir + '/results/pickle/{}/subs'.format(main_exp)
    all_files = os.listdir(directory_subs)
    valid_files = []
    for k in range(30):
        curr_file = '{}_{}_{}_{}_rt_ours_{}_{}.pkl'.format(npcs_exp, args.domain, args.nocs, args.item, choose_threshold, k)
        if curr_file in all_files:
            valid_files.append(directory_subs + '/' + curr_file)
    valid_files.sort()

    result_files = {'pn_gt': pn_gt_file, 'gn_gt': gn_gt_file, 'baseline': baseline_file, 'nonlinear': valid_files}
    test_items = list(result_files.keys())[-2:] # ['baseline', 'nonlinear']
    datas       = {}
    basenames   = {}
    n_raw_err   = {'baseline': [], 'nonlinear': []}
    r_raw_err   = {'baseline': [], 'nonlinear': []}
    t_raw_err   = {'baseline': [], 'nonlinear': []}
    s_raw_err   = {'baseline': [], 'nonlinear': []}
    iou_rat     = {'baseline': [], 'nonlinear': []}
    r_diff_raw_err   = {'baseline': [], 'nonlinear': []}
    t_diff_raw_err   = {'baseline': [], 'nonlinear': []}
    for key, file_name in result_files.items():
        if key == 'nonlinear':
            datas[key] = {}
            if isinstance(file_name, list):
                for cur_file in file_name:
                    curr_rts =  pickle.load( open(cur_file, 'rb'))
                    # print('curr_rts has ', curr_rts)
                    for sub_key, value in curr_rts.items():
                        datas[key][sub_key] = value
            else:
                datas[key] = pickle.load( open(cur_file, 'rb'))
            basenames[key] = list(datas[key].keys())
        else:
            with open(file_name, 'rb') as f:
                datas[key] = pickle.load(f)
                basenames[key] = list(datas[key].keys())
                print('number of data for {} : {}'.format(key, len(basenames[key])))
    for key, cur_data in datas.items():
        if key[-2:] == 'gt':
            continue
        print('fetch error data for', key)
        for i in range(len(basenames[key])):
            basename = basenames[key][i]
            name_info      = basename.split('_')
            instance       = name_info[0]
            art_index      = name_info[1]
            frame_order    = name_info[2]
            if cur_data[ basename ]['scale'] is None or cur_data[ basename ]['scale'] is [] :
                continue
            r_raw_err[key].append(cur_data[ basename ]['rpy_err'][key])
            t_raw_err[key].append(cur_data[ basename ]['xyz_err'][key])

    # >>>>>>>>>>>>>>>>>>>>>>>>>> 3D IoU & boundary computation <<<<<<<<<<<<<<<<<<<<<<<<<<< #
    with open(root_dset + "/pickle/{}.pkl".format(args.item),"rb") as f:
        all_factors = pickle.load(f)
    with open(root_dset + "/pickle/{}_corners.pkl".format(args.item), "rb") as fc:
        all_corners = pickle.load(fc)
    bbox3d_all    = {}
    for instance in test_ins:
        if args.item in ['drawer']:
            target_order =  infos.datasets[args.item].spec_map[instance]
            path_urdf  = group_dir + '/mobility-v0-prealpha3' + '/objects/' + '/' + args.item + '/' + instance
            urdf_ins   = get_urdf_mobility(path_urdf)
            joint_rpy = urdf_ins['joint']['rpy'][target_order[0]]
            rot_mat = euler_matrix(joint_rpy[0], joint_rpy[1], joint_rpy[2])[:3, :3]

        norm_factors = all_factors[instance]
        norm_corners = all_corners[instance]
        bbox3d_per_part = [None] * num_parts
        for p in range(num_parts):
            norm_factor = norm_factors[p+1]
            norm_corner = norm_corners[p+1]
            nocs_corner = np.copy(norm_corner)
            nocs_corner[0] = np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (norm_corner[1] - norm_corner[0]) * norm_factor
            nocs_corner[1] = np.array([0.5, 0.5, 0.5]).reshape(1, 3) + 0.5 * (norm_corner[1] - norm_corner[0]) * norm_factor
            if args.item in ['drawer']:
                nocs_corner[0]   = np.dot(nocs_corner[0].reshape(1, 3) - 0.5, rot_mat.T) + 0.5
                nocs_corner[1]   = np.dot(nocs_corner[1].reshape(1, 3) - 0.5, rot_mat.T) + 0.5
                bbox3d_per_part[target_order.index(p)] = nocs_corner
            else:
                bbox3d_per_part[p] = nocs_corner
        bbox3d_all[instance] = bbox3d_per_part

    iou_better_cnt = {k:0 for k in range(num_parts)}
    iou_worse_cnt  = {k:0 for k in range(num_parts)}
    boundary_all = {'baseline': {}, 'nonlinear': {}}

    pbar = tqdm(range(len(basenames['nonlinear'])))
    for i in pbar:
        pbar.set_description(f'computing for {i}th data entry')
        iou_all       = {'baseline':[], 'nonlinear':[]}
        scale_err_all = {'baseline':[], 'nonlinear':[]}
        volume_err_all= {'baseline':[], 'nonlinear':[]}
        try:
            basename = basenames['nonlinear'][i]
            for key in ['baseline', 'nonlinear']:
                cur_data = datas[key]
                if cur_data[ basename ]['scale'] is None or cur_data[ basename ]['scale'] is [] or np.any(np.isnan(cur_data[ basename ]['translation'][key])) :
                    # print('wrong data for scale or nan: ', basename)
                    continue
                name_info      = basename.split('_')
                instance       = name_info[0]
                art_index      = name_info[1]
                frame_order    = name_info[2]

                rt_gt    = datas['pn_gt'][ basename ]['rt']['gt']
                s_gt     = datas['pn_gt'][ basename ]['scale']['gt']

                rt_g     = datas['gn_gt'][ basename ]['rt']['gt']
                s_g      = datas['gn_gt'][ basename ]['scale']['gt']

                r        = cur_data[ basename ]['rotation'][key]
                t        = cur_data[ basename ]['translation'][key]
                s        = cur_data[ basename ]['scale'][key]

                # retrieve point cloud, use pred part nocs to get Amodal BBox
                f = h5py.File(my_dir + '/results/test_pred/{}/{}.h5'.format(npcs_exp, basename), 'r')
                nocs_pred = f['nocs_per_point'][()]
                nocs_gt   = f['nocs_gt'][()]
                input_pts  =  f['P'][()]
                mask_pred  =  f['instance_per_point'][()]
                mask_gt    =  f['cls_gt'][()]
                cls_per_pt_pred = np.argmax(mask_pred, axis=1)
                cls_per_pt_gt   = mask_gt

                partidx = {'gt': [], 'pred': []}
                for j in range(num_parts):
                    if key == 'nonlinear' and args.nocs == 'NAOCS':
                        tp2g = datas['st_gt'][basename]['pred']['translation'][j]
                        sp2g = datas['st_gt'][basename]['pred']['scale'][j]
                        t[j] = s[j] * np.dot(tp2g, r[j].T) + t[j]
                        s[j] = s[j] * sp2g

                bboxs3d = {'gt': [], 'pred': []}
                boundary= {'canon': [], 'dynam': []}
                rt_0 = compose_rt(r[0], t[0])
                scale_err_per_part  = []
                volume_err_per_part = []
                for j in range(num_parts):
                    partidx['pred'].append(np.where(cls_per_pt_pred==j)[0])
                    centered_nocs = nocs_pred[partidx['pred'][j], 3*j:3*(j+1)] - 0.5
                    scale_pred    = 2 * np.max(abs(centered_nocs), axis=0)
                    bboxs3d['pred'].append(get_3d_bbox(scale_pred, shift=np.array([1/2, 1/2, 1/2])).transpose())
                    boundary['canon'].append(- scale_pred[0]/2 + 0.5)

                    shifted_part_canon = np.dot( np.concatenate([input_pts[partidx['pred'][j], :3], np.ones(( len(partidx['pred'][j]), 1)) ], axis=1), np.linalg.pinv(rt_0.T))[:, :3]
                    boundary['dynam'].append(np.min(shifted_part_canon[:, 0]))
                    partidx['gt'].append(np.where(cls_per_pt_gt==j)[0])
                    scale_gt   = bbox3d_all[instance][j][1][0] - bbox3d_all[instance][j][0][0]
                    bboxs3d['gt'].append( get_3d_bbox(scale_gt, shift=np.array([1/2, 1/2, 1/2])).transpose() )
                    scale_err_per_part.append( np.linalg.norm(scale_pred * s[j] - scale_gt * s_gt[j]))
                    volume_err_per_part.append( scale_pred[0]*scale_pred[1]*scale_pred[2] * s[j]/(scale_gt[0] * scale_gt[1] * scale_gt[2] * s_gt[j][0]) - 1)

                bboxs3d_rt = {'gt': [], 'pred': []}
                iou_per_part = []
                for j in range(num_parts):
                    bb1 = bboxs3d['gt'][j] * s_gt[j][0]
                    bb2 = bboxs3d['pred'][j] * s[j]
                    rt1 = rt_gt[j]
                    rt2 = compose_rt(r[j], t[j])
                    bbox3d_gt = np.dot(bb1, rt1[:3,:3].T) + rt1[:3, 3]
                    bbox3d_pred = np.dot(bb2, rt2[:3,:3].T) + rt2[:3, 3]
                    iou = iou_3d(bbox3d_gt, bbox3d_pred)
                    bboxs3d_rt['gt'].append( bbox3d_gt )
                    bboxs3d_rt['pred'].append( bbox3d_pred )
                    iou_per_part.append(iou)
                iou_all[key] = iou_per_part
                scale_err_all[key]  = scale_err_per_part
                volume_err_all[key] = volume_err_per_part
                assert len(iou_per_part) == num_parts, print('Only get ', len(iou_per_part))
                iou_rat[key].append(iou_per_part)
                boundary_all[key][basename] = boundary
        except:
            pass

    print('For {} object, {} nocs, 3D IoU per part is: '.format(args.domain, args.nocs))
    for item in test_items:
        iou_arr = np.array(iou_rat[item])
        num_valid = iou_arr.shape[0]
        iou_p = []
        for j in range(num_parts):
            iou_p.append(np.sum( iou_arr[:, j] ) / num_valid)
        print(item[0:8], " ".join(["{:0.4f}".format(x) for x in iou_p]))
    print('\n')
