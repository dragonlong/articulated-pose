import numpy as np
import random
import os
import h5py
import csv
import pickle
import re
import argparse
import glob

import _init_paths
from network_config import NetworkConfig
from data_utils import calculate_factor_nocs, get_model_pts, write_pointcloud, get_urdf, split_dataset, get_urdf_mobility
from d3_utils import point_3d_offset_joint
from vis_utils import plot3d_pts, plot_arrows, plot_imgs
from transformations import euler_matrix
from global_info import global_info
epsilon = 10e-8

infos           = global_info()
base_path       = infos.base_path
group_dir       = infos.group_path

class Dataset:
    def __init__(self, root_dir, ctgy_obj, mode, n_max_parts, batch_size, name_dset='shape2motion', num_expr=0.01, domain=None, nocs_type='A', parametri_type='orthogonal', first_n=-1,  \
                   add_noise=False, fixed_order=False, is_debug=False, is_testing=False, is_gen=False, baseline_joints=False):
        self.root_dir     = root_dir
        self.name_dset    = name_dset
        self.ctgy_obj     = ctgy_obj
        infos             = global_info()
        self.ctgy_spec    = infos.datasets[ctgy_obj]
        self.parts_map    = infos.datasets[ctgy_obj].parts_map
        self.baseline_joints = baseline_joints

        self.num_points   = 1024 # fixed for category with < 5 parts
        self.batch_size   = batch_size
        self.n_max_parts  = n_max_parts
        self.fixed_order  = fixed_order
        self.first_n      = first_n
        self.add_noise    = add_noise
        self.is_testing   = is_testing
        self.is_gen       = is_gen
        self.is_debug     = is_debug
        self.nocs_type    = nocs_type
        self.line_space   = parametri_type
        self.hdf5_file_list = []
        if mode == 'train':
            idx_txt = self.root_dir + '/splits/{}/{}/train.txt'.format(ctgy_obj, num_expr)
        elif mode == 'demo':
            idx_txt = self.root_dir + '/splits/{}/{}/demo.txt'.format(ctgy_obj, num_expr)
        else:
            idx_txt = self.root_dir + '/splits/{}/{}/test.txt'.format(ctgy_obj, num_expr)
        with open(idx_txt, "r", errors='replace') as fp:
            line = fp.readline()
            cnt  = 1
            while line:
                # todos: test mode
                hdf5_file = line.strip()
                item = hdf5_file.split('.')[0].split('/')[-3]
                if mode=='test':
                    if domain=='seen' and (item not in infos.datasets[ctgy_obj].test_list):
                        self.hdf5_file_list.append(hdf5_file)
                    if domain=='unseen' and (item in infos.datasets[ctgy_obj].test_list):
                        self.hdf5_file_list.append(hdf5_file)
                    if domain is None:
                        self.hdf5_file_list.append(hdf5_file)
                else:
                    self.hdf5_file_list.append(hdf5_file)

                line = fp.readline()
        if is_debug:
            print('hdf5_file_list: ', len(self.hdf5_file_list), self.hdf5_file_list[0])
        if not fixed_order:
            random.shuffle(self.hdf5_file_list)
        if first_n != -1:
            self.hdf5_file_list = self.hdf5_file_list[:first_n]

        self.basename_list = [ "_".join(["{}".format(q) for q in p.split('.')[0].split('/')[-3:]]) for p in self.hdf5_file_list]
        if is_debug:
            print('basename_list: ', self.basename_list[0])
        self.n_data = len(self.hdf5_file_list)
        self.first_iteration_finished = False
        # whole URDF points, load all obj files
        self.all_factors, self.all_corners = self.fetch_factors_nocs(self.ctgy_obj, is_debug=self.is_debug, is_gen=self.is_gen)
        self.all_joints = self.fetch_joints_params(self.ctgy_obj, is_debug=self.is_debug)

    def fetch_data_at_index(self, i):
        assert not self.first_iteration_finished
        path = self.hdf5_file_list[i]
        if self.is_testing or self.is_debug:
            print('Fetch {}th datapoint from {}'.format(i, path))
        # name = os.path.splitext(os.path.basename(path))[0]
        item = path.split('.')[0].split('/')[-3]
        norm_factor_instance = self.all_factors[item]
        corner_pts_instance  = self.all_corners[item]
        joints = self.all_joints[item]
        if self.is_debug:
            print('Now fetching {}th data from instance {} with norm_factors: {}'.format(i, item, norm_factor_instance))
        with h5py.File(path, 'r') as handle:
            data = self.create_unit_data_from_hdf5(handle, self.n_max_parts, self.num_points, parts_map=self.parts_map, instance=item,  \
                                norm_factors=norm_factor_instance, norm_corners=corner_pts_instance, joints=joints, nocs_type=self.nocs_type, \
                                add_noise=self.add_noise, fixed_order=self.fixed_order, shuffle=not self.fixed_order, line_space=self.line_space,\
                                is_testing=self.is_testing)
            # assert data is not None # assume data are all clean
        if self.is_testing or self.is_debug:
            return data, path
        return data

    def __iter__(self):
        self.current = 0
        if not self.fixed_order and self.first_iteration_finished:
            # shuffle data matrix
            perm = np.random.permutation(self.n_data)
            for key in self.data_matrix.keys():
                self.data_matrix[key] = self.data_matrix[key][perm]

        return self

    def __next__(self):
        if self.current >= self.n_data:
            if not self.first_iteration_finished:
                self.first_iteration_finished = True
            raise StopIteration()

        step = min(self.n_data - self.current, self.batch_size)
        assert step > 0
        self.last_step_size = step
        batched_data        = {}
        if self.first_iteration_finished:
            for key in self.data_matrix.keys():
                batched_data[key] = self.data_matrix[key][self.current:self.current+step, ...]
        else:
            data = []
            for i in range(step):
                dp = self.fetch_data_at_index(self.current + i)
                if dp is not None:
                    data.append(dp)
            # if len(data) < step:
            #     for i in range(len(data), step):
            #         data.append(data[0])
            if not hasattr(self, 'data_matrix'):
                self.data_matrix = {}
                for key in data[0].keys():
                    trailing_ones = np.full([len(data[0][key].shape)], 1, dtype=int)
                    self.data_matrix[key] = np.tile(np.expand_dims(np.zeros_like(data[0][key]), axis=0), [self.n_data, *trailing_ones])
            for key in data[0].keys():
                try:
                    batched_data[key] = np.stack([x[key] for x in data], axis=0)
                    self.data_matrix[key][self.current:self.current+step, ...] = batched_data[key][0:step]
                except:
                    print('error key is ', key)
                    break

        self.current += step
        return batched_data

    def get_last_batch_range(self):
        # return: [l, r)
        return (self.current - self.last_step_size, self.current)

    def get_last_batch_basename_list(self):
        assert self.fixed_order
        l, r = self.get_last_batch_range()
        return self.basename_list[l:r]

    def create_iterator(self):
        return self

    def fetch_factors_nocs(self, obj_category, is_debug=False, is_gen=False):
        if is_gen:
            all_items   = os.listdir(self.root_dir + '/render/' + obj_category) # check according to render folder
            all_factors = {}
            all_corners = {}
            pts_m       = {}
            root_dset   = self.root_dir
            offsets     = None
            for item in all_items:
                if self.name_dset == 'sapien':
                    path_urdf = self.root_dir + '/objects/' + '/' + obj_category + '/' + item
                    urdf_ins   = get_urdf_mobility(path_urdf)
                elif self.name_dset == 'shape2motion':
                    path_urdf = self.root_dir + '/urdf/' + '/' + obj_category
                    urdf_ins   = get_urdf("{}/{}".format(path_urdf, item))
                else:
                    path_urdf = self.root_dir + '/urdf/' + '/' + obj_category
                    urdf_ins   = get_urdf("{}/{}".format(path_urdf, item))
                pts, norm_factors, corner_pts = get_model_pts(self.root_dir, self.ctgy_obj, item, obj_file_list=urdf_ins['obj_name'],  offsets=offsets , is_debug=is_debug)
                all_factors[item]        = norm_factors
                all_corners[item]        = corner_pts

                pt_ii         = []
                bbox3d_per_part = []
                for p, pt in enumerate(pts):
                    pt_s = np.concatenate(pt, axis=0)
                    pt_ii.append(pt_s)
                    print('We have {} pts'.format(pt_ii[p].shape[0]))
                if pt_ii is not []:
                    pts_m[item] = pt_ii
                else:
                    print('!!!!! {} model loading is wrong'.format(item))
            # save into pickle file, need to make pickle folder
            directory = root_dset + "/pickle/"
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(root_dset + "/pickle/{}.pkl".format(obj_category), "wb") as f:
                pickle.dump(all_factors, f)
            with open(root_dset + "/pickle/{}_corners.pkl".format(obj_category), "wb") as fc:
                pickle.dump(all_corners, fc)
            with open(root_dset + "/pickle/{}_pts.pkl".format(obj_category), 'wb') as fp:
                pickle.dump(pts_m, fp)
        else:
            root_dset   = self.root_dir
            # open a file, where you stored the pickled data
            file = open(root_dset + "/pickle/{}.pkl".format(obj_category),"rb")
            # dump information to that file
            data = pickle.load(file)
            all_factors = data
            file.close()
            fc = open(root_dset + "/pickle/{}_corners.pkl".format(obj_category), "rb")
            all_corners = pickle.load(fc)
            fc.close()
        if is_debug:
            print('Now fetching nocs normalization factors', type(all_factors), all_factors)

        return all_factors, all_corners

    def fetch_joints_params(self, obj_category, is_debug=False):
        all_items   = os.listdir(self.root_dir + '/render/' + obj_category) #TODO: which one to choose? urdf or render?
        all_joints  = {}
        root_dset   = self.root_dir
        for item in all_items:
            if self.name_dset == 'shape2motion':
                path_urdf = self.root_dir + '/urdf/' + '/' + obj_category
                urdf_ins   = get_urdf("{}/{}".format(path_urdf, item))
            elif self.name_dset == 'sapien':
                path_urdf = self.root_dir + '/objects/' + '/' + obj_category + '/' + item
                urdf_ins   = get_urdf_mobility(path_urdf)
            else:
                path_urdf = self.root_dir + '/urdf/' + '/' + obj_category
                urdf_ins   = get_urdf("{}/{}".format(path_urdf, item))
            if obj_category == 'bike':
                urdf_ins['link']['xyz'][1], urdf_ins['link']['xyz'][2] = urdf_ins['link']['xyz'][2], urdf_ins['link']['xyz'][1]
                urdf_ins['joint']['axis'][1], urdf_ins['joint']['axis'][2] = urdf_ins['joint']['axis'][2], urdf_ins['joint']['axis'][1]
            all_joints[item] = urdf_ins

            if is_debug:
                print(urdf_ins['link']['xyz'], urdf_ins['joint']['axis'])

        return all_joints

    def create_unit_data_from_hdf5(self, f, n_max_parts, num_points, parts_map = [[0, 3, 4], [1, 2]], instance=None, \
                                norm_corners=[None], norm_factors=[None], joints=None, nocs_type='A', line_space='orth', thres_r=0.2,\
                                add_noise=False, fixed_order=False, check_only=False, shuffle=True,\
                                is_testing=False, is_debug=False):
        n_parts   = len(parts_map)
        if self.name_dset == 'sapien':
            nocs_p, nocs_g, nocs_n, parts_cls, parts_pts, offset_heatmap, offset_unitvec, joint_orient, joint_cls, joint_params, n_total_points = self.create_data_mobility(f, n_max_parts, num_points, parts_map = parts_map, instance=instance,\
                                        norm_corners=norm_corners, norm_factors=norm_factors, joints=joints, nocs_type=nocs_type, line_space=line_space, thres_r=thres_r,\
                                        add_noise=add_noise, fixed_order=fixed_order, check_only=check_only, shuffle=shuffle, \
                                        is_testing=is_testing, is_debug=is_debug)
        else:
            nocs_p, nocs_g, nocs_n, parts_cls, parts_pts, offset_heatmap, offset_unitvec, joint_orient, joint_cls, joint_params, n_total_points = self.create_data_shape2motion(f, n_max_parts, num_points, parts_map = parts_map, instance=instance,\
                                        norm_corners=norm_corners, norm_factors=norm_factors, joints=joints, nocs_type=nocs_type, line_space=line_space, thres_r=thres_r,\
                                        add_noise=add_noise, fixed_order=fixed_order, check_only=check_only, shuffle=shuffle, \
                                        is_testing=is_testing, is_debug=is_debug)
        cls_arr = np.concatenate(parts_cls, axis=0)
        pts_arr = np.concatenate(parts_pts, axis=0)
        offset_heatmap =  np.concatenate(offset_heatmap, axis=0)
        if is_debug:
            print('offset_heatmap max is {}'.format(np.amax(offset_heatmap)))
        offset_unitvec =  np.concatenate(offset_unitvec, axis=0)
        joint_orient   =  np.concatenate(joint_orient, axis=0)
        joint_cls      =  np.concatenate(joint_cls, axis=0)

        if nocs_p[0] is not None:
            p_arr = np.concatenate(nocs_p, axis=0)
        if nocs_n[0] is not None:
            n_arr = np.concatenate(nocs_n, axis=0)
        if nocs_g[0] is not None:
            g_arr = np.concatenate(nocs_g, axis=0)

        if n_parts > n_max_parts:
            print('n_parts {} > n_max_parts {}'.format(n_parts, n_max_parts))
            return None

        if np.amax(cls_arr) >= n_parts:
            print('max label {} > n_parts {}'.format(np.amax(cls_arr), n_parts))
            return None

        if n_total_points < num_points:
            # print('tiling points, n_total_points {} < num_points {} required'.format(n_total_points, num_points))
            # we'll tile the points
            tile_n = int(num_points/n_total_points) + 1
            n_total_points = tile_n * n_total_points
            cls_tiled = np.concatenate([cls_arr] * tile_n, axis=0)
            cls_arr = cls_tiled
            pts_tiled = np.concatenate([pts_arr] * tile_n, axis=0)
            pts_arr = pts_tiled
            offset_heatmap_tiled = np.concatenate([offset_heatmap] * tile_n, axis=0)
            offset_heatmap   = offset_heatmap_tiled
            offset_unitvec_tiled = np.concatenate([offset_unitvec] * tile_n, axis=0)
            offset_unitvec   = offset_unitvec_tiled
            joint_orient_tiled = np.concatenate([joint_orient] * tile_n, axis=0)
            joint_orient     = joint_orient_tiled
            joint_cls_tiled  = np.concatenate([joint_cls] * tile_n, axis=0)
            joint_cls        = joint_cls_tiled
            if nocs_p[0] is not None:
                p_tiled = np.concatenate([p_arr] * tile_n, axis=0)
                p_arr   = p_tiled

            if nocs_n[0] is not None:
                n_tiled = np.concatenate([n_arr] * tile_n, axis=0)
                n_arr   = n_tiled

            if nocs_g[0] is not None:
                g_tiled = np.concatenate([g_arr] * tile_n, axis=0)
                g_arr   = g_tiled

        if check_only:
            return True
        if self.name_dset == 'BMVC15':
            if n_parts < 3:
                mask_array    = np.zeros([num_points, 3], dtype=np.float32)
            else:
                mask_array    = np.zeros([num_points, 6], dtype=np.float32)
        else:
            mask_array    = np.zeros([num_points, n_parts], dtype=np.float32)
        img = f['rgb'][()]
        if is_testing: # return the original unsmapled data
            if self.name_dset == 'sapien':
                target_order = self.ctgy_spec.spec_map[instance]
                joint_rpy = joints['joint']['rpy'][target_order[0]]
                rot_mat = euler_matrix(joint_rpy[0], joint_rpy[1], joint_rpy[2])[:3, :3]
                p_arr   = np.dot(p_arr-0.5, rot_mat.T) + 0.5
                g_arr   = np.dot(g_arr-0.5, rot_mat.T) + 0.5
            result = {
                'img': img,
                'P': pts_arr*norm_factors[0], # todo
                'cls_gt': cls_arr.astype(np.float32),
                'mask_array': mask_array.astype(np.float32),
                'nocs_gt': p_arr,
                'nocs_gt_g': g_arr,
            }
            return result

        perm          = np.random.permutation(n_total_points)
        cls_arr       = cls_arr[perm[:num_points]]
        if self.name_dset == 'BMVC15':
            pts_arr       = pts_arr[perm[:num_points]]
        else:
            pts_arr       = pts_arr[perm[:num_points]] * norm_factors[0]
        offset_heatmap_arr = offset_heatmap[perm[:num_points]]
        offset_unitvec_arr = offset_unitvec[perm[:num_points]]
        joint_orient_arr   =  joint_orient[perm[:num_points]]
        joint_cls_arr = joint_cls[perm[:num_points]]
        # print('joint_cls_arr has shape: ', joint_cls_arr.shape)
        joint_cls_mask = np.zeros((joint_cls_arr.shape[0]), dtype=np.float32)
        id_valid     = np.where(joint_cls_arr>0)[0]
        joint_cls_mask[id_valid] = 1.00

        mask_array[np.arange(num_points), cls_arr.astype(np.int8)] = 1.00

        if nocs_p[0] is not None:
            p_arr = p_arr[perm[:num_points]]
        if nocs_n[0] is not None:
            n_arr = n_arr[perm[:num_points]]
        if nocs_g[0] is not None:
            g_arr = g_arr[perm[:num_points]]

        # rotate according to urdf_ins joint_rpy
        if self.name_dset == 'sapien':
            target_order = self.ctgy_spec.spec_map[instance]
            joint_rpy = joints['joint']['rpy'][target_order[0]]
            rot_mat = euler_matrix(joint_rpy[0], joint_rpy[1], joint_rpy[2])[:3, :3]
            if nocs_p[0] is not None:
                p_arr   = np.dot(p_arr-0.5, rot_mat.T) + 0.5
            g_arr   = np.dot(g_arr-0.5, rot_mat.T) + 0.5
            offset_unitvec_arr= np.dot(offset_unitvec_arr, rot_mat.T)
            joint_orient_arr  = np.dot(joint_orient_arr, rot_mat.T)

        if nocs_type == 'A':
            result = {
                'P': pts_arr,
                'cls_gt': cls_arr.astype(np.float32),
                'mask_array': mask_array.astype(np.float32),
                'nocs_gt'   : p_arr.astype(np.float32),
                'nocs_gt_g' : g_arr.astype(np.float32),
                'heatmap_gt'   : offset_heatmap_arr.astype(np.float32),
                'unitvec_gt'   : offset_unitvec_arr.astype(np.float32),
                'orient_gt'    : joint_orient_arr.astype(np.float32),
                'joint_cls_gt'    : joint_cls_arr.astype(np.float32),
                'joint_cls_mask'  : joint_cls_mask.astype(np.float32),
                'joint_params_gt' : joint_params,
            }
        elif nocs_type == 'B':
            result = {
                'P': pts_arr,
                'cls_gt': cls_arr.astype(np.float32),
                'mask_array': mask_array.astype(np.float32),
                'nocs_gt'   : n_arr.astype(np.float32),
            }
        elif nocs_type == 'C':
            result = {
                'P': pts_arr,
                'cls_gt': cls_arr.astype(np.float32),
                'mask_array': mask_array.astype(np.float32),
                'nocs_gt'   : g_arr.astype(np.float32),
                'heatmap_gt'   : offset_heatmap_arr.astype(np.float32),
                'unitvec_gt'   : offset_unitvec_arr.astype(np.float32),
                'orient_gt'    : joint_orient_arr.astype(np.float32),
                'joint_cls_gt'    : joint_cls_arr.astype(np.float32),
                'joint_cls_mask'  : joint_cls_mask.astype(np.float32),
                'joint_params_gt' : joint_params,
                # 'joint_axis'   : # [B, 2, 7]
            }
        elif nocs_type == 'AC':
            result = {
                'P': pts_arr,
                'cls_gt': cls_arr.astype(np.float32),
                'mask_array': mask_array.astype(np.float32),
                'nocs_gt'   : p_arr.astype(np.float32),
                'nocs_gt_g' : g_arr.astype(np.float32),
                'heatmap_gt'   : offset_heatmap_arr.astype(np.float32),
                'unitvec_gt'   : offset_unitvec_arr.astype(np.float32),
                'orient_gt'    : joint_orient_arr.astype(np.float32),
                'joint_cls_gt'    : joint_cls_arr.astype(np.float32),
                'joint_cls_mask'  : joint_cls_mask.astype(np.float32),
                'joint_params_gt' : joint_params,
            }
        # for keys, item in result.items():
        #     print(keys, item.shape)
        return result

    def create_data_shape2motion(self, f, n_max_parts, num_points, parts_map = [[0, 3, 4], [1, 2]], instance=None, \
                                norm_corners=[None], norm_factors=[None], joints=None, nocs_type='A', line_space='orth', thres_r=0.2,\
                                add_noise=False, fixed_order=False, check_only=False, shuffle=True, \
                                is_testing=False, is_debug=False):
        '''
            f will be a h5py group-like object
        '''
        # read
        n_parts   = len(parts_map)  # parts map to combine points
        parts_pts = [None] * n_parts
        parts_gts = [None] * n_parts
        parts_cls = [None] * n_parts
        parts_nocs= [None] * n_parts
        nocs_p    = [None] * n_parts
        nocs_g    = [None] * n_parts
        nocs_n    = [None] * n_parts
        n_total_points = 0
        parts_parent_joint= [None] * n_parts
        parts_child_joint = [None] * n_parts
        if n_parts ==2:
            parts_offset_joint= [[], []]
            parts_joints      = [[], []]
            joint_index       = [[], []]
        elif n_parts == 3:
            parts_offset_joint= [[], [], []] # per part pts offsets to correponding parts
            parts_joints      = [[], [], []] # joint params list of the joints
            joint_index       = [[], [], []] # joint index recording the corresponding parts
        elif n_parts == 4:
            parts_offset_joint= [[], [], [], []] # per part pts offsets to correponding parts
            parts_joints      = [[], [], [], []] # joint params list of the joints
            joint_index       = [[], [], [], []] # joint index recording the corresponding parts

        joint_xyz = joints['link']['xyz']
        joint_rpy = joints['joint']['axis']
        joint_part= joints['joint']['parent']
        joint_type= joints['joint']['type']

        joint_params = np.zeros((n_parts, 7))
        if line_space == 'plucker':
            joint_params = np.zeros((n_parts, 6))

        for idx, group in enumerate(parts_map):
            P = f['gt_points'][str(group[0])][()][:, :3]
            for i in range(1, len(group)):
                P = np.concatenate((P, f['gt_points'][str(group[i])][()][:, :3]), axis=0)
            parts_pts[idx] = P
            n_total_points += P.shape[0]
            parts_cls[idx] = idx * np.ones((P.shape[0]), dtype=np.float32)
            Pc = f['gt_coords'][str(group[0])][()][:, :3]
            for i in range(1, len(group)):
                Pc = np.concatenate((Pc, f['gt_coords'][str(group[i])][()][:, :3]), axis=0)
            parts_gts[idx] = Pc
            parts_parent_joint[idx] = group[0] # first element as part that serve as child
            parts_child_joint[idx] = [ind for ind, x in enumerate(joint_part) if x == group[-1]] # in a group, we may use the last element to find joint that part serves as parent
        # plot3d_pts([parts_gts], [['part {}'.format(j) for j in range(len(parts_map))]], s=10, title_name=['default'])
        # print('parts_child_joint: ', parts_child_joint)
        # print('parts_parent_joint: ', parts_parent_joint)
        for j in range(n_parts):
            if nocs_type=='A':   # part NOCS compared to A-shape
                norm_factor = norm_factors[j+1]
                norm_corner = norm_corners[j+1]
                nocs_p[j] = (parts_gts[j][:, :3] - norm_corner[0]) * norm_factor + np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (  norm_corner[1] - norm_corner[0]) * norm_factor

            elif nocs_type=='B': # partial visiable points
                tight_w = max(parts_gts[j][:, 0]) - min(parts_gts[j][:, 0])
                tight_l = max(parts_gts[j][:, 1]) - min(parts_gts[j][:, 1])
                tight_h = max(parts_gts[j][:, 2]) - min(parts_gts[j][:, 2])
                norm_factor = np.sqrt(1) / np.sqrt(tight_w**2 + tight_l**2 + tight_h**2)
                left_p      = np.amin(parts_gts[j], axis=0, keepdims=True)
                right_p     = np.amax(parts_gts[j], axis=0, keepdims=True)
                norm_corner = [left_p, right_p]
                nocs_n[j]   = (parts_gts[j][:, :3] - norm_corner[0]) * norm_factor + np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (  norm_corner[1] - norm_corner[0]) * norm_factor

            elif nocs_type=='AC': # part NOCS + global NOCS
                norm_factor = norm_factors[j+1]
                norm_corner = norm_corners[j+1]
                nocs_p[j] = (parts_gts[j][:, :3] - norm_corner[0]) * norm_factor + np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (  norm_corner[1] - norm_corner[0]) * norm_factor

            norm_factor = norm_factors[0]
            norm_corner = norm_corners[0]
            nocs_g[j] = (parts_gts[j][:, :3] - norm_corner[0]) * norm_factor + np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (  norm_corner[1] - norm_corner[0]) * norm_factor
            if j>0:
                joint_P0  = - np.array(joint_xyz[j])
                joint_P0  = (joint_P0 - norm_corner[0]) * norm_factor  + np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (  norm_corner[1] - norm_corner[0]) * norm_factor
                joint_l   = np.array(joint_rpy[j])
                orth_vect = point_3d_offset_joint([joint_P0, joint_l], np.array([0, 0, 0]).reshape(1, 3))
                joint_params[j, 0:3] = joint_l
                joint_params[j, 6]   = np.linalg.norm(orth_vect)
                joint_params[j, 3:6] = orth_vect/joint_params[j, 6]

            if parts_parent_joint[j] !=0:
                joint_P0  = - np.array(joint_xyz[parts_parent_joint[j]])
                joint_P0  = (joint_P0 - norm_corner[0]) * norm_factor  + np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (  norm_corner[1] - norm_corner[0]) * norm_factor
                joint_l   = np.array(joint_rpy[j])
                offset_arr= point_3d_offset_joint([joint_P0, joint_l], nocs_g[j])
                parts_offset_joint[j].append(offset_arr)
                parts_joints[j].append([joint_P0, joint_l])
                joint_index[j].append(parts_parent_joint[j])
                # if is_debug:
                #     plot_arrows(nocs_g[j], offset_arr, [joint_P0, joint_l], title_name='NOCS {} to joint {}'.format(j, j))
            if parts_child_joint[j] is not None:
                for m in parts_child_joint[j]:
                    joint_P0  = - np.array(joint_xyz[m])
                    joint_P0  = (joint_P0 - norm_corner[0]) * norm_factor  + np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (  norm_corner[1] - norm_corner[0]) * norm_factor
                    joint_l   = np.array(joint_rpy[m])
                    offset_arr= point_3d_offset_joint([joint_P0, joint_l], nocs_g[j])
                    parts_offset_joint[j].append(offset_arr)
                    parts_joints[j].append([joint_P0, joint_l])
                    joint_index[j].append(m)
                        # if is_debug:
                        #     plot_arrows(nocs_g[j], offset_arr, [joint_P0, joint_l], title_name='NOCS {} to joint {}'.format(j, m))
        # sampling & fusion
        # rate_sampling = num_points_in/num_points
        offset_heatmap = [None] * n_parts
        offset_unitvec = [None] * n_parts
        joint_orient   = [None] * n_parts
        joint_cls      = [None] * n_parts
        for j, offsets in enumerate(parts_offset_joint):
            offset_heatmap[j] = np.zeros((parts_gts[j].shape[0]))
            offset_unitvec[j] = np.zeros((parts_gts[j].shape[0], 3))
            joint_orient[j]   = np.zeros((parts_gts[j].shape[0], 3))
            joint_cls[j]      = np.zeros((parts_gts[j].shape[0]))
            for k, offset in enumerate(offsets):
                heatmap = np.linalg.norm(offset, axis=1)
                unitvec = offset/(heatmap.reshape(-1, 1) + epsilon)
                idc     = np.where(heatmap<thres_r)[0]
                offset_heatmap[j][idc]    = 1 - heatmap[idc]/thres_r
                offset_unitvec[j][idc, :] = unitvec[idc, :]
                joint_orient[j][idc, :]   = parts_joints[j][k][1]
                joint_cls[j][idc]         = joint_index[j][k]

        if nocs_type == 'C':
            if is_debug:
                plot_arrows_list(nocs_g, parts_offset_joint, parts_joints, title_name='joint offset')
                plot_arrows_list_threshold(nocs_g, parts_offset_joint, parts_joints, title_name='joint offset')

        return  nocs_p, nocs_g, nocs_n, parts_cls, parts_pts, offset_heatmap, offset_unitvec, joint_orient, joint_cls, joint_params, n_total_points

    def create_data_mobility(self, f, n_max_parts, num_points, parts_map = [[0, 3, 4], [1, 2]], instance=None, \
                                norm_corners=[None], norm_factors=[None], joints=None, nocs_type='A', line_space='orth', thres_r=0.2,\
                                add_noise=False, fixed_order=False, check_only=False, shuffle=True, \
                                is_testing=False, is_debug=False):
        '''
            f will be a h5py group-like object
        '''
        # read
        n_parts   = len(parts_map)  # parts map to combine points
        parts_pts = [None] * n_parts
        parts_gts = [None] * n_parts
        parts_cls = [None] * n_parts
        parts_nocs= [None] * n_parts
        nocs_p    = [None] * n_parts
        nocs_g    = [None] * n_parts
        nocs_n    = [None] * n_parts
        n_total_points = 0
        parts_parent_joint= [None] * n_parts
        parts_child_joint = [None] * n_parts

        if n_parts == 3:
            parts_offset_joint= [[], [], []] # per part pts offsets to correponding parts
            parts_joints      = [[], [], []] # joint params list of the joints
            joint_index       = [[], [], []] # joint index recording the corresponding parts
        elif n_parts == 4:
            parts_offset_joint= [[], [], [], []] # per part pts offsets to correponding parts
            parts_joints      = [[], [], [], []] # joint params list of the joints
            joint_index       = [[], [], [], []] # joint index recording the corresponding parts
        elif n_parts == 5:
            parts_offset_joint= [[], [], [], [], []] # per part pts offsets to correponding parts
            parts_joints      = [[], [], [], [], []] # joint params list of the joints
            joint_index       = [[], [], [], [], []] # joint index recording the corresponding parts

        # fetch link & joint offsets from urdf_ins
        link_xyz = joints['link']['xyz'][1:] # we only need to consider link xyz, n_parts
        joint_rpy = joints['joint']['rpy']
        joint_xyz = joints['joint']['xyz']
        joint_axis= joints['joint']['axis']
        joint_parent= joints['joint']['parent']
        joint_type  = joints['joint']['type']
        joint_child = joints['joint']['child']

        #
        joint_params = np.zeros((n_parts, 7))
        if line_space == 'plucker':
            joint_params = np.zeros((n_parts, 6))

        for idx, group in enumerate(parts_map):
            P = f['gt_points'][str(group[0])][()][:, :3]
            for i in range(1, len(group)):
                P = np.concatenate((P, f['gt_points'][str(group[i])][()][:, :3]), axis=0)
            parts_pts[idx] = P              # input pt cloud
            n_total_points += P.shape[0]
            parts_cls[idx] = idx * np.ones((P.shape[0]), dtype=np.float32)
            Pc = f['gt_coords'][str(group[0])][()][:, :3]
            for i in range(1, len(group)):
                Pc = np.concatenate((Pc, f['gt_coords'][str(group[i])][()][:, :3]), axis=0)
            parts_gts[idx] = Pc - np.array(link_xyz[idx][0]).reshape(1, 3) # pts in canonical coords
            parts_parent_joint[idx] = group[0] # first element as child, assume link and its parent joint have same index
            parts_child_joint[idx]  = [ind for ind, x in enumerate(joint_parent) if x-1 == group[-1]] # in a group, we may use the last element to find joint that part serves as parent

        for j in range(n_parts):
            if nocs_type=='A':   # part NOCS compared to A-shape
                norm_factor = norm_factors[j+1]
                norm_corner = norm_corners[j+1]
                nocs_p[j] = (parts_gts[j][:, :3] - norm_corner[0]) * norm_factor + np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (  norm_corner[1] - norm_corner[0]) * norm_factor

            elif nocs_type=='B': # partial visiable points
                tight_w = max(parts_gts[j][:, 0]) - min(parts_gts[j][:, 0])
                tight_l = max(parts_gts[j][:, 1]) - min(parts_gts[j][:, 1])
                tight_h = max(parts_gts[j][:, 2]) - min(parts_gts[j][:, 2])
                norm_factor = np.sqrt(1) / np.sqrt(tight_w**2 + tight_l**2 + tight_h**2)
                left_p      = np.amin(parts_gts[j], axis=0, keepdims=True)
                right_p     = np.amax(parts_gts[j], axis=0, keepdims=True)
                norm_corner = [left_p, right_p]
                nocs_n[j] = (parts_gts[j][:, :3] - norm_corner[0]) * norm_factor + np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (  norm_corner[1] - norm_corner[0]) * norm_factor

            elif nocs_type=='AC': # part NOCS + global NOCS
                norm_factor = norm_factors[j+1]
                norm_corner = norm_corners[j+1]
                nocs_p[j] = (parts_gts[j][:, :3] - norm_corner[0]) * norm_factor + np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (  norm_corner[1] - norm_corner[0]) * norm_factor

            norm_factor = norm_factors[0]
            norm_corner = norm_corners[0]
            nocs_g[j] = (parts_gts[j][:, :3] - norm_corner[0]) * norm_factor + np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (  norm_corner[1] - norm_corner[0]) * norm_factor

            def compute_joint_params(j_position, j_axis, j_type, pts, norm_factor, norm_corner):
                joint_P0 =  - np.array(j_position)
                joint_P0  = (joint_P0 - norm_corner[0]) * norm_factor  + np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (  norm_corner[1] - norm_corner[0]) * norm_factor
                if j_type == 'fixed':
                    joint_l   = np.array([0, 0, 1])
                    orth_vect = np.ones_like(pts) * 0.5 * thres_r
                elif j_type == 'prismatic':
                    joint_l   = np.array(j_axis)
                    orth_vect = np.ones_like(pts) * 0.5 * thres_r
                else:
                    joint_l   = np.array(j_axis)
                    orth_vect = point_3d_offset_joint([joint_P0, joint_l], np.array([0, 0, 0]).reshape(1, 3))
                return orth_vect, joint_P0, joint_l

            # we use 0th xyz since we have multiple obj xyz
            orth_vect, joint_P0, joint_l = compute_joint_params(link_xyz[j][0], joint_axis[j], joint_type[j], np.array([0, 0, 0]).reshape(1, 3), norm_factor, norm_corner)
            joint_params[j, 0:3] = joint_l
            joint_params[j, 6]   = np.linalg.norm(orth_vect)
            joint_params[j, 3:6] = orth_vect/joint_params[j, 6]

            idj = parts_parent_joint[j]
            offset_arr, joint_P0, joint_l = compute_joint_params(link_xyz[idj][0], joint_axis[ idj ], joint_type[ idj ], nocs_g[j], norm_factor, norm_corner)
            parts_offset_joint[j].append(offset_arr)
            parts_joints[j].append([joint_P0, joint_l])
            joint_index[j].append(idj)

            if parts_child_joint[j] !=[]:
                for m in parts_child_joint[j]:
                    offset_arr, joint_P0, joint_l = compute_joint_params(link_xyz[m][0], joint_axis[m], joint_type[m], nocs_g[j], norm_factor, norm_corner)
                    parts_offset_joint[j].append(offset_arr)
                    parts_joints[j].append([joint_P0, joint_l])
                    joint_index[j].append(m)

        # sampling & fusion
        offset_mask    = [None] * n_parts
        offset_heatmap = [None] * n_parts
        offset_unitvec = [None] * n_parts
        joint_orient   = [None] * n_parts
        joint_cls      = [None] * n_parts

        for j, offsets in enumerate(parts_offset_joint):
            offset_mask[j]    = np.zeros((parts_gts[j].shape[0]))
            offset_heatmap[j] = np.zeros((parts_gts[j].shape[0]))
            offset_unitvec[j] = np.zeros((parts_gts[j].shape[0], 3))
            joint_orient[j]   = np.zeros((parts_gts[j].shape[0], 3))
            joint_cls[j]      = np.ones((parts_gts[j].shape[0])) * n_parts

            for k, offset in enumerate(offsets):
                id_j = joint_index[j][k]
                heatmap = np.linalg.norm(offset, axis=1)
                unitvec = offset/(heatmap.reshape(-1, 1) + epsilon)
                if joint_type[id_j] == 'prismatic':
                    idc     = np.where(heatmap > 0)[0]
                elif joint_type[id_j] == 'fixed':
                    continue
                else:
                    idc     = np.where(heatmap<=thres_r)[0]
                offset_heatmap[j][idc]    = 1 - heatmap[idc]/thres_r
                offset_unitvec[j][idc, :] = unitvec[idc, :]
                joint_orient[j][idc, :]   = parts_joints[j][k][1]
                joint_cls[j][idc]         = joint_index[j][k] #

        if nocs_type == 'C':
            if is_debug:
                plot_arrows_list(nocs_g, parts_offset_joint, parts_joints, title_name='joint offset')
                plot_arrows_list_threshold(nocs_g, parts_offset_joint, parts_joints, title_name='joint offset')
        # start to exchange the labels
        # [0, 1, 2, 3] to [3, 0, 1, 2]
        target_order = self.ctgy_spec.spec_map[instance]

        for j, idx in enumerate(target_order):
            parts_cls[idx] = np.ones_like(parts_cls[idx]) * j
            joint_cls[idx] = np.ones_like(joint_cls[idx]) * j

        return  nocs_p, nocs_g, nocs_n, parts_cls, parts_pts, offset_heatmap, offset_unitvec, joint_orient, joint_cls, joint_params, n_total_points


if __name__=='__main__':

    random.seed(30)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='./network_config.yml', help='YAML configuration file')
    parser.add_argument('--dataset', default='shape2motion', help='name of dataset')
    parser.add_argument('--item', default='oven', help='name of the dataset we use')
    # parser.add_argument('--dataset', default='sapien', help='name of the dataset we use')
    # parser.add_argument('--item', default='cabinet', help='name of the dataset we use')
    # parser.add_argument('--dataset', default='BMVC15', help='name of the dataset we use')
    # parser.add_argument('--item', default='Laptop', help='name of the dataset we use')
    parser.add_argument('--num_expr', default='0.01', help='get configuration file') # this 0.01 is just the default folder where we keep our record
    parser.add_argument('--nocs_type', default='AC', help='nocs_type') # this 0.01 is just the default folder where we keep our record

    parser.add_argument('--pred_joint', action='store_true', help='whether we want to predict joint offsets')
    parser.add_argument('--pred_joint_ind', action='store_true', help='whether we want to predict joint offsets index')
    parser.add_argument('--early_split', action='store_true', help='whether we want to early split')
    parser.add_argument('--split', action='store_true', help='to split the dataset')
    parser.add_argument('--gen', action='store_true', help='to split the dataset')
    parser.add_argument('--test', action='store_true', help='Run network in test time')
    parser.add_argument('--show_fig', action='store_true', help='Run network in test time')
    parser.add_argument('--save_fig', action='store_true', help='Run network in test time')
    parser.add_argument('--save_h5',  action='store_true', help='save h5 to test or demo folder')

    parser.add_argument('--domain', default='unseen', help='choose seen or unseen objects')
    parser.add_argument('--mode', default='train', help='help indicate split dataset')
    parser.add_argument('--debug', action='store_true', help='Run network in test time')
    parser.add_argument('--new_start', action='store_true', help='are we trying resume our training from pretrained stage1')
    parser.add_argument('--stage1_training', action='store_true', help='do we want to train stage 1 weights')
    parser.add_argument('--stage2_training', action='store_true', help='do we want to train stage 2 weights')
    parser.add_argument('--cycle', action='store_true', help='whether we want to enforce cycle consistency on part and global nocs')
    parser.add_argument('--early_split_nocs', action='store_true', help='whether we want to predict parallelly for ')
    args = parser.parse_args()

    # config file fixed
    num_expr = args.num_expr
    infos    = global_info()
    conf = NetworkConfig(args)
    is_testing = True if args.test else False
    is_debug   = True if args.debug else False
    is_split   = True if args.split else False
    is_gen     = True if args.gen else False
    name_dset  = args.dataset

    if args.dataset == 'sapien':
        root_dset = group_path + '/dataset/' + args.dataset
    else:
        root_dset = base_path + '/dataset/' + args.dataset

    ctgy_objs  = [args.item]
    batch_size = conf.get_batch_size()
    n_max_parts = infos.datasets[args.item].num_parts

    if is_split:
        split_dataset(root_dset, ctgy_objs, args, test_ins=infos.datasets[args.item].test_list, spec_ins=infos.datasets[args.item].spec_list, train_ins=infos.datasets[args.item].train_list)

    test_ins=infos.datasets[args.item].test_list

    train_data = Dataset(
        root_dir=root_dset,
        ctgy_obj=args.item,
        domain=args.domain,
        mode=args.mode,
        batch_size=batch_size,
        name_dset=args.dataset,
        n_max_parts=n_max_parts,
        add_noise=conf.is_train_data_add_noise(),
        fixed_order=True,
        num_expr=args.num_expr,
        first_n=conf.get_train_data_first_n(),
        nocs_type=args.nocs_type,
        is_debug=is_debug,
        is_testing=is_testing,
        is_gen=is_gen)

    # np.random.seed(0)
    # selected_index = np.random.randint(1000, size=20)
    selected_index = np.arange(0, 10)
    # selected_index = [train_data.basename_list.index('0016_0_0')] + list(np.arange(0, len(train_data.basename_list)))
    for i in selected_index:
        basename =  train_data.basename_list[i]
        if basename.split('_')[0] not in test_ins:
            continue
        instance = basename.split('_')[0]
        print('reading data point: ', i, train_data.basename_list[i])
        if is_debug or is_testing:
            data_pts, _ =  train_data.fetch_data_at_index(i)
        else:
            data_pts=  train_data.fetch_data_at_index(i)
        # print('fetching ', path)
        for keys, item in data_pts.items():
            print(keys, item.shape)
        # rgb_img   = data_pts['img']
        input_pts = data_pts['P']
        nocs_gt   = {}
        nocs_gt['pn']   = data_pts['nocs_gt']
        if args.nocs_type == 'AC':
            nocs_gt['gn']   = data_pts['nocs_gt_g']
        # print('nocs_gt has {}, {}'.format( np.amin(nocs_gt['pn'], axis=0), np.amax(nocs_gt, axis=0)))
        mask_gt   = data_pts['cls_gt']
        num_pts = input_pts.shape[0]
        num_parts = n_max_parts
        part_idx_list_gt   = []
        for j in range(num_parts):
            part_idx_list_gt.append(np.where(mask_gt==j)[0])
        if not args.test:
            heatmap_gt= data_pts['heatmap_gt']
            unitvec_gt= data_pts['unitvec_gt']
            orient_gt = data_pts['orient_gt']
            joint_cls_gt = data_pts['joint_cls_gt']
            joint_params_gt = data_pts['joint_params_gt']
            # print('joint_params_gt is: ', joint_params_gt)
            joint_idx_list_gt   = []
            for j in range(num_parts):
                joint_idx_list_gt.append(np.where(joint_cls_gt==j)[0])

        #>>>>>>>>>>>>>>>>>>>>>------ For segmentation visualization ----
        # plot_imgs([rgb_img], ['rgb img'], title_name='RGB', sub_name=str(i), show_fig=args.show_fig, save_fig=args.save_fig, save_path=root_dset + '/NOCS/' + args.item)
        plot3d_pts([[input_pts]], [['Part {}'.format(0)]], s=50, title_name=['GT seg on input point cloud'], sub_name=str(i), show_fig=args.show_fig, axis_off=True, save_fig=args.save_fig, save_path=root_dset + '/NOCS/' + args.item)
        plot3d_pts([[input_pts[part_idx_list_gt[j], :] for j in range(num_parts)]], [['Part {}'.format(j) for j in range(num_parts)]], s=50, title_name=['GT seg on input point cloud'], sub_name=str(i), show_fig=args.show_fig, axis_off=True, save_fig=args.save_fig, save_path=root_dset + '/NOCS/' + args.item)
        plot3d_pts([[nocs_gt['gn'][part_idx_list_gt[j], :] for j in range(num_parts)]], [['Part {}'.format(j) for j in range(num_parts)]], s=50, title_name=['GT global NOCS'], sub_name=str(i), show_fig=args.show_fig, save_fig=args.save_fig, save_path=root_dset + '/NOCS/' + args.item)
        plot3d_pts([[nocs_gt['pn'][part_idx_list_gt[j], :] for j in range(num_parts)]], [['Part {}'.format(j) for j in range(num_parts)]], s=50, title_name=['GT part NOCS'], sub_name=str(i), show_fig=args.show_fig, save_fig=args.save_fig, save_path=root_dset + '/NOCS/' + args.item)
        # for j in range(num_parts):
        #     plot3d_pts([[nocs_gt['pn'][part_idx_list_gt[j], :] ]], [['Part {}'.format(j)]], s=15, dpi=200, title_name=['GT part NOCS'], color_channel=[[ nocs_gt['pn'][part_idx_list_gt[j], :] ]], sub_name='{}_part_{}'.format(i, j) , show_fig=args.show_fig, save_fig=args.save_fig, save_path=root_dset + '/NOCS/' + args.item)
        #     plot3d_pts([[nocs_gt['gn'][part_idx_list_gt[j], :] ]], [['Part {}'.format(j)]], s=15, dpi=200, title_name=['GT global NOCS'], color_channel=[[ nocs_gt['gn'][part_idx_list_gt[j], :] ]], sub_name='{}_part_{}'.format(i, j), show_fig=args.show_fig, save_fig=args.save_fig, save_path=root_dset + '/NOCS/' + args.item)

        if not args.test:
            plot3d_pts([[input_pts[joint_idx_list_gt[j], :] for j in range(num_parts)]], [['unassigned Pts'] + ['Pts of joint {}'.format(j) for j in range(1, num_parts)]], s=15, \
                  title_name=['GT association of pts to joints '], sub_name=str(i), show_fig=args.show_fig, save_fig=args.save_fig, save_path=root_dset + '/NOCS/' + args.item, axis_off=True)
            plot3d_pts([[ input_pts ]], [['Part 0-{}'.format(num_parts-1)]], s=15, \
                  dpi=200, title_name=['Input Points distance heatmap'], color_channel=[[250*np.concatenate([heatmap_gt.reshape(-1, 1), np.zeros((heatmap_gt.shape[0], 2))], axis=1)]], show_fig=args.show_fig)

            thres_r       = 0.2
            offset        = unitvec_gt * (1- heatmap_gt.reshape(-1, 1)) * thres_r
            joint_pts     = nocs_gt['gn'] + offset
            joints_list   = []
            idx           = np.where(joint_cls_gt > 0)[0]
            plot_arrows(nocs_gt['gn'][idx], [0.5*orient_gt[idx]], whole_pts=nocs_gt['gn'], title_name='{}_joint_pts_axis'.format('GT'), dpi=200, s=25, thres_r=0.1, show_fig=args.show_fig, sparse=True, save=args.save_fig, index=i, save_path=root_dset + '/NOCS/' + args.item)
            plot_arrows(nocs_gt['gn'][idx], [offset[idx]], whole_pts=nocs_gt['gn'], title_name='{}_joint_offsets'.format('GT'), dpi=200, s=25, thres_r=0.1, show_fig=args.show_fig, sparse=True, save=args.save_fig, index=i, save_path=root_dset + '/NOCS/' + args.item)
            # plot_arrows(input_pts[idx], [0.5*orient_gt[idx]], whole_pts=input_pts, title_name='{}_joint_pts_axis'.format('GT'), dpi=200, s=25, thres_r=0.2, show_fig=args.show_fig, sparse=True, save=args.save_fig, index=i, save_path=root_dset + '/NOCS/' + args.item)
            # plot_arrows(input_pts[idx], [offset[idx]], whole_pts=input_pts, title_name='{}_joint_offsets'.format('GT'), dpi=200, s=25, thres_r=0.2, show_fig=args.show_fig, sparse=True, save=args.save_fig, index=i, save_path=root_dset + '/NOCS/' + args.item)
        if args.test:
            # channels_map = ['x', 'y', 'z']
            # directory = root_dset +  '/NOCS/{}'.format(args.item)
            # if not os.path.exists(directory):
            #     os.makedirs(directory)
            # print('We are saving data into {}'.format(directory))
            # filename  = directory + '/{}_input_{}.ply'.format(i, 'pc_nocs_color')
            # write_pointcloud(filename, input_pts, rgb_points=(nocs_gt['pn']*255).astype(np.uint8))
            # filename  = directory + '/{}_input_{}.ply'.format(i, 'pc')
            # write_pointcloud(filename, input_pts, rgb_points=((input_pts+1)*50).astype(np.uint8))
            # for j in range(num_parts):
            #     filename  = directory + '/{}_gt_{}_{}.ply'.format(i, 'pn', j)
            #     write_pointcloud(filename, nocs_gt['pn'][part_idx_list_gt[j], :], rgb_points=(nocs_gt['pn'][part_idx_list_gt[j], :]*255).astype(np.uint8))
            # filename  = directory + '/{}_gt_{}.ply'.format(i, 'gn')
            # write_pointcloud(filename, nocs_gt['gn'], rgb_points=(nocs_gt['gn']*255).astype(np.uint8))
            # basename
            dir_save_h5 = root_dset +  '/results/demo/{}'.format(args.item)
            if not os.path.exists(dir_save_h5):
                os.makedirs(dir_save_h5)
            name_save_h5= dir_save_h5 + '/{}.h5'.format(basename)
            print('Writing to ', name_save_h5)
            f = h5py.File(name_save_h5, 'w')
            f.attrs['basename'] = basename
            f.create_dataset('P', data=data_pts['P'])
            f.create_dataset('cls_gt', data=data_pts['cls_gt'])
            f.create_dataset('nocs_gt', data=data_pts['nocs_gt'])
            if args.nocs_type == 'AC':
                f.create_dataset('nocs_gt_g', data=data_pts['nocs_gt_g'])
            f.close()
