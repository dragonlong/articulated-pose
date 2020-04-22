import numpy as np
import random
import os
import time
import h5py
import csv
import pickle
import yaml
import json
import pandas
import re
import argparse
import os.path
import sys
import struct
import trimesh

import glob
import platform
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, tostring, SubElement, Comment, ElementTree, XML
import xml.dom.minidom
import math

import matplotlib
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
from descartes import PolygonPatch

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.ioff()

import _init_paths
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from lib.vis_utils import plot_arrows, plot_arrows_list, plot_arrows_list_threshold, plot3d_pts, plot_lines
from global_info import global_info

infos = global_info()
base_path = infos.base_path
epsilon = 10e-8

def collect_file(root_dset, ctgy_objs, mode='train', selected_list=None):
    # list rgb
    for name_obj in ctgy_objs:
        if mode == 'train':
            name_instances = os.listdir(root_dset + '/render/' + name_obj)
        elif mode == 'demo':
            name_instances = os.listdir(root_dset + '/demo/' + name_obj)
        print('We have {} different instances'.format(len( name_instances )))

        for instance in name_instances:
            if selected_list is not None and instance not in selected_list:
                continue
            if mode == 'train':
                directory = root_dset + '/render/' + name_obj + '/' + instance + '/*'
            elif mode == 'demo':
                directory = root_dset + '/demo/' + name_obj + '/' + instance + '/*'
            else:
                directory = root_dset + '/render/' + name_obj + '/' + instance + '/*'
            for dir_arti in glob.glob(directory):
                h5_frames = os.listdir(dir_arti + '/rgb')
                h5_frames.sort()
                h5_list   = []
                for file in h5_frames:
                    if file.endswith('.png'):
                        h5_list.append(file.split('.')[0])
                all_csv = dir_arti + '/all.txt'
                with open(all_csv, 'w') as ft:
                  for item in h5_list:
                      ft.write('{}\n'.format(item))
                print('done for {} {}'.format(instance, dir_arti))

def split_dataset(root_dset, ctgy_objs, args, test_ins, spec_ins=[], train_ins=None):
    num_expr = args.num_expr
    if args.mode == 'train' or args.mode=='test':
        for name_obj in ctgy_objs:
            train_csv  = root_dset + '/splits/{}/{}/train.txt'.format(name_obj, num_expr)
            test_csv   = root_dset + '/splits/{}/{}/test.txt'.format(name_obj, num_expr)
            train_list = []
            test_list  = []
            if not os.path.exists(root_dset + '/splits/{}/{}'.format(name_obj, num_expr)):
                os.makedirs(root_dset + '/splits/{}/{}'.format(name_obj, num_expr))
            name_instances = os.listdir(root_dset + '/hdf5/' + name_obj)
            print('We have {} different instances'.format(len( name_instances )), name_instances)
            random.shuffle(name_instances)
            rm_ins = []
            ins_to_remove = test_ins + rm_ins
            print(ins_to_remove)
            for instance in ins_to_remove:
                if instance in name_instances:
                    name_instances.remove(instance)
            # remove tricycles
            if train_ins is None:
                train_ins = name_instances

            for instance in train_ins:
                if len(spec_ins)>0 and instance in spec_ins:
                    continue
                for dir_arti in glob.glob(root_dset + '/hdf5/' + name_obj + '/' + instance + '/*'):
                    h5_frames = glob.glob(dir_arti + '/*')
                    h5_list   = []
                    for file in h5_frames:
                        if file.endswith('.h5'):
                            h5_list.append(file)
                    print('training h5 has {} for {} {}'.format(len(h5_list) -1, instance, dir_arti))
                    random.shuffle(h5_list)
                    try:
                        train_list= train_list + h5_list[:-1]
                        test_list = test_list  + [h5_list[-1]]
                    except:
                        continue

            for instance in test_ins:
                for dir_arti in glob.glob(root_dset + '/hdf5/' + name_obj + '/' + instance + '/*'):
                    h5_frames = glob.glob(dir_arti + '/*')
                    h5_list   = []
                    for file in h5_frames:
                        if file.endswith('.h5'):
                            h5_list.append(file)
                    print('testing h5 has {} for {} {}'.format(len(h5_list), instance, dir_arti))
                    test_list = test_list  + h5_list
        print('train_list: \n', len(train_list))
        print('test list: \n', len(test_list))
        with open(train_csv, 'w') as ft:
          for item in train_list:
              ft.write('{}\n'.format(item))

        with open(test_csv, 'w') as ft:
          for item in test_list:
              ft.write('{}\n'.format(item))
    else:
        for name_obj in ctgy_objs:
            demo_csv  = root_dset + '/splits/{}/{}/demo.txt'.format(name_obj, num_expr)
            demo_list  = []
            if not os.path.exists(root_dset + '/splits/{}/{}'.format(name_obj, num_expr)):
                os.makedirs(root_dset + '/splits/{}/{}'.format(name_obj, num_expr))
            name_instances = os.listdir(root_dset + '/hdf5_demo/' + name_obj)
            print('We have {} different instances'.format(len( name_instances )))
            random.shuffle(name_instances)
            # test_ins  = ['0001']
            # remove tricycles
            demo_ins = name_instances

            for instance in demo_ins:
                for dir_arti in glob.glob(root_dset + '/hdf5_demo/' + name_obj + '/' + instance + '/*'):
                    h5_frames = glob.glob(dir_arti + '/*')
                    h5_list   = []
                    for file in h5_frames:
                        if file.endswith('.h5'):
                            h5_list.append(file)
                    print('training h5 has {}'.format(len(h5_list) -1 ))
                    random.shuffle(h5_list)
                    demo_list= demo_list + h5_list

            with open(demo_csv, 'w') as ft:
              for item in demo_list:
                  ft.write('{}\n'.format(item))


def write_pointcloud(filename,xyz_points,rgb_points=None):
    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'
    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))
    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffccc",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],rgb_points[i,0].tostring(),rgb_points[i,1].tostring(),rgb_points[i,2].tostring())))
    fid.close()


def get_pose(root_dset, obj_category, item, art_index, frame_order, mode='train', num_parts=5):
    # pose infos
    parts_world_pos    = [None]*num_parts
    parts_world_orn    = [None]*num_parts
    parts_model2world  = [None]*num_parts
    if mode == 'demo':
        sub_dir0        = root_dset  + '/demo/' + obj_category + '/' + item + '/' + art_index
    else:
        sub_dir0        = root_dset  + '/render/' + obj_category + '/' + item + '/' + art_index
    meta_file       = open(sub_dir0 + '/gt.yml', 'r')
    meta_instance   = yaml.load(meta_file)
    pose_dict = meta_instance['frame_{}'.format(frame_order)]
    viewMat   = np.array(pose_dict['viewMat']).reshape(4, 4).T
    projMat   = np.array(pose_dict['projMat']).reshape(4, 4).T
    #
    parts_world_pos[0] = np.array([0, 0, 0])
    parts_world_orn[0] = np.array([0, 0, 0, 1])
    for link in range(0, num_parts):
        # parts_urdf_box[link] = np.array(urdf_dict['link']['box'][link])
        if link >0:
            parts_world_pos[link] = np.array(pose_dict['obj'][link-1][4]).astype(np.float32)
            parts_world_orn[link] = np.array(pose_dict['obj'][link-1][5]).astype(np.float32)
    # manual correction
    if obj_category == 'bike':
        parts_world_pos[1], parts_world_pos[2] = parts_world_pos[2], parts_world_pos[1]
        parts_world_orn[1], parts_world_orn[2]  = parts_world_orn[2], parts_world_orn[1]

    # target rotation, translation, and target points
    for k in range(num_parts):
        # matrix computation
        center_world_orn   = parts_world_orn[k]
        center_world_orn   = np.array([center_world_orn[3], center_world_orn[0], center_world_orn[1], center_world_orn[2]])
        my_model2world_r   = quaternion_matrix(center_world_orn)[:4, :4] # [w, x, y, z]
        my_model2world_t   = parts_world_pos[k]
        my_model2world_mat = my_model2world_r
        for m in range(3):
            my_model2world_mat[m, 3] = my_model2world_t[m]
        my_world2camera_mat   = viewMat
        my_camera2clip_mat    = projMat
        my_model2camera_mat   = np.dot(my_world2camera_mat, my_model2world_mat)
        parts_model2world[k]  = my_model2world_mat

    return parts_model2world, viewMat, projMat

def get_urdf_mobility(inpath, verbose=True):
    urdf_ins = {}
    tree_urdf     = ET.parse(inpath + "/mobility.urdf") # todo
    num_real_links= len(tree_urdf.findall('link'))
    root_urdf     = tree_urdf.getroot()
    rpy_xyz       = {}
    list_xyz      = [None] * num_real_links
    list_rpy      = [None] * num_real_links
    list_box      = [None] * num_real_links
    list_obj      = [None] * num_real_links
    # ['obj'] ['link/joint']['xyz/rpy'] [0, 1, 2, 3, 4]
    num_links     = 0
    for link in root_urdf.iter('link'):
        num_links += 1
        index_link = None
        if link.attrib['name']=='base':
            index_link = 0
        else:
            index_link = int(link.attrib['name'].split('_')[1]) + 1 # since the name is base, link_0, link_1
        list_xyz[index_link] = []
        list_rpy[index_link] = []
        list_obj[index_link] = []
        for visual in link.iter('visual'):
            for origin in visual.iter('origin'):
                if 'xyz' in origin.attrib:
                    list_xyz[index_link].append([float(x) for x in origin.attrib['xyz'].split()])
                else:
                    list_xyz[index_link].append([0, 0, 0])
                if 'rpy' in origin.attrib:
                    list_rpy[index_link].append([float(x) for x in origin.attrib['rpy'].split()])
                else:
                    list_rpy[index_link].append([0, 0, 0])
            for geometry in visual.iter('geometry'):
                for mesh in geometry.iter('mesh'):
                    if 'home' in mesh.attrib['filename'] or 'work' in mesh.attrib['filename']:
                        list_obj[index_link].append(mesh.attrib['filename'])
                    else:
                        list_obj[index_link].append(inpath + '/' + mesh.attrib['filename'])

    rpy_xyz['xyz']   = list_xyz
    rpy_xyz['rpy']   = list_rpy # here it is empty list
    urdf_ins['link'] = rpy_xyz
    urdf_ins['obj_name'] = list_obj

    rpy_xyz       = {}
    list_type     = [None] * (num_real_links - 1)
    list_parent   = [None] * (num_real_links - 1)
    list_child    = [None] * (num_real_links - 1)
    list_xyz      = [None] * (num_real_links - 1)
    list_rpy      = [None] * (num_real_links - 1)
    list_axis     = [None] * (num_real_links - 1)
    list_limit    = [[0, 0]] * (num_real_links - 1)
    # here we still have to read the URDF file
    for joint in root_urdf.iter('joint'):
        joint_index            = int(joint.attrib['name'].split('_')[1])
        list_type[joint_index] = joint.attrib['type']

        for parent in joint.iter('parent'):
            link_name = parent.attrib['link']
            if link_name == 'base':
                link_index = 0
            else:
                link_index = int(link_name.split('_')[1]) + 1
            list_parent[joint_index] = link_index
        for child in joint.iter('child'):
            link_name = child.attrib['link']
            if link_name == 'base':
                link_index = 0
            else:
                link_index = int(link_name.split('_')[1]) + 1
            list_child[joint_index] = link_index
        for origin in joint.iter('origin'):
            if 'xyz' in origin.attrib:
                list_xyz[joint_index] = [float(x) for x in origin.attrib['xyz'].split()]
            else:
                list_xyz[joint_index] = [0, 0, 0]
            if 'rpy' in origin.attrib:
                list_rpy[joint_index] = [float(x) for x in origin.attrib['rpy'].split()]
            else:
                list_rpy[joint_index] = [0, 0, 0]
        for axis in joint.iter('axis'): # we must have
            list_axis[joint_index]= [float(x) for x in axis.attrib['xyz'].split()]
        for limit in joint.iter('limit'):
            list_limit[joint_index]= [float(limit.attrib['lower']), float(limit.attrib['upper'])]

    rpy_xyz['type']      = list_type
    rpy_xyz['parent']    = list_parent
    rpy_xyz['child']     = list_child
    rpy_xyz['xyz']       = list_xyz
    rpy_xyz['rpy']       = list_rpy
    rpy_xyz['axis']      = list_axis
    rpy_xyz['limit']     = list_limit


    urdf_ins['joint']    = rpy_xyz
    urdf_ins['num_links']= num_real_links
    if verbose:
        for j, pos in enumerate(urdf_ins['link']['xyz']):
            if len(pos) > 3:
                print('link {} xyz: '.format(j), pos[0])
            else:
                print('link {} xyz: '.format(j), pos)
        for j, orient in enumerate(urdf_ins['link']['rpy']):
            if len(orient) > 3:
                print('link {} rpy: '.format(j), orient[0])
            else:
                print('link {} rpy: '.format(j), orient)
        # for joint
        for j, pos in enumerate(urdf_ins['joint']['xyz']):
            print('joint {} xyz: '.format(j), pos)
        for j, orient in enumerate(urdf_ins['joint']['rpy']):
            print('joint {} rpy: '.format(j), orient)
        for j, orient in enumerate(urdf_ins['joint']['axis']):
            print('joint {} axis: '.format(j), orient)
        for j, child in enumerate(urdf_ins['joint']['child']):
            print('joint {} has child link: '.format(j), child)
        for j, parent in enumerate(urdf_ins['joint']['parent']):
            print('joint {} has parent link: '.format(j), parent)
        # plot_lines(urdf_ins['joint']['axis'])

    return urdf_ins

# todo
def get_urdf(inpath, num_real_links=None):
    urdf_ins = {}
    # urdf parameters
    tree_urdf     = ET.parse(inpath + "/syn.urdf") # todo
    if num_real_links is None:
        num_real_links = len(os.listdir(inpath)) - 1 # todo
    root_urdf     = tree_urdf.getroot()
    rpy_xyz       = {}
    list_xyz      = [None] * num_real_links
    list_rpy      = [None] * num_real_links
    list_box      = [None] * num_real_links
    list_obj      = [None] * num_real_links
    # ['obj'] ['link/joint']['xyz/rpy'] [0, 1, 2, 3, 4]
    num_links     = 0
    for link in root_urdf.iter('link'):
        num_links += 1
        index_link = None
        if link.attrib['name']=='base_link':
            index_link = 0
        else:
            index_link = int(link.attrib['name'])
        for visual in link.iter('visual'):
            for origin in visual.iter('origin'):
                list_xyz[index_link] = [float(x) for x in origin.attrib['xyz'].split()]
                list_rpy[index_link] = [float(x) for x in origin.attrib['rpy'].split()]
            for geometry in visual.iter('geometry'):
                for mesh in geometry.iter('mesh'):
                    list_obj[index_link] = mesh.attrib['filename']

    rpy_xyz['xyz']   = list_xyz
    rpy_xyz['rpy']   = list_rpy
    urdf_ins['link'] = rpy_xyz
    urdf_ins['obj_name'] = list_obj

    rpy_xyz       = {}
    list_xyz      = [None] * num_real_links
    list_rpy      = [None] * num_real_links
    list_axis     = [None] * num_real_links
    list_type     = [None] * num_real_links
    list_part     = [None] * num_real_links
    # here we still have to read the URDF file
    for joint in root_urdf.iter('joint'):
        index_child = int(joint.attrib['name'].split('_')[-1])
        index_parent= int(joint.attrib['name'].split('_')[0])
        list_type[index_child] = joint.attrib['type']
        list_part[index_child] = index_parent
        for origin in joint.iter('origin'):
            list_xyz[index_child] = [float(x) for x in origin.attrib['xyz'].split()]
            list_rpy[index_child] = [float(x) for x in origin.attrib['rpy'].split()]
        for axis in joint.iter('axis'):
            list_axis[index_child]= [float(x) for x in axis.attrib['xyz'].split()]
    rpy_xyz['xyz']       = list_xyz
    rpy_xyz['rpy']       = list_rpy
    rpy_xyz['axis']      = list_axis
    rpy_xyz['type']      = list_type
    rpy_xyz['parent']      = list_part

    urdf_ins['joint']    = rpy_xyz
    urdf_ins['num_links']= num_links

    return urdf_ins

def fetch_gt_bmvc(basepath, basename, num_parts=2):
    base_infos = basename.split('_')
    # Laptop_Seq_1_00020.h5
    pose_dict = {}
    BB_dict   = {}
    info_files = []
    for k in range(num_parts):
        info_files.append(basepath + '/{0}_{1}_{2}/info/info_{3}_{4:03d}.txt'.format(base_infos[0], base_infos[1], base_infos[2], base_infos[3], k))
    # print(info_files)
    for k, info_file in enumerate(info_files):
        with open(info_file, "r", errors='replace') as fp:
            line = fp.readline()
            cnt  = 1
            viewMat = np.eye(4)# from object coordinate to camera coordinate
            tight_bb = np.zeros((3))
            while line:
                if len(line.strip()) == 9 and line.strip()[:8] == 'rotation':
                    for i in range(3):
                        line = fp.readline()
                        viewMat[i, :3] = [float(x) for x in line.strip().split()]
                if len(line.strip()) == 7 and line.strip()[:6] == 'center':
                    line = fp.readline()
                    viewMat[:3, 3] = [float(x) for x in line.strip().split()]
                if len(line.strip()) == 7 and line.strip()[:6] == 'extent':
                    line = fp.readline()
                    tight_bb[:] = [float(x) for x in line.strip().split()]
                    break
                line = fp.readline()
        pose_dict[k] = viewMat
        BB_dict[k]   = tight_bb
    return pose_dict, BB_dict

def get_all_objs(root_dset, obj_category, item, obj_file_list=None, offsets=None, is_debug=False, verbose=False):
    """
    offsets is usually 0, but sometimes could be [x, y, z] in array 1*3, it could be made to a K*3 array if necessary
    """
    norm_factors = []
    pts_list     = []
    name_list    = []
    target_dir   = root_dset + '/objects/' + obj_category + '/' +  item

    offset = 0
    if obj_file_list is None:
        for k, obj_file in enumerate(glob.glob( target_dir + '/part_objs/*.obj')):
            if offsets is not None:
                offset = offsets[k:k+1, :]
            if is_debug:
                print('obj_file is: ', obj_file)
            try:
                tm = trimesh.load(obj_file)
                vertices_obj = np.array(tm.vertices)
            except:
                dict_mesh, _, _, _ = load_model_split(obj_file)
                vertices_obj = np.concatenate(dict_mesh['v'], axis=0)
            pts_list.append(vertices_obj + offset)
            name_obj  = obj_file.split('.')[0].split('/')[-1]
            name_list.append(name_obj)
    else:
        for k, obj_files in enumerate(obj_file_list):
            if offsets is not None:
                offset = offsets[k:k+1, :]
            if obj_files is not None and not isinstance(obj_files, list):
                try:
                    tm = trimesh.load(obj_files)
                    vertices_obj = np.array(tm.vertices)
                except:
                    dict_mesh, _, _, _ = load_model_split(obj_files)
                    vertices_obj = np.concatenate(dict_mesh['v'], axis=0)
                pts_list.append(vertices_obj + offset)
                name_obj  = obj_files.split('.')[0].split('/')[-1]
                name_list.append(name_obj) # which should follow the right order
            elif isinstance(obj_files, list):
                if verbose:
                    print('{} part has {} obj files'.format(k, len(obj_files)))
                part_pts = []
                name_objs = []
                for obj_file in obj_files:
                    if obj_file is not None and not isinstance(obj_file, list):
                        try:
                            tm = trimesh.load(obj_file)
                            vertices_obj = np.array(tm.vertices)
                        except:
                            dict_mesh, _, _, _ = load_model_split(obj_file)
                            vertices_obj = np.concatenate(dict_mesh['v'], axis=0)
                        name_obj  = obj_file.split('.')[0].split('/')[-1]
                        name_objs.append(name_obj)
                        part_pts.append(vertices_obj)
                part_pts_whole = np.concatenate(part_pts, axis=0)
                pts_list.append(part_pts_whole + offset)
                name_list.append(name_objs) # which should follow the right

    if is_debug:
        print('name_list is: ', name_list)

    parts_a    = []
    parts_a    = pts_list
    parts_b    = [None] * len(obj_file_list)
    # dof_rootd_Aa001_r.obj  dof_rootd_Aa002_r.obj  none_motion.obj
    # bike: part2: 'dof_Aa001_Ca001_r', 'dof_rootd_Aa001_r'
    if obj_category=='bike':
        part0    = []
        part1    = []
        part2    = []
        part0    = pts_list
        for i, name_obj in enumerate(name_list):
            if name_obj in ['dof_Aa001_Ca001_r', 'dof_rootd_Aa001_r']:
                print('part 2 adding ', name_obj)
                part2.append(pts_list[i])
            else:
                print('part 1 adding ', name_obj)
                part1.append(pts_list[i])
        parts      = [part0, part1, part2]

    elif obj_category=='eyeglasses':
        for i, name_obj in enumerate(name_list):
            if name_obj in ['none_motion']:
                parts_b[0] = []
                parts_b[0].append(pts_list[i])
            if name_obj in ['dof_rootd_Aa001_r']:
                parts_b[1] = []
                parts_b[1].append(pts_list[i])
            elif name_obj in ['dof_rootd_Aa002_r']:
                parts_b[2] = []
                parts_b[2].append(pts_list[i])

        parts      = [parts_a] +  parts_b

    else:
        parts_a    = []
        parts_a    = pts_list
        parts_b    = [None] * len(name_list)
        for i, name_obj in enumerate(name_list):
            parts_b[i] = []
            parts_b[i].append(pts_list[i])

        parts      = [parts_a] +  parts_b

    corner_pts = [None] * len(parts)

    for j in range(len(parts)):
        if is_debug:
            print('Now checking ', j)
        part_gts = np.concatenate(parts[j], axis=0)
        print('part_gts: ', part_gts.shape)
        tight_w = max(part_gts[:, 0]) - min(part_gts[:, 0])
        tight_l = max(part_gts[:, 1]) - min(part_gts[:, 1])
        tight_h = max(part_gts[:, 2]) - min(part_gts[:, 2])
        corner_pts[j] = np.amin(part_gts, axis=1)
        norm_factor = np.sqrt(1) / np.sqrt(tight_w**2 + tight_l**2 + tight_h**2)
        norm_factors.append(norm_factor)
        corner_pt_left = np.amin(part_gts, axis=0, keepdims=True)
        corner_pt_right= np.amax(part_gts, axis=0, keepdims=True)
        corner_pts[j]  = [corner_pt_left, corner_pt_right] # [index][left/right][x, y, z], numpy array
        if is_debug:
            print('Group {} has {} points with shape {}'.format(j, len(corner_pts[j]), corner_pts[j][0].shape))
        if verbose:
            plot3d_pts([[part_gts[::2]]], ['model pts'], s=15, title_name=['GT model pts {}'.format(j)], sub_name=str(j))
        # for k in range(len(parts[j])):
        #     plot3d_pts([[parts[j][k][::2]]], ['model pts of part {}'.format(k)], s=15, title_name=['GT model pts'], sub_name=str(k))

    return parts[1:], norm_factors, corner_pts

def calculate_factor_nocs(root_dset, obj_category, item, parts_map, obj_file_list=None, offsets=None, is_debug=False, verbose=True):
    """
    read all .obj files,
    group 1:  dof_rootd_Ba001_r.obj  dof_rootd_Ca002_r.obj  none_motion.obj;
    group 2:  dof_Aa001_Ca001_r.obj  dof_rootd_Aa001_r.obj;
    [global, part0, part1]
    """
    if obj_file_list is not None and obj_file_list[0] == []:
        obj_file_list = obj_file_list[1:]
    _, norm_factors, corner_pts = get_all_objs(root_dset, obj_category, item, obj_file_list=obj_file_list, offsets=offsets, is_debug=is_debug, verbose=False)
    if verbose:
        print('norm_factors for global NOCS: ', norm_factors[0])
        print('norm_factors for part NOCS: ', norm_factors[1:])
    return norm_factors, corner_pts

def get_model_pts(root_dset, obj_category, item='0001', obj_file_list=None, offsets=None, is_debug=False):
    """
    read all .obj files,
    group 1:  dof_rootd_Ba001_r.obj  dof_rootd_Ca002_r.obj  none_motion.obj;
    group 2:  dof_Aa001_Ca001_r.obj  dof_rootd_Aa001_r.obj;
    [global, part0, part1]
    """
    if obj_file_list is not None and obj_file_list[0] == []:
        print('removing the 0th name list')
        obj_file_list = obj_file_list[1:]
    model_pts, norm_factors, corner_pts = get_all_objs(root_dset, obj_category, item, obj_file_list=obj_file_list, offsets=offsets, is_debug=is_debug)
    # read
    return model_pts, norm_factors, corner_pts

def get_boundary(cpts):
    p = 0
    x_min = cpts[p][0][0][0]
    y_min = cpts[p][0][0][1]
    z_min = cpts[p][0][0][2]
    x_max = cpts[p][1][0][0]
    y_max = cpts[p][1][0][1]
    z_max = cpts[p][1][0][2]
    boundary = np.array([[x_min, y_min, z_min],
                           [x_max, y_min, z_min],
                           [x_max, y_min, z_max],
                           [x_min, y_min, z_max],
                           [x_min, y_max, z_min],
                           [x_min, y_max, z_max],
                           [x_max, y_max, z_max],
                          [x_max, y_max, z_min],
                          [x_min, y_min, z_max],
                          [x_min, y_max, z_max],
                          [x_max, y_max, z_max],
                          [x_max, y_min, z_max],
                          ]
                          )
    return boundary

def load_model_split(inpath):
    vsplit = []
    fsplit = []
    vcount = 0
    fcount = 0
    dict_mesh = {}
    list_group= []
    list_xyz  = []
    list_face = []
    with open(inpath, "r", errors='replace') as fp:
        line = fp.readline()
        cnt  = 1
        while line:
            # print('cnt: ', cnt, line)
            if len(line)<2:
                line = fp.readline()
                cnt +=1
                continue
            xyz  = []
            face = []
            mesh = {}
            if line[0] == 'g':
                list_group.append(line[2:])
            if line[0:2] == 'v ':
                vcount = 0
                while line[0:2] == 'v ':
                    xyz.append([float(coord) for coord in line[2:].strip().split()])
                    vcount +=1
                    line = fp.readline()
                    cnt  +=1
                vsplit.append(vcount)
                list_xyz.append(xyz)

            # it has intermediate g/obj
            if line[0] == 'f':
                fcount = 0
                while line[0] == 'f':
                    face.append([num for num in line[2:].strip().split()])
                    fcount +=1
                    line = fp.readline()
                    cnt +=1
                    if not line:
                        break
                fsplit.append(fcount)
                list_face.append(face)
            # print("Line {}: {}".format(cnt, line.strip()))
            line = fp.readline()
            cnt +=1
    # print('vsplit', vsplit, '\n', 'fsplit', fsplit)
    # print("list_mesh", list_mesh)
    dict_mesh['v'] = list_xyz
    dict_mesh['f'] = list_face
    vsplit_total   = sum(vsplit)
    fsplit_total   = sum(fsplit)

    return dict_mesh, list_group, vsplit, fsplit

def save_objmesh(name_obj, dict_mesh):
    with open(name_obj,"w+") as fp:
        for i in range(len(dict_mesh['v'])):
            xyz  = dict_mesh['v'][i]
            face = dict_mesh['f'][i]
            for j in range(len(xyz)):
                fp.write('v {} {} {}\n'.format(xyz[j][0], xyz[j][1], xyz[j][2]))
            for m in range(len(face)):
                fp.write('f {} {} {}\n'.format(face[m][0], face[m][1], face[m][2]))
            # fprintf(fid, 'vt %f %f\n',(i-1)/(l-1),(j-1)/(h-1));
            # if (normals) fprintf(fid, 'vn %f %f %f\n', nx(i,j),ny(i,j),nz(i,j)); end
            # Iterate vertex data collected in each material
            # for name, material in obj_model.materials.items():
            #     # Contains the vertex format (string) such as "T2F_N3F_V3F"
            #     # T2F, C3F, N3F and V3F may appear in this string
            #     material.vertex_format
            #     # Contains the vertex list of floats in the format described above
            #     material.vertices
            #     # Material properties
            #     material.diffuse
            #     material.ambient
            #     material.texture
        fp.write('g mesh\n');
        fp.write('g\n\n')

def save_multiobjmesh(name_obj, dict_mesh):
    with open(name_obj,"w+") as fp:
        for i in range(len(dict_mesh['v'])):
            xyz  = dict_mesh['v'][i]
            face = dict_mesh['f'][i]
            for j in range(len(xyz)):
                fp.write('v {} {} {}\n'.format(xyz[j][0], xyz[j][1], xyz[j][2]))
            for m in range(len(face)):
                fp.write('f {} {} {}\n'.format(face[m][0], face[m][1], face[m][2]))
            # fprintf(fid, 'vt %f %f\n',(i-1)/(l-1),(j-1)/(h-1));
            # if (normals) fprintf(fid, 'vn %f %f %f\n', nx(i,j),ny(i,j),nz(i,j)); end
            # Iterate vertex data collected in each material
            # for name, material in obj_model.materials.items():
            #     # Contains the vertex format (string) such as "T2F_N3F_V3F"
            #     # T2F, C3F, N3F and V3F may appear in this string
            #     material.vertex_format
            #     # Contains the vertex list of floats in the format described above
            #     material.vertices
            #     # Material properties
            #     material.diffuse
            #     material.ambient
            #     material.texture
        fp.write('g mesh\n')
        fp.write('g\n\n')
# mo:
#  {'type': 'R', 'movPart': 0.0, 'refPart': 1.0,
#  'origin': array([[ 0.022484 , -0.0113691,  0.11376  ]]),
#  'x': array([[-0.983804  , -0.00750761, -0.179091  ]]),
#  'y': array([[ 7.76757e-03, -9.99970e-01, -7.50390e-04]]),
#  'z': array([[-0.17908   , -0.00212934,  0.983832  ]]),
#  'motionDir': 'y', 'movPartExtentOnMotionDir': 0.0244756, 'min': -10.0, 'max': 80.0}
# mo:
#  {'type': 'R', 'movPart': 6.0, 'refPart': 0.0,
#  'origin': array([[-3.05619e-01,  2.57100e-05, -1.13209e-01]]),
#  'x': array([[-1.,  0.,  0.]]), 'y': array([[ 0., -1.,  0.]]), 'z': array([[0., 0., 1.]]),
#  'motionDir': 'y', 'movPartExtentOnMotionDir': 0.0222945,
#  'min': -180.0, 'max': 180.0}
def get_sampled_model_pts(basepath, urdf_path, args, viz=False):
    pts_m         = {}
    bbox3d_all    = {}
    start         = time.time()
    m_file        = basepath + '/shape2motion/pickle/{}_pts.pkl'.format(args.item)
    c_file        = basepath + '/shape2motion/pickle/{}_corners.pkl'.format(args.item)
    n_file        = basepath + '/shape2motion/pickle/{}.pkl'.format(args.item)

    if args.process:
        root_dset = basepath + '/shape2motion'
        for item in os.listdir(urdf_path):
            print('now fetching for item {}'.format(item))
            pts, nf, cpts = get_model_pts(root_dset, args.item, item)
            pt_ii         = []
            bbox3d_per_part = []
            for p, pt in enumerate(pts):
                pt_s = np.concatenate(pt, axis=0)
                np.random.shuffle(pt_s)
                # pt_s = pt_s[::20, :]
                pt_ii.append(pt_s)
                print('We have {} pts'.format(pt_ii[p].shape[0]))
            if pt_ii is not []:
                pts_m[item] = pt_ii
            else:
                print('!!!!! {} model loading is wrong'.format(item))
        end_t          = time.time()

        with open(m_file, 'wb') as f:
            pickle.dump(pts_m, f)
    else:
        with open(m_file, 'rb') as f:
            pts_m = pickle.load(f)

        with open(c_file, 'rb') as f:
            pts_c = pickle.load(f)

        with open(n_file, 'rb') as f:
            pts_n = pickle.load(f)

        for item in list(pts_m.keys()):
            pts  = pts_m[item]
            norm_factors = pts_n[item]
            norm_corners = pts_c[item]
            pt_ii  = []
            bbox3d_per_part = []
            for p, pt in enumerate(pts): # todo: assume we are dealing part-nocs, so model pts are processed
                norm_factor = norm_factors[p+1]
                norm_corner = norm_corners[p+1]
                nocs_corner = np.copy(norm_corner) # copy is very important, as they are
                print('norm_corner:\n', norm_corner)
                pt_nocs = (pt- norm_corner[0]) * norm_factor + np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (  norm_corner[1] - norm_corner[0]) * norm_factor
                nocs_corner[0] = np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (norm_corner[1] - norm_corner[0]) * norm_factor
                nocs_corner[1] = np.array([0.5, 0.5, 0.5]).reshape(1, 3) + 0.5 * (norm_corner[1] - norm_corner[0]) * norm_factor
                bbox3d_per_part.append(nocs_corner)
                np.random.shuffle(pt_nocs)
                pt_ii.append(pt_nocs[0:2000, :])  # sub-sampling
                print('We have {} pts'.format(pt_ii[p].shape[0]))
            if pt_ii is not []:
                pts_m[item] = pt_ii
            else:
                print('!!!!! {} model loading is wrong'.format(item))
            assert bbox3d_per_part != []
            bbox3d_all[item] = bbox3d_per_part

        end_t          = time.time()
    if viz:
        print('It takes {} seconds to get: \n'.format(end_t - start), list(pts_m.keys()))
    return bbox3d_all, pts_m


def get_part_bounding_box(my_dir, test_ins, args, viz=False):
    pts_m         = {}
    bbox3d_all    = {}
    start         = time.time()
    m_file        = my_dir + '/shape2motion/pickle/{}_pts.pkl'.format(args.item)
    c_file        = my_dir + '/shape2motion/pickle/{}_corners.pkl'.format(args.item)
    n_file        = my_dir + '/shape2motion/pickle/{}.pkl'.format(args.item)

    if args.process:
        for item in os.listdir(test_ins):
            print('now fetching for item {}'.format(item))
            pts, nf, cpts = get_model_pts(root_dset, args.item, item)
            pt_ii  = []
            for p, pt in enumerate(pts):
                pt_s = np.concatenate(pt, axis=0)
                np.random.shuffle(pt_s)
                # pt_s = pt_s[::20, :]
                pt_ii.append(pt_s)
                print('We have {} pts'.format(pt_ii[p].shape[0]))
            if pt_ii is not []:
                pts_m[item] = pt_ii
            else:
                print('!!!!! {} model loading is wrong'.format(item))
        end_t          = time.time()

        with open(m_file, 'wb') as f:
            pickle.dump(pts_m, f)
    else:
        with open(m_file, 'rb') as f:
            pts_m = pickle.load(f)

        with open(c_file, 'rb') as f:
            pts_c = pickle.load(f)

        with open(n_file, 'rb') as f:
            pts_n = pickle.load(f)

        # for item in list(pts_m.keys()):
        for item in test_ins:
            pts  = pts_m[item]
            norm_factors = pts_n[item]
            norm_corners = pts_c[item]
            pt_ii  = []
            bbox3d_per_part = []
            for p, pt in enumerate(pts): # todo: assume we are dealing part-nocs, so model pts are processed
                norm_factor = norm_factors[p+1]
                norm_corner = norm_corners[p+1]
                nocs_corner = np.copy(norm_corner) # copy is very important, as they are
                print('norm_corner:\n', norm_corner)
                pt_nocs = (pt- norm_corner[0]) * norm_factor + np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (  norm_corner[1] - norm_corner[0]) * norm_factor
                nocs_corner[0] = np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (norm_corner[1] - norm_corner[0]) * norm_factor
                nocs_corner[1] = np.array([0.5, 0.5, 0.5]).reshape(1, 3) + 0.5 * (norm_corner[1] - norm_corner[0]) * norm_factor
                bbox3d_per_part.append(nocs_corner)
                np.random.shuffle(pt_nocs)
                pt_ii.append(pt_nocs[0:2000, :])  # sub-sampling
                print('We have {} pts'.format(pt_ii[p].shape[0]))
            if pt_ii is not []:
                pts_m[item] = pt_ii
            else:
                print('!!!!! {} model loading is wrong'.format(item))
            assert bbox3d_per_part != []
            bbox3d_all[item] = bbox3d_per_part
        end_t          = time.time()

    print('It takes {} seconds to get: \n'.format(end_t - start), list(pts_m.keys()))
    if viz:
        plot3d_pts([pts_m['0001']], [['part 0', 'part 1']], s=5, title_name=['sampled model pts'], dpi=200)

    return bbox3d_all, pts_m

def get_test_seq(all_test_h5, unseen_instances, domain='seen', spec_instances=[], category=None):
    seen_test_h5    = []
    unseen_test_h5  = []
    for test_h5 in all_test_h5:
        if test_h5[0:4] in spec_instances or test_h5[-2:] !='h5':
            continue
        name_info      = test_h5.split('.')[0].split('_')
        item           = name_info[0]
        if item in unseen_instances:
            unseen_test_h5.append(test_h5)
        else:
            seen_test_h5.append(test_h5)
    if domain == 'seen':
        test_group = seen_test_h5
    else:
        test_group = unseen_test_h5

    return test_group

def get_test_group(all_test_h5, unseen_instances, domain='seen', spec_instances=[], category=None):
    seen_test_h5    = []
    unseen_test_h5  = []
    seen_arti_select = list(np.arange(0, 31, 3)) # todo,  15 * 1 * [24 - 83], half
    seen_arti_select = [str(x) for x in seen_arti_select]

    unseen_frame_select =  list(np.arange(0, 30, 5)) # todo, 6 * 31 * 3
    unseen_frame_select =  [str(x) for x in unseen_frame_select]
    for test_h5 in all_test_h5:
        if test_h5[0:4] in spec_instances or test_h5[-2:] !='h5':
            continue
        name_info      = test_h5.split('.')[0].split('_')
        item           = name_info[0]
        art_index      = name_info[1]
        frame_order    = name_info[2]

        if item in unseen_instances and frame_order in unseen_frame_select :
            unseen_test_h5.append(test_h5)
        elif item not in unseen_instances and art_index in seen_arti_select:
            seen_test_h5.append(test_h5)

    if domain == 'seen':
        test_group = seen_test_h5
    else:
        test_group = unseen_test_h5

    return test_group

def get_full_test(all_test_h5, unseen_instances, domain='seen', spec_instances=[], category=None):
    seen_test_h5    = []
    unseen_test_h5  = []
    for test_h5 in all_test_h5:
        if test_h5[0:4] in spec_instances or test_h5[-2:] !='h5':
            continue
        name_info      = test_h5.split('.')[0].split('_')
        item           = name_info[0]
        art_index      = name_info[1]
        frame_order    = name_info[2]

        if item in unseen_instances:
            unseen_test_h5.append(test_h5)
        elif item not in unseen_instances:
            seen_test_h5.append(test_h5)

    if domain == 'seen':
        test_group = seen_test_h5
    else:
        test_group = unseen_test_h5

    return test_group


def get_demo_h5(all_test_h5, spec_instances=[]):
    demo_full_h5    = []
    for test_h5 in all_test_h5:
        if test_h5[0:4] in spec_instances or test_h5[-2:] !='h5':
            continue
        demo_full_h5.append(test_h5)

    return demo_full_h5

def get_pickle(data, base_path, index):
    """
    data: better to be dict-like structure
    """
    file_name = base_path + '/pickle/datapoint_{}.pkl'.format(index)
    directory = base_path + '/pickle'
    if not os.path.exists( directory ):
        os.makedirs(directory)
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
    print('Saving the data into ' + base_path + '/pickle/datapoint_{}.pkl'.format(index))


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)

    return data


if __name__ == '__main__':
    pass
