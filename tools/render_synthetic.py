"""
Func: data rendering from URDF
      it will render data with:
    - rgb image;
    - depth image;
    - part masks;
    - pose labels;
    - joint states;
# make sure to compile pybullet with PYBULLET_USE_NUMPY enabled
"""
import numpy as np
import pybullet

# here we add one
import sys
import os
import time
import cv2
import h5py
import yaml
import argparse
import threading
import xml.etree.ElementTree as ET
from collections import OrderedDict
from yaml import CLoader as Loader, CDumper as Dumper
from yaml.representer import SafeRepresenter
_mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

def dict_representer(dumper, data):
    return dumper.represent_dict(data.items())
def dict_constructor(loader, node):
    return OrderedDict(loader.construct_pairs(node))

Dumper.add_representer(OrderedDict, dict_representer)
Loader.add_constructor(_mapping_tag, dict_constructor)
Dumper.add_representer(str, SafeRepresenter.represent_str)

# custom libs
import _init_paths
from global_info import global_info
from lib.data_utils import get_model_pts, get_urdf

# Step through simulation time
def step_simulation():
    # self._sim_time = 0
    while True:
        pybullet.stepSimulation()
        # self._sim_time += self._sim_timestep
        time.sleep(0.01)
        
#>>>>>>>>>>>>>>>>>>>>>>---------Rendering setup----------<<<<<<<<<<<<<<<<<<<<<<<<<#
def render_data(data_root, name_obj, cur_urdf, args=None, cam_dis=1, urdf_file='NA', _WRITE_FLAG=True, _RENDER_FLAG=True, _CREATE_FOLDER=True, RENDER_NUM=100, ARTIC_CNT=20, _RENDER_MODE='random', _USE_GUI=True, _IS_DUBUG=True):
    #>>>>>>>>>>>>>>>>>>>>>>>>>> internal config >>>>>>>>>>>>>>>>>>>>>>#
    save_path     = data_root + '/render/' + name_obj
    path_urdf = data_root + '/urdf/' + name_obj 
    roll          = 0
    steeringAngle = 0
    camPosX       = 0
    camPosY       = 0
    camPosZ       = 0

    upAxisIndex = 2 # align with z
    camDistance = 0
    pixelWidth  = 512
    pixelHeight = 512
    nearPlane   = 0.01
    farPlane    = 10
    fov         = 75
    cameraUp     = [0, 0.5*np.sqrt(3), 0.5]      # z axis
    cameraPos    = [-1.1, -1.1, 1.1]
    #>>>>>>>>>>>>>>>>>>>>>>>>> internal config end >>>>>>>>>>>>>>>>>>>#
    pitch_low, pitch_high  = [int(x) for x in args.pitch.split(',')]
    yaw_low, yaw_high      = [int(x) for x in args.yaw.split(',')]
    max_angles = [float(x)/180 * np.pi for x in args.max_angles.split(',')]
    min_angles = [float(x)/180 * np.pi for x in args.min_angles.split(',')]
    print('we are setting pitch and yaw to: ', pitch_low, pitch_high, yaw_low, yaw_high)
    if not _WRITE_FLAG:
        camInfo  = pybullet.getDebugVisualizerCamera()

    tree_urdf = ET.parse("{}/{}/syn.urdf".format(path_urdf, cur_urdf))
    root      = tree_urdf.getroot()

    num_joints = 0
    num_joints = len(os.listdir("{}/{}/".format(path_urdf, cur_urdf))) -2

    obj_parts = []
    pybullet.setGravity(0, 0, -10)

    for i in range(num_joints+1): #
        urdf_file = "{}/{}/syn_p{}.urdf".format(path_urdf, cur_urdf, i)
        print('loading ', urdf_file)
        obj_p = pybullet.loadURDF(urdf_file)
        obj_parts.append(obj_p)
        if i == 0:
            for joint in range(pybullet.getNumJoints(obj_parts[i])):
                print("joint[",joint,"]=", pybullet.getJointInfo(obj_parts[i], joint))
                pybullet.setJointMotorControl2(obj_parts[i], joint, pybullet.VELOCITY_CONTROL, targetVelocity=0,force=0)
                pybullet.getJointInfo(obj_parts[i], joint)

    simu_cnt   = 0
    main_start = time.time()

    urdf_ins   = get_urdf("{}/{}".format(path_urdf, cur_urdf))
    num_joints = len(urdf_ins['obj_name']) -1

    # instance-wise offset for camera distance
    try: 
        model_pts, norm_factors, corner_pts = get_model_pts(data_root, name_obj, cur_urdf, obj_file_list=urdf_ins['obj_name'])
        center_pts = [(x[0] + x[1])/2 for x in corner_pts]
        tight_bb   = corner_pts[0][1] - corner_pts[0][0] # the size of this objects
        min_dis    = np.linalg.norm(tight_bb) /2 * np.tan(fov/180 /2 * np.pi) # todo
        offset     = min_dis / 2
    except:
        offset     = 0.5

    if _RENDER_MODE == 'random':
        steeringAngleArray          = np.random.rand(ARTIC_CNT, num_joints) * np.array([max_angles]).reshape(-1, num_joints)
    elif _RENDER_MODE == 'linear':
        steeringAngleArray          = np.tile(np.linspace(0, 1, ARTIC_CNT).reshape(-1, 1), (1, num_joints)) * (np.array([max_angles]).reshape(-1) - np.array([min_angles]).reshape(-1)) + np.array([min_angles]).reshape(-1)

    rdn_offset                  = 2 * offset * (np.random.rand(ARTIC_CNT, RENDER_NUM) - 0.5)        # -0.4, 0.4
    lightDirectionArray         = 10  * np.random.rand(ARTIC_CNT, RENDER_NUM, 3)             # coming direction of light
    lightDistanceArray          = 0.9   + 0.2  * np.random.rand(ARTIC_CNT, RENDER_NUM)
    lightColorArray             = 0.9 + 0.1 * np.random.rand(ARTIC_CNT, RENDER_NUM, 3)
    lightSpecularCoeffArray     = 0.85 + 0.1 * np.random.rand(ARTIC_CNT, RENDER_NUM)
    lightAmbientCoeffArray      = 0.1  + 0.2 * np.random.rand(ARTIC_CNT, RENDER_NUM)
    lightDiffuseCoeffArray      = 0.85 + 0.1 * np.random.rand(ARTIC_CNT, RENDER_NUM)

    # get joint state
    while (simu_cnt < ARTIC_CNT):
        if (not os.path.exists(save_path + '/{}/{}/depth/'.format(cur_urdf, simu_cnt))) and _CREATE_FOLDER:
            os.makedirs(save_path + '/{}/{}/depth/'.format(cur_urdf, simu_cnt))
            os.makedirs(save_path + '/{}/{}/rgb/'.format(cur_urdf, simu_cnt))
            os.makedirs(save_path + '/{}/{}/mask/'.format(cur_urdf, simu_cnt))
        yml_dict = OrderedDict()
        yml_file = save_path + '/{}/{}/gt.yml'.format(cur_urdf, simu_cnt)

        # set articulation status
        print('Rendering with joint angles {}'.format(steeringAngleArray[simu_cnt, :]*180 /np.pi))
        for steer in range(num_joints):
            steeringAngle = steeringAngleArray[simu_cnt, steer]
            for j in range(num_joints+1): #
                pybullet.setJointMotorControl2(obj_parts[j], steer, pybullet.POSITION_CONTROL, targetVelocity=0, targetPosition=steeringAngle)
                start = time.time()
                # time.sleep(0.2)

        # make sure the angle matches with set value
        lstate0 = pybullet.getLinkState(obj_parts[0], linkIndex=pybullet.getNumJoints(obj_parts[0]) - 1, computeForwardKinematics=True)
        for m in range(10):
            time.sleep(0.1)
            lstate1 = pybullet.getLinkState(obj_parts[0], linkIndex=pybullet.getNumJoints(obj_parts[0]) - 1, computeForwardKinematics=True)
            q0 = lstate0[5]
            q1 = lstate1[5]
            angle_diff = 2 * np.arccos( min(1, np.sum(np.array(q0) * np.array(q1)) ) ) / np.pi * 180
            # print('angle difference is: ', angle_diff)
            if angle_diff < 0.05:
                break
            lstate0 = lstate1

        img_id = 0
        lastTime = time.time()
        view_num = 100
        pitch_choices = pitch_low + (pitch_high - pitch_low) *np.random.rand(view_num)
        yaw_choices   = yaw_low   + (yaw_high - yaw_low) * np.random.rand(view_num)
        for i in range(view_num):
            pitch = pitch_choices[i]
            yaw   = yaw_choices[i]
            if(img_id < RENDER_NUM and _RENDER_FLAG):
                camTargetPos = 0.8 * min_dis * (np.random.rand(3) - 0.5) + center_pts[0]
                nowTime = time.time()
                offset                 = rdn_offset[simu_cnt, img_id]
                lightDirection         = lightDirectionArray[simu_cnt, img_id, :]
                lightDistance          = lightDistanceArray[simu_cnt, img_id]
                lightColor             = list(lightColorArray[simu_cnt, img_id, :])
                lightAmbientCoeff      = lightAmbientCoeffArray[simu_cnt, img_id]
                lightDiffuseCoeff      = lightDiffuseCoeffArray[simu_cnt, img_id]
                lightSpecularCoeff     = lightSpecularCoeffArray[simu_cnt, img_id]

                camDistance_final      = min_dis * 2.8 + offset # [1.6, 2.6] * min_dis
                viewMatrix             = pybullet.computeViewMatrixFromYawPitchRoll(camTargetPos[0], camDistance_final, yaw, pitch, roll, upAxisIndex)
                aspect                 = pixelWidth / pixelHeight
                projectionMatrix       = pybullet.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
                img_arr = pybullet.getCameraImage(pixelWidth, pixelHeight, viewMatrix, projectionMatrix, lightDirection=lightDirection,\
                                                  renderer= pybullet.ER_BULLET_HARDWARE_OPENGL)

                w         = img_arr[0]
                h         = img_arr[1]
                rgb       = img_arr[2]
                depth_raw = img_arr[3].astype(np.float32)
                mask      = img_arr[4]
                depth     = 255.0 * nearPlane / (farPlane - (farPlane - nearPlane) * depth_raw) # *farPlane/255.0
                far       = farPlane
                near      = nearPlane
                depth_to_save = 2.0 * far * near / (far  + near - (far - near) * (2 * depth_raw - 1.0))

                np_rgb_arr  = np.reshape(rgb, (h, w, 4))[:, :, :3]
                np_depth_arr= np.reshape(depth, (h, w, 1))
                np_mask_arr = (np.reshape(mask, (h, w, 1))).astype(np.uint8)
                image_path  = save_path + '/{}/{}'.format(cur_urdf, simu_cnt)

                rgb_name   = image_path + '/rgb/{0:06d}.png'.format(img_id)
                depth_img_name   = image_path + '/depth/{0:06d}.png'.format(img_id)
                depth_name   = image_path + '/depth/{0:06d}.h5'.format(img_id)
                mask_name  = image_path + '/mask/{0:06d}.png'.format(img_id)

                if i == 0:
                    joint_pos = OrderedDict()
                    for joint in range(pybullet.getNumJoints(obj_parts[0])):
                        lstate = pybullet.getLinkState(obj_parts[0], linkIndex=joint, computeForwardKinematics=True)
                        joint_pos[joint] = OrderedDict(
                                            [(0, list(lstate[0])),
                                             (1, list(lstate[1])),
                                             (2, list(lstate[2])),
                                             (3, list(lstate[3])),
                                             (4, list(lstate[4])),
                                             (5, list(lstate[5]))]
                        )
                        # print('Joint {} lstate under {} : \n'.format(joint, steeringAngleArray[simu_cnt, :]), lstate[4:6])

                if _WRITE_FLAG is True:
                    cv2.imwrite(rgb_name, np_rgb_arr)
                    cv2.imwrite(depth_img_name, np_depth_arr)
                    cv2.imwrite(mask_name, np_mask_arr)
                    hf = h5py.File(depth_name, 'w')
                    hf.create_dataset('data', data=depth_to_save)
                yml_dict['frame_{}'.format(img_id)] = OrderedDict( [ ('obj', joint_pos),
                                                  ('viewMat', list(viewMatrix)),
                                                  ('projMat', list(projectionMatrix))
                                                  ])
                if not _WRITE_FLAG:
                    time.sleep(1)
                img_id+=1
                lastTime = nowTime

        if _WRITE_FLAG:
            with open(yml_file, 'w') as f:
                yaml.dump(yml_dict, f, default_flow_style=False)

        simu_cnt      += 1
        stop = time.time()

    main_stop = time.time()
    print ("Total time %f" % (main_stop - main_start))
    pybullet.resetSimulation()


if __name__ == "__main__":
    #>>>>>>>>>>>>>>>>>>>>>>>>>> config regions >>>>>>>>>>>>>>>>>>>>>>>>#
    infos     = global_info()
    my_dir    = infos.base_path

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='indicating whether in debug mode')
    parser.add_argument('--dataset', default='shape2motion', help='name of the dataset we use')
    parser.add_argument('--item', default='eyeglasses', help='name of category we use')
    parser.add_argument('--dis',   default=3, help='default camera2object distance')
    parser.add_argument('--mode',  default='train', help='mode decides saving folder:train/demo')
    parser.add_argument('--roll', default='30,40', help='camera view angle', required=True)
    parser.add_argument('--pitch', default='30,40', help='camera view angle', required=True)
    parser.add_argument('--yaw',  default='30,40', help='camera view angle', required=True)
    parser.add_argument('--min_angles',  default='30,40,50', help='minimum joint angles', required=True)
    parser.add_argument('--max_angles',  default='30,40,50', help='maximum joint angles', required=True)
    parser.add_argument('--cnt', default=30, help='count of articulation change')
    parser.add_argument('--num', default=10, help='number of rendering per articulation')
    args = parser.parse_args()
    #>>>>>>>>>>>>>>>>>>>>>>>> config end here >>>>>>>>>>>>>>>>>>>>>>>>>#

    is_debug = args.debug
    if is_debug:
        _WRITE   = False
        _RENDER  = True
        _CREATE  = True
        _USE_GUI = True
    else:
        _WRITE   = True
        _RENDER  = True
        _CREATE  = True
        _USE_GUI = False

    num_render     = int(args.num)                                               # viewing angles
    cnt_artic      = int(args.cnt)
    cam_dis        = float(args.dis)                                             # articulation change
    name_dataset   = args.dataset

    data_root = my_dir + '/dataset/' + name_dataset
    all_ins   = os.listdir(data_root + '/urdf/' + args.item) 
    all_ins.sort()

    if _USE_GUI:
        pybullet.connect(pybullet.GUI)
        pybullet.setRealTimeSimulation(1) # does not work with p.DIRECT
    else:
        pybullet.connect(pybullet.DIRECT)
        step_sim_thread = threading.Thread(target=step_simulation)
        step_sim_thread.daemon = True
        step_sim_thread.start()

    np.random.seed(5) # better to have this random seed here
    if is_debug:
        for instance in all_ins:
            render_data(data_root, args.item, instance, cam_dis=cam_dis, args=args,  _WRITE_FLAG=_WRITE, _RENDER_FLAG=_RENDER, _CREATE_FOLDER=_CREATE, RENDER_NUM=num_render, ARTIC_CNT=cnt_artic, _USE_GUI=_USE_GUI, _IS_DUBUG=is_debug)
    else:
        for instance in all_ins: #todo
            render_data(data_root, args.item, instance, cam_dis=cam_dis, args=args,  _WRITE_FLAG=_WRITE, _RENDER_FLAG=_RENDER, _CREATE_FOLDER=_CREATE, RENDER_NUM=num_render, ARTIC_CNT=cnt_artic, _USE_GUI=_USE_GUI, _IS_DUBUG=is_debug)
