"""
func: parse articulation infos from json into URDF files used by Pybullet
    - traverse over path and objects
    - parse from json file(dict, list, array);
    - obj files path;
    - inverse computing on rpy/xyz;
    - xml writer to urdf
"""
import os
import os.path
import glob
import sys
import time
import json
import copy
import numpy as np
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, tostring, SubElement, Comment, ElementTree, XML
import xml.dom.minidom

import _init_paths
from global_info import global_info

def iterdict(d):
    for k,v in d.items():
        if k == 'children':
            if v is not None:
                for child in v:
                    iterdict(child)
        else:
            print (k,":",v)

def traverse_dict(d, link_dict, joint_dict):
    """
    link_dict  = {} # name - attributes;
    joint_list = [] # [{parent-child}, {}, {}]
    link: name + all attrs;
    joints: parent + child;
    """
    link            = {}
    joint           = {}
    for k, v in d.items():
        if k != 'children':
            link[k] = v
        else:
            if v is not None:
                for child in v:
                    traverse_dict(child, link_dict, joint_dict)
                    joint_dict[child['dof_name']] = d['dof_name']
    link_dict[d['dof_name']] = link


if __name__ == '__main__':
    #>>>>>>>>>>>>>>>> you only need to change this part >>>>>>>>>>>>>
    is_debug  = False
    dataset   = 'shape2motion' 
    infos     = global_info()
    my_dir    = infos.base_path
    base_path = my_dir + '/dataset/' # with a lot of different objects, and then 0001/0002/0003/0004/0005
    #>>>>>>>>>>>>>>>>>>>>>>>> config end >>>>>>>>>>>>>>>>>>>>>>>>>>>># 

    all_objs     = os.listdir( base_path  + dataset  + '/objects' )
    print('all_objs are: ', all_objs)
    object_nums = []
    object_joints = []
    with open(base_path + '/' + dataset + '/statistics.txt', "a+") as f:
        for obj_n in all_objs:
            f.write('{}\t'.format(obj_n))
        f.write('\n')
    for obj_name in all_objs:
        object_joints_per = {}
        instances_per_obj = sorted(glob.glob(base_path  + dataset  + '/objects/' + obj_name + '/*'))
        obj_num =len(instances_per_obj)
        object_nums.append(obj_num)
        if is_debug:
            selected_objs = instances_per_obj[0:1]
        else:
            selected_objs = instances_per_obj
        for sub_dir in selected_objs:                        # regular expression
            #
            instance_name= sub_dir.split('/')[-1]
            print('Now working on {}, with instance {}'.format(obj_name, instance_name))
            save_dir     = base_path + '/' + dataset  + '/urdf/{}/{}'.format(obj_name, instance_name) # todo
            json_name    = glob.glob(sub_dir + '/*.json')
            with open(json_name[0]) as json_file:
                motion_attrs = json.load(json_file)
                # print(json.dumps(motion_attrs, sort_keys=True, indent=4))
                link_dict  = {} # name - attributes;
                joint_dict = {} # child- parent
                joint_list = [] #
                link_list  = [] #
                # dict is a better choice for this kind of iteration
                traverse_dict(motion_attrs, link_dict, joint_dict)
                for child, parent in joint_dict.items():
                    joint = {}
                    joint['parent'] = parent
                    joint['child']  = child
                    joint_list.append(joint)
                # for link, params_link in link_dict.items():
                keys_link = ['dof_rootd'] +  list(joint_dict.keys()) #
                #>>>>>>>>>>>>>>>>>> contruct links and joints
                root  = Element('robot', name="block")
                num   = len(joint_list) + 1
                links_name = ["base_link"] + [str(i+1) for i in range(num)]
                all_kinds_joints = ["revolute", "fixed", "prismatic", "continuous", "planar"]
                joints_name = []
                joints_type = []
                joints_pos  = []
                links_pos   = [None] * num
                joints_axis = []
                # parts connection
                rotation_joint = 0
                translation_joint = 0
                for i in range(num-1):
                    child_name  = joint_list[i]['child']
                    index_parent= keys_link.index(joint_list[i]['parent'])
                    index_child = keys_link.index(child_name)
                    assert index_child==i+1
                    child_obj   = link_dict[ child_name ]
                    vector_pos  = np.array(child_obj['center'])
                    links_pos[index_child] = -vector_pos
                    joints_name.append("{}_j_{}".format(index_parent, index_child))
                    joints_axis.append(child_obj['direction'])
                    if child_obj['motion_type'] == "rotation":
                        joints_type.append('revolute')
                        rotation_joint +=1
                    else:
                        joints_type.append('prismatic')
                        translation_joint +=1
                    while joint_dict[child_name] !='dof_rootd':
                        print('joint {} now looking at child {}, has parent {}'.format(i, child_name, joint_dict[child_name]))
                        child_name  = joint_dict[ child_name ]
                        child_obj   = link_dict[ child_name ]
                        vector_pos  = vector_pos - np.array(child_obj['center'])
                    joints_pos.append(vector_pos)
                object_joints_per['revolute']  = rotation_joint
                object_joints_per['prismatic'] = translation_joint
                # >>>>>>>>>>> start parsing urdf,
                children = [
                    Element('link', name=links_name[i])
                    for i in range(num)
                    ]
                joints = [
                    Element('joint', name=joints_name[i], type=joints_type[i])
                    for i in range(num-1)
                    ]
                # add inertial component
                node_inertial = XML('''<inertial><origin rpy="0 0 0" xyz="0 0 0"/><mass value="1.0"/><inertia ixx="0.9" ixy="0.9" ixz="0.9" iyy="0.9" iyz="0" izz="0.9"/></inertial>''')
                #>>>>>>>>>>>. 1. links
                for i in range(num):
                    visual   = SubElement(children[i], 'visual')
                    dof_name = link_dict[keys_link[i]]['dof_name']
                    if dof_name == 'dof_rootd':
                        origin   = SubElement(visual, 'origin', rpy="0.0 0.0 0.0", xyz="0 0 0")
                    else:
                        origin   = SubElement(visual, 'origin', rpy="0.0 0.0 0.0", xyz="{} {} {}".format(links_pos[i][0], links_pos[i][1], links_pos[i][2]))
                    geometry = SubElement(visual, 'geometry')
                    if i == 0 :
                        mesh     = SubElement(geometry, 'mesh', filename="{}/part_objs/none_motion.obj".format(sub_dir))
                    else:
                        mesh     = SubElement(geometry, 'mesh', filename="{}/part_objs/{}.obj".format(sub_dir, dof_name))
                    # materials assignment
                    inertial = SubElement(children[i], 'inertial')
                    node_inertial = XML('''<inertial><origin rpy="0 0 0" xyz="0 0 0"/><mass value="3.0"/><inertia ixx="100" ixy="100" ixz="100" iyy="100" iyz="100" izz="100"/></inertial>''')
                    inertial.extend(node_inertial)
                    if i == 0:
                        for mass in inertial.iter('mass'):
                            mass.set('value', "0.0")
                        for inertia in inertial.iter('inertia'):
                            inertia.set('ixx', "0.0")
                            inertia.set('ixy', "0.0")
                            inertia.set('ixz', "0.0")
                            inertia.set('iyy', "0.0")
                            inertia.set('iyz', "0.0")
                            inertia.set('izz', "0.0")
                #>>>>>>>>>>> 2. joints
                for i in range(num - 1):
                    index_parent= keys_link.index(joint_list[i]['parent'])
                    index_child = keys_link.index(joint_list[i]['child'])
                    parent = SubElement(joints[i], "parent", link=links_name[index_parent])
                    child  = SubElement(joints[i], "child",  link=links_name[index_child])
                    origin = SubElement(joints[i], "origin", xyz="{} {} {}".format(joints_pos[i][0], joints_pos[i][1], joints_pos[i][2]), rpy="0 0 0")
                    # we may need to change the joint name
                    if joints_type[i]=='revolute':
                        axis = SubElement(joints[i], "axis", xyz="{} {} {}".format(joints_axis[i][0], joints_axis[i][1], joints_axis[i][2]))
                        limit= SubElement(joints[i], "limit", effort="1.0", lower="-3.1415", upper="3.1415", velocity="1000")
                #>>>>>>>>>>>> 3. construct the trees
                root.extend(children)
                root.extend(joints)
                xml_string = xml.dom.minidom.parseString(tostring(root))
                xml_pretty_str = xml_string.toprettyxml()
                # print(xml_pretty_str)
                tree = ET.ElementTree(root)
                # save
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                with open(save_dir + '/syn.urdf', "w") as f:
                    print('writing to ', save_dir + '/syn.urdf')
                    f.write(xml_pretty_str)
                #>>>>>>>>>>>>>>>>> coding >>>>>>>>>>>>>>
                # Create a copy
                for i in range(num):
                    member_part = copy.deepcopy(root)
                    # remove all visual nodes directly
                    for link in member_part.findall('link'):
                        if link.attrib['name']!=links_name[i]:
                            for visual in link.findall('visual'):
                                link.remove(visual)
                    xml_string = xml.dom.minidom.parseString(tostring(member_part))
                    xml_pretty_str = xml_string.toprettyxml()
                    tree = ET.ElementTree(member_part)
                    with open(save_dir + '/syn_p{}.urdf'.format(i), "w") as f:
                        f.write(xml_pretty_str)
        object_joints.append(object_joints_per)
    with open(base_path + '/' + dataset + '/statistics.txt', "a+") as f:
        for obj_num in object_nums:
            f.write('{}\t'.format(obj_num))
        f.write('\n')
    with open(base_path + '/' + dataset + '/statistics.txt', "a+") as f:
        for obj_j in object_joints:
            f.write('{}/{}\t'.format(obj_j['revolute'], obj_j['prismatic']))
        f.write('\n')

