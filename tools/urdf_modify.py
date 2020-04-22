"""
func: python code to generate urdf format per parts:
- traverse over path and objects
- parse from json file(dict, list, array);
- obj files path;
- inverse computing on rpy/xyz;
- xml writer to urdf
"""
import platform
import os
import os.path
import glob
import sys
import time
import random as rdn

import h5py
import yaml
import json
import copy
import collections

import numpy as np
import math

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, tostring, SubElement, Comment, ElementTree, XML
import xml.dom.minidom

def modify_urdf(urdf_dir):
    """
    urdf cleaning
    """
    urdf_file  =  urdf_dir + '/mobility.urdf'
    save_dir   =  urdf_dir
    xml_string = xml.dom.minidom.parse(urdf_file)
    xml_pretty_str = xml_string.toprettyxml()

    tree = ET.parse(urdf_file)
    root = tree.getroot()
    links_name = [link.attrib['name'] for link in root.findall('link')]
    print(links_name)
    num = len(links_name)

    #>>>>>>>>>>>>>> joint features
    #
    for i in range(num):
        member_part = copy.deepcopy(root)
        for link in member_part.findall('link'):
            if link.attrib['name']!=links_name[i]:
                for visual in link.findall('visual'):
                    link.remove(visual)
                for collision in link.findall('collision'):
                    link.remove(collision)
                # for geometry in link.iter('geometry'):
                #     print(geometry[0].attrib['filename'])
                #     for mesh in geometry.iter('mesh'):
                #         son_package = mesh.attrib['filename'].split('//')[1].split('/')[0]
                #         print(mesh.attrib['filename'].split('//')[1].split('/')[0])
                #         for urdf_path in n_package:
                #             if son_package in os.listdir(urdf_path):
                #                 mesh.set('filename',  urdf_path + mesh.attrib['filename'].split('//')[1])
            else:
                # for geometry in link.iter('geometry'):
                #     if 'filename' in geometry[0].attrib:
                #         print(geometry[0].attrib['filename'])
                #         for mesh in geometry.iter('mesh'):
                #             son_package = mesh.attrib['filename'].split('//')[1].split('/')[0]
                #             # print(mesh.attrib['filename'].split('//')[1].split('/')[0])
                #             for urdf_path in n_package:
                #                 if son_package in os.listdir(urdf_path):
                #                     mesh.set('filename',  urdf_path + mesh.attrib['filename'].split('//')[1])
                for collision in link.findall('collision'):
                    link.remove(collision)
            if link.findall('inertial') == []:
                inertial = SubElement(link, 'inertial')
                node_inertial = XML('''<inertial><origin rpy="0 0 0" xyz="0 0 0"/><mass value="3.0"/><inertia ixx="0.9" ixy="0.9" ixz="0.9" iyy="0.9" iyz="0" izz="0.9"/></inertial>''')
                inertial.extend(node_inertial)
                if link.attrib['name']=='base':
                    for mass in inertial.iter('mass'):
                        mass.set('value', "0.0")
                    for inertia in inertial.iter('inertia'):
                        inertia.set('ixx', "0.0")
                        inertia.set('ixy', "0.0")
                        inertia.set('ixz', "0.0")
                        inertia.set('iyy', "0.0")
                        inertia.set('iyz', "0.0")
                        inertia.set('izz', "0.0")

        xml_string     = xml.dom.minidom.parseString(tostring(member_part))
        xml_pretty_str = xml_string.toprettyxml()
        tree = ET.ElementTree(member_part)
        with open(save_dir + '/syn_p{}.urdf'.format(i), "w") as f:
            f.write(xml_pretty_str)

if __name__ == '__main__':
    base_path = '/work/cascades/lxiaol9/6DPOSE/BMVC15/urdf/Train'
    urdf_dirs  = glob.glob(base_path + '/*/') #
    for urdf_dir in urdf_dirs:
        print('modifying ', urdf_dir)
        modify_urdf(urdf_dir)
