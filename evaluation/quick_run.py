import subprocess
import numpy as np
import os
import sys
import time
import json
import h5py
import pickle

if __name__ == '__main__':
    # items = ['bike', 'eyeglasses', 'oven', 'washing_machine', 'laptop', 'drawer', 'cabinet']
    items = ['drawer']
    process_all = []
    for item in items:
        print('python', 'baseline_gn.py', '--item='+item, '--nocs=global', '--domain=unseen')
        p1 = subprocess.Popen(['python', 'baseline_gn.py', '--item='+item, '--nocs=global', '--domain=unseen'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process_all.append(p1)

    for process in process_all:
        stdout, stderr = process.communicate()
        print(stdout)
        print(stderr)
