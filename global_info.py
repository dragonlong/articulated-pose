import collections
import os
import sys
import _init_paths
import platform

DatasetInfo = collections.namedtuple(
    'DatasetInfo',
    ['dataset_name', 'num_object', 'parts_map', 'num_parts', 'train_size', 'test_size', 'train_list', 'test_list', 'spec_list', 'spec_map', 'exp', 'baseline', 'joint_baseline',  'style']
)

TaskData= collections.namedtuple('TaskData', ['query', 'target'])

_DATASETS = dict(
    eyeglasses=DatasetInfo(
        dataset_name='shape2motion',
        num_object=24,
        parts_map=[[0], [1], [2]],
        num_parts=3,
        train_size=13000,
        test_size=3480,
        train_list=None,
        test_list=['0007', '0016', '0036'],
        spec_list=['0006'],
        spec_map=None,
        exp='3.9',
        baseline='3.91',
        joint_baseline='5.0',
        style='new'
       ),

    oven=DatasetInfo(
        dataset_name='shape2motion',
        num_object=42,
        parts_map=[[0], [1]],
        num_parts=2,
        train_size=25000,
        test_size=5480,
        train_list=None,
        test_list=['0003', '0016', '0029'], # for dataset.py
        spec_list=['0006', '0015', '0035', '0038'], # for dataset.py
        spec_map=None,
        exp='3.0',
        baseline='3.01',
        joint_baseline='5.2',
        style='old'
       ),

    laptop=DatasetInfo(
        dataset_name='shape2motion',
        num_object=86,
        parts_map=[[0], [1]],
        num_parts=2,
        train_size=67603,
        test_size=5036,
        train_list=None,
        test_list=['0004', '0008', '0069'],
        spec_list=['0003', '0006', '0041', '0080', '0081'],
        spec_map=None,
        exp='3.6',
        baseline='3.61',
        joint_baseline='5.1',
        style='new'
       ),

    washing_machine=DatasetInfo(
        dataset_name='shape2motion',
        num_object=62,
        parts_map=[[0], [1]],
        num_parts=2,
        train_size=43000,
        test_size=3480,
        train_list=None,
        test_list=['0003', '0029'], # for dataset.py
        spec_list=['0001', '0002', '0006', '0007', '0010',
                   '0027', '0031', '0040', '0050', '0009',
                   '0029', '0038', '0039', '0041', '0046',
                   '0052', '0058'], # for dataset.py
       spec_map=None,
       exp='3.1',
       baseline='3.11',
       joint_baseline='5.3',
       style='old'
       ),

    Laptop=DatasetInfo(
        dataset_name='BMVC15',
        num_object=1,
        parts_map=[[0], [1]],
        num_parts=2,
        train_size=13000,
        test_size=3480,
        train_list=['0001'],
        test_list=['0006'],
        spec_list=[],
        spec_map=None,
        exp=None,
        baseline=None,
        joint_baseline=None,
        style=None
       ),

    Cabinet=DatasetInfo(
        dataset_name='BMVC15',
        num_object=1,
        parts_map=[[0], [1], [2]], # (001)base + (002)drawer + (000)door
        num_parts=3,
        train_size=13000,
        test_size=3480,
        train_list=['0001'],
        test_list=['0006'],
        spec_list=[],
        spec_map={'0001': [1, 2, 0], '0006':[1, 2, 0]},
        exp=None,
        baseline=None,
        joint_baseline=None,
        style=None
       ),

    Cupboard=DatasetInfo(
        dataset_name='BMVC15',
        num_object=1,
        parts_map=[[0], [1]], # base(000) + drawer(001)
        num_parts=2,
        train_size=13000,
        test_size=3480,
        train_list=['0001'],
        test_list=['0006'],
        spec_list=[],
        spec_map={'0001':[0, 1], '0006':[0, 1]},
        exp=None,
        baseline=None,
        joint_baseline=None,
        style=None
       ),

    Train=DatasetInfo(
        dataset_name='BMVC15',
        num_object=1,
        parts_map=[[0], [1], [2], [3]],
        num_parts=4,
        train_size=13000,
        test_size=3480,
        train_list=['0001'],
        test_list=['0006'],
        spec_list=[],
        spec_map={'0001':[0, 1, 2, 3], '0006':[0, 1, 2, 3]},
        exp=None,
        baseline=None,
        joint_baseline=None,
        style=None
       ),

    drawer=DatasetInfo(
        dataset_name='sapien',
        num_object=1,
        parts_map=[[0], [1], [2], [3]],
        num_parts=4,
        train_size=13000,
        test_size=3480,
        train_list=['40453', '44962', '45132',
                    '45290', '46130', '46334',  '46462',
                    '46537', '46544', '46641', '47178', '47183',
                    '47296', '47233', '48010', '48253',  '48517',
                    '48740', '48876', '46230', '44853', '45135',
                    '45427', '45756', '46653', '46879', '47438', '47711', '48491'],
        test_list=[ '46123',  '45841', '46440'],
        spec_list=[],
        spec_map={  '40453':[3, 0, 1, 2], '44962':[3, 0, 1, 2], '45132':[3, 0, 1, 2], '45290':[3, 0, 1, 2], '46123':[3, 0, 1, 2],
                    '46130':[3, 0, 1, 2], '46334':[3, 0, 1, 2], '46440':[3, 0, 1, 2], '46462':[3, 0, 1, 2], '46537':[3, 0, 1, 2],
                    '46544':[3, 0, 1, 2], '46641':[3, 0, 1, 2], '47178':[3, 0, 1, 2], '47183':[3, 0, 1, 2], '47296':[3, 0, 1, 2],
                    '47233':[3, 0, 1, 2], '48010':[3, 0, 1, 2], '48253':[3, 0, 1, 2], '48517':[3, 0, 1, 2], '48740':[3, 0, 1, 2],
                    '48876':[3, 0, 1, 2], '46230':[3, 0, 1, 2],
                    '44853':[3, 1, 2, 0], '45135':[3, 1, 0, 2], '45427':[3, 2, 0, 1], '45756':[3, 1, 2, 0], '45841':[0, 1, 2, 3],
                    '46653':[0, 1, 2, 3], '46879':[3, 1, 2, 0], '47438':[3, 2, 1, 0], '47711':[0, 1, 2, 3], '48491':[0, 1, 2, 3]},
        exp='3.3',
        baseline='3.31',
        joint_baseline='5.4',
        style='new'
       ),
)

class global_info(object):
    def __init__(self):
        self.name      = 'art6d'
        self.datasets  = _DATASETS
        self.model_type= 'pointnet++'
        # primary path, with sub-folders:
        # model/: put all training profiles & checkpoints;
        # results/: put network raw predictions + pose estimation results + error evaluation results;
        # dataset/: all data we use
        self.base_path = '/work/cascades/lxiaol9/6DPOSE'
        self.group_path= './' # useful when we have additional dataset;

if __name__ == '__main__':
    infos = global_info()
    print(infos.datasets['eyeglasses'].dataset_name)
