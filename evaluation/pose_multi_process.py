import os
import time
import pickle
import argparse
import numpy as np
import multiprocessing
from multiprocessing import Process

import _init_paths
from global_info import global_info
from parallel_ancsh_pose import solver_ransac_nonlinear
from lib.data_utils import get_test_group

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', default='unseen', help='which sub test set to choose')
    parser.add_argument('--nocs', default='ANCSH', help='which sub test set to choose')
    parser.add_argument('--item', default='oven', help='object category for benchmarking')
    args = parser.parse_args()

    infos           = global_info()
    dset_info       = infos.datasets[args.item]
    num_parts       = dset_info.num_parts
    num_ins         = dset_info.num_object
    unseen_instances= dset_info.test_list
    special_ins     = dset_info.spec_list
    main_exp        = dset_info.exp
    baseline_exp    = dset_info.baseline
    test_exp        = main_exp
    my_dir          = infos.base_path
    base_path       = my_dir + '/results/test_pred'
    choose_threshold = 0.1

    # testing
    test_h5_path    = base_path + '/{}'.format(test_exp)
    all_test_h5     = os.listdir(test_h5_path)
    test_group      = get_test_group(all_test_h5, unseen_instances, domain=args.domain, spec_instances=special_ins)

    problem_ins     = []
    print('we have {} testing data for {} {}'.format(len(test_group), args.domain, args.item))

    start_time = time.time()
    rts_all = pickle.load( open(my_dir + '/results/pickle/{}/{}_{}_{}_rt.pkl'.format(main_exp, args.domain, args.nocs, args.item), 'rb' ))

    directory = my_dir + '/results/pickle/{}'.format(main_exp)
    file_name = directory + '/{}_{}_{}_{}_rt_ours_{}.pkl'.format(baseline_exp, args.domain, args.nocs, args.item, choose_threshold)

    # s_ind = 0
    # e_ind = 10
    # solver_ransac_nonlinear(s_ind, e_ind, test_exp, baseline_exp, choose_threshold, num_parts, test_group, problem_ins, rts_all, file_name)

    starttime = time.time()
    processes = []
    cpuCount    = os.cpu_count() - 2
    num_per_cpu = int(len(test_group)/cpuCount) + 1
    directory = my_dir + '/results/pickle/{}/subs'.format(main_exp)
    if not os.path.exists(directory):
        os.makedirs(directory)
    for k in range(cpuCount):
        sub_file_name = directory + '/{}_{}_{}_{}_rt_ours_{}_{}.pkl'.format(baseline_exp, args.domain, args.nocs, args.item, choose_threshold, k)
        e_ind = min(num_per_cpu*(k+1), len(test_group))
        p=Process(target=solver_ransac_nonlinear, args=(num_per_cpu*k, e_ind, test_exp, baseline_exp, choose_threshold, num_parts, test_group, problem_ins, rts_all, sub_file_name))
        processes.append(p)
        p.start()
    
    for process in processes:
        process.join()
    print('Process {} took {} seconds, with average {} seconds per data'.format(num_per_cpu*cpuCount, time.time() - starttime, (time.time() - starttime)/(num_per_cpu*cpuCount) ))
