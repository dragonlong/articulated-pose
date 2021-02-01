import os, sys
import argparse
BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, 'lib'))
sys.path.append(os.path.join(BASE_DIR, '..'))

#>>>>>>>>>>>>>>>>>> custom packages
from network_config import NetworkConfig
from network import Network
from dataset import Dataset
from global_info import global_info

#>>>>>>>>>>>>>>>>>> python lib
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

if __name__ == '__main__':
    infos       = global_info()
    tf.set_random_seed(1234)
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='./cfg/network_config.yml', help='YAML configuration file')
    parser.add_argument('--name_data', default='shape2motion', help='name of the dataset we use')
    parser.add_argument('--item', default='eyeglasses', help='name of the dataset we use')
    parser.add_argument('--num_expr', default=0.01, help='small set data used for testing')
    parser.add_argument('--nocs_type', default='ancsh', help='whether use global or part level NOCS') # default A/B/C
    parser.add_argument('--data_mode', default='test', help='how to split and choose data')

    # control model architecture
    parser.add_argument('--pred_joint', action='store_true', help='whether we want to predict joint offsets')
    parser.add_argument('--pred_joint_ind', action='store_true', help='whether we want to predict joint offsets index')
    parser.add_argument('--early_split', action='store_true', help='whether we want to early split for joints prediction')
    parser.add_argument('--early_split_nocs', action='store_true', help='whether we want to split for two nocs heads')

    # control init & loss
    parser.add_argument('--test', action='store_true', help='Run network in test time')
    parser.add_argument('--debug', action='store_true', help='indicating whether in debug mode')
    parser.add_argument('--gpu', default='0', help='help with parallel running')
    args = parser.parse_args()
    data_infos  = infos.datasets[args.item]
    if args.nocs_type == 'ancsh':
        print('print training ANCSH network')
        args.num_expr = data_infos.exp
        nocs_type = 'AC'
        args.pred_joint     = True
        args.pred_joint_ind = True
        args.early_split    = True
        args.early_split_nocs = True
    elif args.nocs_type == 'npcs':
        args.num_expr = data_infos.baseline
        nocs_type = 'A'

    conf = NetworkConfig(args)

    visible_GPUs = args.gpu
    if visible_GPUs is not None:
        print('Setting CUDA_VISIBLE_DEVICES={}'.format(','.join(visible_GPUs)))
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(visible_GPUs)

    is_testing  = True if args.test else False
    batch_size  = conf.get_batch_size()
    n_max_parts = data_infos.num_parts
    if args.name_data == 'mobility-v0-prealpha3':
        root_data = infos.group_path + '/dataset/' + args.name_data
    else:
        root_data = infos.base_path + '/dataset/' + args.name_data

    is_debug    = args.debug

    if is_debug:
        output_msg = 'max_parts:  {}\n root_data: {}\n nocs_type : {}\n'.format(n_max_parts, root_data, nocs_type)
        print(output_msg)

    print('Building network...')
    tf_conf = tf.ConfigProto()
    tf_conf.allow_soft_placement = True
    tf_conf.gpu_options.allow_growth = True

    in_model_dir = conf.get_in_model_dir()
    ckpt = tf.train.get_checkpoint_state(in_model_dir)
    should_restore = (ckpt is not None) and (ckpt.model_checkpoint_path is not None)
    net = Network(n_max_parts=n_max_parts, config=conf, is_new_training=not should_restore)

    with tf.Session(config=tf_conf, graph=net.graph) as sess:
        if conf.is_debug_mode():
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        if should_restore:
            tvars = tf.trainable_variables()
            for var in tvars:
                print(var.name)
            print('\n \n ')
            checkpoint_path = ckpt.model_checkpoint_path
            print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='', all_tensors=False, all_tensor_names=True)
            print('Restoring ' + checkpoint_path + ' ...')
            tf.train.Saver().restore(sess, checkpoint_path)
        else:
            assert not is_testing
            print('Starting a new training...')
            sess.run(tf.global_variables_initializer())

        print('Loading data...')
        if is_debug:
            data_pts =  train_data.fetch_data_at_index(1)
            print(data_pts)

        if is_testing:
            # batch testing
            print('Entering testing mode using test set')
            test_data = Dataset(
            root_dir=root_data,
            ctgy_obj=args.item,
            mode=args.data_mode,
            name_dset=args.name_data,
            batch_size=batch_size,
            n_max_parts=n_max_parts,
            add_noise=conf.is_train_data_add_noise(),
            nocs_type=nocs_type,
            parametri_type=conf.get_parametri_type(),
            fixed_order=True,
            first_n=conf.get_train_data_first_n(),
            is_debug=is_debug)
            if args.data_mode == 'demo':
                save_dir = conf.get_demo_prediction_dir()
            else:
                save_dir = conf.get_test_prediction_dir()
            net.predict_and_save(
                sess,
                dset=test_data,
                save_dir=save_dir,
            )
        else:
            print('Entering training mode!!!')
            train_data = Dataset(
                root_dir=root_data,
                ctgy_obj=args.item,
                mode='train',
                name_dset=args.name_data,
                batch_size=batch_size,
                n_max_parts=n_max_parts,
                add_noise=conf.is_train_data_add_noise(),
                nocs_type=nocs_type,
                parametri_type=conf.get_parametri_type(),
                fixed_order=False,
                first_n=conf.get_train_data_first_n(),
                is_debug=is_debug)

            # seen instances
            val1_data = Dataset(
                root_dir=root_data,
                ctgy_obj=args.item,
                mode='test',
                name_dset=args.name_data,
                batch_size=batch_size,
                n_max_parts=n_max_parts,
                add_noise=conf.is_val_data_add_noise(),
                nocs_type=nocs_type,
                domain='seen',
                parametri_type=conf.get_parametri_type(),
                fixed_order=True,
                first_n=conf.get_val_data_first_n(),
                is_debug=is_debug)

            # unseen instances
            val2_data = Dataset(
                root_dir=root_data,
                ctgy_obj=args.item,
                mode='test',
                name_dset=args.name_data,
                batch_size=batch_size,
                n_max_parts=n_max_parts,
                add_noise=conf.is_val_data_add_noise(),
                nocs_type=nocs_type,
                domain='unseen',
                parametri_type=conf.get_parametri_type(),
                fixed_order=True,
                first_n=conf.get_val_data_first_n(),
                is_debug=is_debug)

            net.train(
                sess,
                train_data=train_data,
                vals_data=[val1_data, val2_data],
                n_epochs=conf.get_n_epochs(),
                val_interval=conf.get_val_interval(),
                snapshot_interval=conf.get_snapshot_interval(),
                model_dir=conf.get_out_model_dir(),
                log_dir=conf.get_log_dir(),
            )
