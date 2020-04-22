import yaml
import platform
import numpy as np
import _init_paths
from global_info import global_info

infos = global_info()
class NetworkConfig(object):
    def __init__(self, args):
        self.conf      = yaml.load(open(args.config_file, 'r'))
        self.num_exp = str(args.num_expr)
        self.nocs_type = args.nocs_type
        self.pred_joint      = args.pred_joint
        self.pred_joint_ind  = args.pred_joint_ind
        self.early_split     = args.early_split
        self.early_split_nocs= args.early_split_nocs # control early split for part & global NOCS estimation

    def fetch(self, name, default_value=None):
        result = self.conf.get(name, default_value)
        assert result is not None
        return result

    def get_in_model_dir(self):
        return infos.base_path + '/' + self.fetch('in_model_dir') + '/' + self.num_exp

    def get_pretrain_model_dir(self):
        return infos.base_path + '/' + self.fetch('in_model_dir') + '/2.001'

    def get_out_model_dir(self):
        return infos.base_path + '/' +self.fetch('out_model_dir')  + '/' + self.num_exp

    def get_log_dir(self):
        return infos.base_path + '/' +self.fetch('log_dir')  + '/' + self.num_exp

    def get_val_prediction_dir(self):
        return infos.base_path + '/' + self.fetch('val_prediction_dir') + '/' + self.num_exp

    def get_test_prediction_dir(self):
        return infos.base_path + '/' + self.fetch('test_prediction_dir') + '/' + self.num_exp

    def get_demo_prediction_dir(self):
        return infos.base_path + '/' + self.fetch('demo_prediction_dir') + '/' + self.num_exp

    def get_nn_name(self):
        return self.fetch('nn_name')

    def get_batch_size(self):
        return self.fetch('batch_size')

    def get_nocs_type(self):
        return self.nocs_type

    def get_parametri_type(self):
        return self.fetch('parametri_type')

    def get_pred_joint(self):
        return self.pred_joint

    def get_total_loss_multiplier(self):
        return self.fetch('total_loss_multiplier')

    def get_nocs_loss_multiplier(self):
        return self.fetch('nocs_loss_multiplier')

    def get_gocs_loss_multiplier(self):
        return self.fetch('gocs_loss_multiplier')

    def get_offset_loss_multiplier(self):
        return self.fetch('offset_loss_multiplier')

    def get_orient_loss_multiplier(self):
        return self.fetch('orient_loss_multiplier')

    def get_index_loss_multiplier(self):
        return self.fetch('index_loss_multiplier')

    def get_direct_loss_multiplier(self):
        return self.fetch('direct_loss_multiplier')

    def get_type_loss_multiplier(self):
        return self.fetch('type_loss_multiplier')

    def get_residue_loss_multiplier(self):
        return self.fetch('residue_loss_multiplier')

    def get_parameter_loss_multiplier(self):
        return self.fetch('parameter_loss_multiplier')

    def get_miou_loss_multiplier(self):
        return self.fetch('miou_loss_multiplier')

    def get_bn_decay_step(self):
        return self.fetch('bn_decay_step', -1)

    def get_init_learning_rate(self):
        return self.fetch('init_learning_rate')

    def get_decay_step(self):
        return self.fetch('decay_step')

    def get_decay_rate(self):
        return self.fetch('decay_rate')

    def get_n_epochs(self):
        return self.fetch('n_epochs')

    def get_val_interval(self):
        return self.fetch('val_interval', 5)

    def get_snapshot_interval(self):
        return self.fetch('snapshot_interval', 100)

    def get_train_data_file(self):
        return self.fetch('train_data_file')

    def get_train_data_first_n(self):
        return self.fetch('train_first_n')

    def is_train_data_add_noise(self):
        return self.fetch('train_data_add_noise')

    def get_val_data_file(self):
        return self.fetch('val_data_file')

    def get_val_data_first_n(self):
        return self.fetch('val_first_n')

    def is_val_data_add_noise(self):
        return self.fetch('val_data_add_noise')

    def get_val_prediction_n_keep(self):
        return self.fetch('val_prediction_n_keep')

    def get_test_data_file(self):
        return self.fetch('test_data_file')

    def get_test_data_first_n(self):
        return self.fetch('test_first_n')

    def is_test_data_add_noise(self):
        return self.fetch('test_data_add_noise')

    def get_CUDA_visible_GPUs(self):
        return self.fetch('CUDA_visible_GPUs')

    def get_writer_start_step(self):
        return self.fetch('writer_start_step')

    def is_debug_mode(self):
        return self.fetch('debug_mode')

    def get_n_max_parts(self):
        return self.fetch('n_max_parts')

    def get_list_of_primitives(self):
        return self.fetch('list_of_primitives')

    def use_direct_regression(self):
        return self.fetch('use_direct_regression')

    def get_nocs_loss(self):
        return self.fetch('coord_regress_loss')
