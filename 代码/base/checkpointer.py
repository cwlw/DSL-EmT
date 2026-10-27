from base.utils import load_pickle, save_to_pickle

import os
import time

import numpy as np
import pandas as pd


class GenericCheckpointer(object):
    r"""
    Save the trainer and parameter controller at runtime, and load them
        in another run if resume = 1.
    """
    def __init__(self, path, trainer, parameter_controller, resume, fusion_module=None, feature_mode=None):
        self.checkpoint = {}
        self.path = path
        self.trainer = trainer
        self.parameter_controller = parameter_controller
        self.resume = resume
        # [MOD] Optional external gate state. None keeps all non-fusion checkpoints backward compatible.
        self.fusion_module = fusion_module
        self.feature_mode = feature_mode

    def _restore_fusion_state_and_optimizer(self):
        if self.fusion_module is None:
            return

        saved_feature_mode = self.checkpoint.get('feature_mode')
        if saved_feature_mode is not None and self.feature_mode is not None and saved_feature_mode != self.feature_mode:
            raise ValueError(
                f"Checkpoint feature_mode={saved_feature_mode!r} does not match current mode={self.feature_mode!r}."
            )

        fusion_state = self.checkpoint.get('fusion_state_dict')
        if fusion_state is not None:
            # [MOD] Restore the explicit alpha state in addition to the Trainer pickle.
            self.fusion_module.load_state_dict(fusion_state)

        self.trainer.feature_fusion = self.fusion_module
        optimizer = getattr(self.trainer, 'optimizer', None)
        if optimizer is None:
            return

        fusion_group = None
        for group in optimizer.param_groups:
            if group.get('_is_fusion_group', False):
                fusion_group = group
                break

        new_params = list(self.fusion_module.parameters())
        if fusion_group is None:
            # Backward compatibility for older fusion checkpoints without a marked group.
            optimizer.add_param_group({'params': new_params, '_is_fusion_group': True})
            return

        old_params = list(fusion_group['params'])
        old_states = [optimizer.state.pop(param, {}) for param in old_params]
        fusion_group['params'] = new_params

        # [MOD] Rebind optimizer state to the current Experiment-owned alpha Parameter,
        # preserving Adam moments when they exist.
        for old_state, new_param in zip(old_states, new_params):
            optimizer.state[new_param] = old_state

    def load_checkpoint(self):
        # If checkpoint file exists, then read it.
        if os.path.isfile(self.path):
            print("Loading checkpoint. Are you sure it is intended?")
            self.checkpoint = {**self.checkpoint, **load_pickle(self.path)}
            print("Checkpoint loaded!")

            self.trainer = self.checkpoint['trainer']
            self.trainer.resume = True
            self.parameter_controller = self.checkpoint['param_control']
            self.parameter_controller.trainer = self.trainer
            self._restore_fusion_state_and_optimizer()
        else:
            raise ValueError("Checkpoint not exists!!")
        return self.trainer, self.parameter_controller

    def save_checkpoint(self, trainer, parameter_controller, path):
        self.checkpoint['trainer'] = trainer
        self.checkpoint['param_control'] = parameter_controller
        # [MOD] Persist feature mode and external gate state without altering the original
        # Trainer/LOSO checkpoint payload semantics.
        self.checkpoint['feature_mode'] = self.feature_mode
        if self.fusion_module is not None:
            self.checkpoint['fusion_state_dict'] = self.fusion_module.state_dict()

        print("Saving checkpoint.")
        path = os.path.join(path, "checkpoint.pkl")
        save_to_pickle(path, self.checkpoint, replace=True)
        print("Checkpoint saved.")


class Checkpointer(GenericCheckpointer):
    def __init__(self, path, trainer, parameter_controller, resume, fusion_module=None, feature_mode=None):
        super().__init__(path, trainer, parameter_controller, resume, fusion_module=fusion_module, feature_mode=feature_mode)
        self.columns = []
        self.feature_mode_log = feature_mode

    def _fusion_alpha_value(self):
        if self.fusion_module is None:
            return "NA"
        alpha = self.fusion_module.alpha.detach().cpu().numpy()
        return float(np.mean(alpha))

    def save_log_to_csv(self, epoch=None, mean_train_record=None, mean_validate_record=None, test_record=None):
        fusion_alpha = self._fusion_alpha_value()

        if epoch is not None:
            num_layers_to_update = len(self.trainer.optimizer.param_groups[0]['params'])
            csv_records = [time.time(), epoch, int(self.trainer.best_epoch_info['epoch']), num_layers_to_update,
                           self.trainer.optimizer.param_groups[0]['lr'], self.trainer.train_losses[-1], self.trainer.validate_losses[-1],
                           mean_train_record['rmse'], mean_train_record['pcc'][0],
                           mean_train_record['pcc'][1], mean_train_record['ccc'],
                           mean_validate_record['rmse'], mean_validate_record['pcc'][0],
                           mean_validate_record['pcc'][1], mean_validate_record['ccc'],
                           self.feature_mode_log, fusion_alpha]
        else:
            csv_records = ["Test results:", "rmse: ", test_record['rmse'],
                           "pcc: ", test_record['pcc'][0], test_record['pcc'][1],
                           "ccc: ", test_record['ccc'],
                           "feature_mode: ", self.feature_mode_log,
                           "fusion_alpha: ", fusion_alpha]

        row_df = pd.DataFrame(data=csv_records)
        row_df.T.to_csv(self.trainer.csv_filename, mode='a', index=False, header=False)

    def init_csv_logger(self, args, config):
        self.trainer.csv_filename = os.path.join(self.trainer.save_path, "training_logs.csv")

        # Record the arguments.
        arguments_dict = vars(args)
        arguments_dict = pd.json_normalize(arguments_dict, sep='_')

        df_args = pd.DataFrame(data=arguments_dict)
        df_args.to_csv(self.trainer.csv_filename, index=False)

        config = pd.json_normalize(config, sep='_')
        df_config = pd.DataFrame(data=config)
        df_config.to_csv(self.trainer.csv_filename, mode='a', index=False)

        self.columns = ['time', 'epoch', 'best_epoch', 'layer_to_update', 'lr',
                        'tr_loss', 'val_loss', 'tr_rmse', 'tr_pcc_v', 'tr_pcc_conf', 'tr_ccc',
                        'val_rmse', 'val_pcc_v', 'val_pcc_conf', 'val_ccc',
                        # [MOD] Explicit ablation identifiers in every epoch row.
                        'feature_mode', 'fusion_alpha']
        df = pd.DataFrame(columns=self.columns)
        df.to_csv(self.trainer.csv_filename, mode='a', index=False)


class ClassificationCheckpointer(GenericCheckpointer):
    r"""
    Write training logs into csv files.
    """
    def __init__(self, path, trainer, parameter_controller, resume):
        super().__init__(path, trainer, parameter_controller, resume)
        self.columns = []

    def save_log_to_csv(self, epoch=None):
        np.set_printoptions(suppress=True)
        num_layers_to_update = len(self.trainer.optimizer.param_groups[0]['params'])

        if epoch is None:
            csv_records = ["Test results: ", "accuracy: ", self.trainer.test_accuracy, "kappa: ", self.trainer.test_kappa, "conf_mat: ", self.trainer.test_confusion_matrix]
        else:
            csv_records = [time.time(), epoch, int(self.trainer.best_epoch_info['epoch']), num_layers_to_update,
                           self.trainer.optimizer.param_groups[0]['lr'], self.trainer.train_losses[-1],
                           self.trainer.validate_losses[-1], self.trainer.train_accuracies[-1], self.trainer.validate_accuracies[-1],
                           self.trainer.train_kappas[-1], self.trainer.validate_kappas[-1],
                           self.trainer.train_confusion_matrices[-1], self.trainer.validate_confusion_matrices[-1]]

        row_df = pd.DataFrame(data=csv_records)
        row_df.T.to_csv(self.trainer.csv_filename, mode='a', index=False, header=False)
        np.set_printoptions()

    def init_csv_logger(self, args, config):
        self.trainer.csv_filename = os.path.join(self.trainer.save_path, "training_logs.csv")

        # Record the arguments.
        arguments_dict = vars(args)
        self.print_dict(arguments_dict)
        self.print_dict(config)

        self.columns = ['time', 'epoch', 'best_epoch', 'layer_to_update', 'lr',
                        'tr_loss', 'val_loss', 'tr_acc', 'val_acc', 'tr_kappa', 'val_kappa', 'tr_conf_mat', 'val_conf_mat']

        df = pd.DataFrame(columns=self.columns)
        df.to_csv(self.trainer.csv_filename, mode='a', index=False)

    def print_dict(self, data_dict):
        for key, value in data_dict.items():
            csv_records = [str(key) + " = " + str(value)]
            row_df = pd.DataFrame(data=csv_records)
            row_df.T.to_csv(self.trainer.csv_filename, mode='a', index=False, header=False, sep=' ')

