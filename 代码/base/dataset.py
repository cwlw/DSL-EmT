import torch
import os
from operator import itemgetter
import numpy as np
import random
from torchvision.transforms import transforms
import torchvision
from torch.utils.data import Dataset
from base.utils import roll_list


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)

    def __call__(self, Imgs):
        L, C, H, W = Imgs.shape
        tensor = []
        for k in range(L):
            img = self.normalize(Imgs[k, :, :, :])
            tensor.append(img)
        return torch.stack(tensor, dim=0)


class _EEGFeatureMixin(object):
    EEG_SHAPE = (32, 6)

    def _init_eeg_frontend(self, feature_mode, feature_stats, fusion_module):
        self.feature_mode = feature_mode
        self.feature_stats = feature_stats
        self.fusion_module = fusion_module  # 保留但不再使用

        if self.feature_mode not in ("rPSD", "DE", "fusion"):
            raise ValueError(
                f"Unsupported feature_mode={self.feature_mode!r}. "
                "Expected 'rPSD', 'DE', or 'fusion'."
            )

        # ===== 【删除】强制检查 fusion_module 非空 =====
        # 融合已在模型内部完成，Dataset 只需返回 rPSD 和 DE

        if not isinstance(self.feature_stats, dict):
            raise RuntimeError(
                "Global EEG feature statistics are missing from dataset_info.pkl. "
                "Regenerate the dataset metadata before training."
            )

    @staticmethod
    def _as_float32_tensor(array):
        return torch.from_numpy(np.asarray(array, dtype=np.float32))

    def _normalize_rpsd(self, data):
        stats = self.feature_stats.get("eeg_RP")
        if stats is None:
            raise RuntimeError("Missing global eeg_RP min/max in dataset_info.pkl.")
        minimum = float(stats["min"])
        maximum = float(stats["max"])
        denominator = maximum - minimum
        if denominator <= 0.0:
            raise RuntimeError("Invalid global rPSD MinMax statistics: max must be > min.")
        return (np.asarray(data, dtype=np.float32) - minimum) / denominator

    def _normalize_de(self, data):
        stats = self.feature_stats.get("eeg_DE")
        if stats is None:
            raise RuntimeError("Missing global eeg_DE mean/std in dataset_info.pkl.")
        mean = float(stats["mean"])
        std = float(stats["std"])
        if std <= 0.0:
            raise RuntimeError("Invalid global DE Z-Score statistics: std must be > 0.")
        return (np.asarray(data, dtype=np.float32) - mean) / std

    def _reshape_eeg(self, data):
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 2 or data.shape[-1] != int(np.prod(self.EEG_SHAPE)):
            raise ValueError(
                "Expected offline EEG feature shape [T, 192] before reshape, "
                f"got {tuple(data.shape)}."
            )
        return data.reshape(data.shape[0], *self.EEG_SHAPE)

    def _load_eeg_source(self, path, indices, feature_name, random_index_override=None):
        filename = os.path.join(path, feature_name + ".npy")
        if not os.path.isfile(filename):
            raise FileNotFoundError(
                f"Required EEG feature file not found: {filename}. "
                "Generate eeg_RP.npy and eeg_DE.npy first."
            )
        data = np.load(filename, mmap_mode='c')
        if random_index_override is None:
            selected = data[indices]
        else:
            selected = data[random_index_override]

        if feature_name == "eeg_RP":
            selected = self._normalize_rpsd(selected)
        elif feature_name == "eeg_DE":
            selected = self._normalize_de(selected)
        else:
            raise ValueError(f"Unsupported EEG source feature: {feature_name}")

        return self._reshape_eeg(selected)

    def _sample_eeg_pair(self, path, length, index):
        if length < self.window_length:
            multiplier = self.multiplier.get("eeg_RP", 1)
            x = random.randint(0, multiplier - 1)
            random_index = index * multiplier + x

            rpsd_window = self._load_eeg_source(path, random_index, "eeg_RP")
            de_window = self._load_eeg_source(path, random_index, "eeg_DE")

            padded_rpsd = np.zeros((self.window_length, *self.EEG_SHAPE), dtype=np.float32)
            padded_de = np.zeros((self.window_length, *self.EEG_SHAPE), dtype=np.float32)
            padded_rpsd[index] = rpsd_window
            padded_de[index] = de_window
            return padded_rpsd, padded_de

        x = random.randint(0, self.multiplier.get("eeg_RP", 1) - 1)
        random_index = index * self.multiplier.get("eeg_RP", 1) + x

        rpsd_window = self._load_eeg_source(path, length, "eeg_RP", random_index_override=random_index)
        de_window = self._load_eeg_source(path, length, "eeg_DE", random_index_override=random_index)

        return rpsd_window, de_window

    def _single_eeg_example(self, path, length, index, source_feature):
        multiplier = self.multiplier.get(source_feature, self.multiplier.get("eeg_bandpower", 1))
        x = random.randint(0, multiplier - 1)
        random_index = index * multiplier + x

        if length < self.window_length:
            example = np.zeros((self.window_length, *self.EEG_SHAPE), dtype=np.float32)
            selected = self._load_eeg_source(path, index, source_feature)
            example[index] = selected
            return self._as_float32_tensor(example)

        selected = self._load_eeg_source(path, length, source_feature, random_index_override=random_index)
        return self._as_float32_tensor(selected)

    def _build_eeg_example(self, path, length, index):
        if self.feature_mode == "rPSD":
            return self._single_eeg_example(path, length, index, "eeg_RP")
        if self.feature_mode == "DE":
            return self._single_eeg_example(path, length, index, "eeg_DE")

        # ===== fusion 模式：返回两个独立张量 =====
        rpsd, de = self._sample_eeg_pair(path, length, index)
        return self._as_float32_tensor(rpsd), self._as_float32_tensor(de)


class MyDataset(_EEGFeatureMixin, Dataset):
    def __init__(self, data_list, continuous_label_dim, modality, multiplier,
                 feature_dimension, window_length, mode,
                 mean_std=None, time_delay=0, feature_extraction=0,
                 feature_mode="rPSD", feature_stats=None, fusion_module=None):
        self.data_list = data_list
        self.continuous_label_dim = continuous_label_dim
        self.mean_std = mean_std
        self.mean_std_info = 0
        self.time_delay = time_delay
        self.modality = modality
        self.multiplier = multiplier
        self.feature_dimension = feature_dimension
        self.feature_extraction = feature_extraction
        self.window_length = window_length
        self.mode = mode
        self.transform_dict = {}

        self._init_eeg_frontend(feature_mode, feature_stats, fusion_module)
        self.get_3D_transforms()

    def get_feature_transform(self, feature):
        if "video" in feature:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[self.mean_std[feature]['mean']],
                                     std=[self.mean_std[feature]['std']])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        return transform

    def get_3D_transforms(self):
        for feature in self.modality:
            if "continuous_label" not in feature and "video" not in feature and feature != "eeg_bandpower":
                self.transform_dict[feature] = self.get_feature_transform(feature)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        path, trial, length, sample_index = self.data_list[index]
        examples = {}

        eeg_data = self._build_eeg_example(path, length, sample_index)

        if self.feature_mode == "fusion":
            examples["rpsd_feat"] = eeg_data[0]
            examples["de_feat"] = eeg_data[1]
            examples["eeg_bandpower"] = torch.zeros(1)  # 占位
        else:
            examples["eeg_bandpower"] = eeg_data

        for feature in self.modality:
            if feature == "eeg_bandpower":
                continue
            examples[feature] = self.get_example(path, length, sample_index, feature)

        if len(sample_index) < self.window_length:
            sample_index = np.arange(self.window_length)

        return examples, trial, length, sample_index

    def get_example(self, path, length, index, feature):
        x = random.randint(0, self.multiplier[feature] - 1)
        random_index = index * self.multiplier[feature] + x

        if length < self.window_length:
            shape = (self.window_length,) + self.feature_dimension[feature]
            dtype = np.float32
            if feature == "video":
                dtype = np.int8
            example = np.zeros(shape=shape, dtype=dtype)
            example[index] = self.load_data(path, random_index, feature)
        else:
            example = self.load_data(path, random_index, feature)

        if "continuous_label" in feature and self.time_delay != 0:
            example = np.concatenate(
                (example[self.time_delay:, :],
                 np.repeat(example[-1, :][np.newaxis], repeats=self.time_delay, axis=0)), axis=0)

        if "continuous_label" not in feature:
            example = self.transform_dict[feature](np.asarray(example, dtype=np.float32))

        return example

    def load_data(self, path, indices, feature):
        filename = os.path.join(path, feature + ".npy")
        data = np.zeros(((len(indices),) + self.feature_dimension[feature]), dtype=np.float32)
        if os.path.isfile(filename):
            if self.feature_extraction:
                data = np.load(filename, mmap_mode='c')
            else:
                data = np.load(filename, mmap_mode='c')[indices]

        if "continuous_label" in feature:
            data = self.processing_label(data)
        return data

    def processing_label(self, label):
        label = label[:, self.continuous_label_dim]
        if label.ndim == 1:
            label = label[:, None]
        return label


class MyDatasetPreLoad(_EEGFeatureMixin, Dataset):
    def __init__(self, data_list, continuous_label_dim, modality, multiplier,
                 feature_dimension, window_length, mode,
                 mean_std=None, time_delay=0, feature_extraction=0,
                 feature_mode="rPSD", feature_stats=None, fusion_module=None):
        self.data_list = data_list
        self.continuous_label_dim = continuous_label_dim
        self.mean_std = mean_std
        self.mean_std_info = 0
        self.time_delay = time_delay
        self.modality = modality
        self.multiplier = multiplier
        self.feature_dimension = feature_dimension
        self.feature_extraction = feature_extraction
        self.window_length = window_length
        self.mode = mode
        self.transform_dict = {}

        self._init_eeg_frontend(feature_mode, feature_stats, fusion_module)
        self.get_3D_transforms()
        self.examples_list, self.trial_list, self.length_list, self.index_list = self.pre_load()

    def get_feature_transform(self, feature):
        if "video" in feature:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[self.mean_std[feature]['mean']],
                                     std=[self.mean_std[feature]['std']])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        return transform

    def get_3D_transforms(self):
        for feature in self.modality:
            if "continuous_label" not in feature and "video" not in feature and feature != "eeg_bandpower":
                self.transform_dict[feature] = self.get_feature_transform(feature)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        cached = self.examples_list[index]
        examples = {}

        if self.feature_mode == "fusion":
            examples["rpsd_feat"] = cached["__rpsd"]
            examples["de_feat"] = cached["__de"]
            examples["eeg_bandpower"] = torch.zeros(1)
        else:
            examples["eeg_bandpower"] = cached["eeg_bandpower"]

        for feature, value in cached.items():
            if feature.startswith("__") or feature == "eeg_bandpower":
                continue
            examples[feature] = value

        return examples, self.trial_list[index], self.length_list[index], self.index_list[index]

    def pre_load(self):
        examples_list, trial_list, length_list, index_list = [], [], [], []

        for data_idx in range(len(self.data_list)):
            path, trial, length, sample_index = self.data_list[data_idx]
            examples = {}

            if self.feature_mode == "fusion":
                rpsd, de = self._sample_eeg_pair(path, length, sample_index)
                examples["__rpsd"] = self._as_float32_tensor(rpsd)
                examples["__de"] = self._as_float32_tensor(de)
            else:
                examples["eeg_bandpower"] = self._build_eeg_example(path, length, sample_index)

            for feature in self.modality:
                if feature == "eeg_bandpower":
                    continue
                examples[feature] = self.get_example(path, length, sample_index, feature)

            if len(sample_index) < self.window_length:
                sample_index = np.arange(self.window_length)

            examples_list.append(examples)
            trial_list.append(trial)
            length_list.append(length)
            index_list.append(sample_index)

        return examples_list, trial_list, length_list, index_list

    def get_example(self, path, length, index, feature):
        x = random.randint(0, self.multiplier[feature] - 1)
        random_index = index * self.multiplier[feature] + x

        if length < self.window_length:
            shape = (self.window_length,) + self.feature_dimension[feature]
            dtype = np.float32
            if feature == "video":
                dtype = np.int8
            example = np.zeros(shape=shape, dtype=dtype)
            example[index] = self.load_data(path, random_index, feature)
        else:
            example = self.load_data(path, random_index, feature)

        if "continuous_label" in feature and self.time_delay != 0:
            example = np.concatenate(
                (example[self.time_delay:, :],
                 np.repeat(example[-1, :][np.newaxis], repeats=self.time_delay, axis=0)), axis=0)

        if "continuous_label" not in feature:
            example = self.transform_dict[feature](np.asarray(example, dtype=np.float32))

        return example

    def load_data(self, path, indices, feature):
        filename = os.path.join(path, feature + ".npy")
        if os.path.isfile(filename):
            if self.feature_extraction:
                data = np.load(filename)
            else:
                data = np.load(filename)[indices]

            if "continuous_label" in feature:
                data = self.processing_label(data)
            return data

        shape = (len(indices),) + self.feature_dimension[feature]
        return np.zeros(shape, dtype=np.float32)

    def processing_label(self, label):
        label = label[:, self.continuous_label_dim]
        if label.ndim == 1:
            label = label[:, None]
        return label


class DataArranger(object):
    def __init__(self, dataset_info, dataset_path, debug, task, case, seed):
        self.task = task
        self.case = case
        self.seed = seed
        self.dataset_info = dataset_info
        self.debug = debug
        self.trial_list = self.generate_raw_trial_list(dataset_path)
        self.partition_range = self.partition_range_fn()
        self.fold_to_partition = self.assign_fold_to_partition()

    def generate_partitioned_trial_list(self, window_length, hop_length, fold, windowing=True):
        partition_range = list(roll_list(self.partition_range, fold))
        partitioned_trial = {}
        trial_list = self.trial_list
        for partition, num_fold in self.fold_to_partition.items():
            partitioned_trial[partition] = []
            if partition == "validate":
                partitioned_trial['validate'] = validate_trial
            for i in range(num_fold):
                index = partition_range.pop(0)
                trial_of_this_fold = list(itemgetter(*index)(trial_list))
                if len(index) == 1:
                    trial_of_this_fold = [trial_of_this_fold]
                for path, trial, length in trial_of_this_fold:
                    partitioned_trial[partition].append([path, trial, length])
            # Split the total N‑1 folds into training and validation sets (80% to 20%).
            if partition == "train":
                random.Random(self.seed).shuffle(partitioned_trial['train'])
                count = len(partitioned_trial['train'])
                num_train_trials = int(count * 0.8)
                train_idx = np.arange(num_train_trials)
                validate_idx = np.arange(num_train_trials, count)
                train_trial = list(itemgetter(*train_idx)(partitioned_trial["train"]))
                validate_trial = list(itemgetter(*validate_idx)(partitioned_trial["train"]))
                partitioned_trial['train'] = train_trial
        windowed_partitioned_trial = {key: [] for key in partitioned_trial}
        for partition, trials in partitioned_trial.items():
            for path, trial, length in trials:
                if windowing:
                    windowed_indices = self.windowing(np.arange(length), window_length=window_length,
                                                      hop_length=hop_length)
                else:
                    windowed_indices = self.windowing(np.arange(length), window_length=length,
                                                      hop_length=hop_length)
                for index in windowed_indices:
                    windowed_partitioned_trial[partition].append([path, trial, length, index])
        return windowed_partitioned_trial

    def generate_raw_trial_list(self, dataset_path):
        trial_path = os.path.join(dataset_path, self.dataset_info['data_folder'])
        train_list = []
        for trial in self.generate_iterator():
            idx = self.dataset_info['trial'].index(trial)
            trial = self.dataset_info['trial'][idx]
            path = os.path.join(trial_path, trial)
            length = self.dataset_info['length'][idx]
            train_list.append([path, trial, length])
        return train_list

    def generate_iterator(self):
        iterator = []
        for idx, trial in enumerate(self.dataset_info['trial']):
            if self.task == "reg":
                if self.dataset_info['has_continuous_label'][idx]:
                    iterator.append(trial)
            elif self.task == "cls":
                if self.dataset_info['has_eeg'][idx]:
                    iterator.append(trial)
        return iterator

    def partition_range_fn(self):
        if self.task == "reg":
            if self.case == "trs":
                partition_range = [np.arange(a, a+24) for a in np.arange(0, 239, 24)]
                partition_range[-1] = np.insert(partition_range[-1], obj=[0], values=partition_range[-2][-1])
                partition_range[-2] = np.delete(partition_range[-2], [-1])
                partition_range[-1] = np.delete(partition_range[-1], [-1])
            elif self.case == "loso":
                partition_range = [np.arange(0, 19), np.arange(19, 24), np.arange(24, 37),
                                   np.arange(37, 46), np.arange(46, 59), np.arange(59, 68),
                                   np.arange(68, 81), np.arange(81, 94), np.arange(94, 99),
                                   np.arange(99, 105), np.arange(105, 118), np.arange(118, 120),
                                   np.arange(120, 121), np.arange(121, 132), np.arange(132, 149),
                                   np.arange(149, 159), np.arange(159, 173), np.arange(173, 188),
                                   np.arange(188, 200), np.arange(200, 216), np.arange(216, 222),
                                   np.arange(222, 226), np.arange(226, 236), np.arange(236, 239)]
        elif self.task == "cls":
            if self.case == "trs":
                partition_range = [np.arange(a, a+53) for a in np.arange(0, 527, 53)]
                for i in range(3):
                    partition_range[-1] = np.insert(partition_range[-1], obj=[0], values=partition_range[-2][-1])
                    partition_range[-2] = np.delete(partition_range[-2], [-1])
            elif self.case == "loso":
                partition_range = [np.arange(0, 20), np.arange(20, 40), np.arange(40, 57),
                                   np.arange(57, 77), np.arange(77, 97), np.arange(97, 117),
                                   np.arange(117, 137), np.arange(137, 157), np.arange(157, 171),
                                   np.arange(171, 191), np.arange(191, 211), np.arange(211, 231),
                                   np.arange(231, 251), np.arange(251, 267), np.arange(267, 287),
                                   np.arange(287, 307), np.arange(307, 327), np.arange(327, 347),
                                   np.arange(347, 367), np.arange(367, 387), np.arange(387, 407),
                                   np.arange(407, 427), np.arange(427, 447), np.arange(447, 467),
                                   np.arange(467, 487), np.arange(487, 507), np.arange(507, 527)]
        else:
            raise ValueError("Unknown task!")
        if self.debug == 1:
            if self.case == "loso":
                num_folds = 24
            elif self.case == "trs":
                num_folds = 10
            else:
                raise ValueError("Unknown case!")
            partition_range = [np.arange(a, a + 1) for a in range(num_folds)]
        return partition_range

    def assign_fold_to_partition(self):
        if self.task == "reg":
            if self.case == "trs":
                fold_to_partition = {'train': 9, 'validate': 0, 'test': 1}
            elif self.case == "loso":
                fold_to_partition = {'train': 23, 'validate': 0, 'test': 1}
            else:
                raise ValueError("Unknown case!!")
        elif self.task == "cls":
            if self.case == "trs":
                fold_to_partition = {'train': 9, 'validate': 0, 'test': 1}
            elif self.case == "loso":
                fold_to_partition = {'train': 26, 'validate': 0, 'test': 1}
            else:
                raise ValueError("Unknown case!!")
        else:
            raise ValueError("Unknown task!")
        return fold_to_partition

    @staticmethod
    def get_feature_list():
        feature_list = ['eeg_bandpower', 'eeg_raw']
        return feature_list

    @staticmethod
    def windowing(x, window_length, hop_length):
        length = len(x)
        if length >= window_length:
            steps = (length - window_length) // hop_length + 1
            sampled_x = []
            for i in range(steps):
                start = i * hop_length
                end = start + window_length
                sampled_x.append(x[start:end])
            if sampled_x[-1][-1] < length - 1:
                sampled_x.append(x[-window_length:])
        else:
            sampled_x = [x]
        return sampled_x