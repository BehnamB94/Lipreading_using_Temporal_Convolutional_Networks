import os
import glob
import torch
import random
import librosa
import numpy as np
import sys
from lipreading.utils import read_txt_lines

label_convert_dict = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 15,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 12,
    12: 10,
    13: 4,
    14: 11,
    15: 12,
    16: 15,
    17: 15,
    18: 13,
    19: 14,
    21: 15,
}

class MyDataset(object):
    def __init__(self, modality, data_partition, data_dir, label_fp, annonation_direc=None,
        preprocessing_func=None, data_suffix='.npz', merge_classes=False):
        assert os.path.isfile( label_fp ), "File path provided for the labels does not exist. Path iput: {}".format(label_fp)
        self._data_partition = data_partition
        self._data_dir = data_dir
        self._data_suffix = data_suffix

        self._label_fp = label_fp
        self._annonation_direc = annonation_direc

        self.modality = modality
        self.fps = 25 if modality == "video" else 16000
        self.is_var_length = True
        self.label_idx = -3
        self.merge_classes = merge_classes

        self.preprocessing_func = preprocessing_func

        self._data_files = []

        self.load_dataset()

    def load_dataset(self):

        # -- read the labels file
        self._labels = read_txt_lines(self._label_fp)


        # -- add examples to self._data_files
        self._get_files_for_partition()


        # -- from self._data_files to self.list
        self.list = dict()
        self.instance_ids = dict()
        if self.modality == "mixed":
            instances = dict()
            for v in self._data_files[0]:
                video_instance = self._get_instance_id_from_path(v)
                instances[video_instance] = [v]
            for a in self._data_files[1]:
                audio_instance = self._get_instance_id_from_path(a)
                tmp_list = instances.setdefault(audio_instance, list())
                tmp_list.append(a)
            i = 0
            for inst in instances.keys():
                if len(instances[inst]) != 2:
                    continue
                video, audio = instances[inst]
                if not "visual_data" in video:
                    video, audio = audio, video
                label = self._get_label_from_path(audio)
                label_index = self._labels.index(label)
                if self.merge_classes:
                    if label == "tir":
                        continue
                    label_index = label_convert_dict[label_index]
                self.list[i] = video, audio, label_index
                self.instance_ids[i] = self._get_instance_id_from_path(audio)
                i += 1
        else:
            for i, x in enumerate(self._data_files):
                label = self._get_label_from_path( x )
                self.list[i] = [ x, self._labels.index( label ) ]
                self.instance_ids[i] = self._get_instance_id_from_path( x )

        print('Partition {} loaded'.format(self._data_partition))

    def _get_instance_id_from_path(self, x):
        # for now this works for npz/npys, might break for image folders
        instance_id = x.split('/')[-1]
        return os.path.splitext( instance_id )[0]

    def _get_label_from_path(self, x):
        return x.split('/')[self.label_idx]

    def _get_files_for_partition(self):
        def get_all_paths(dir_fp):
            result = []
            # get npy/npz/avi files
            search_str_npz = os.path.join(dir_fp, '*', self._data_partition, '*.npz')
            search_str_npy = os.path.join(dir_fp, '*', self._data_partition, '*.npy')
            search_str_avi = os.path.join(dir_fp, '*', self._data_partition, '*.avi')
            result.extend( glob.glob( search_str_npz ) )
            result.extend( glob.glob( search_str_npy ) )
            result.extend( glob.glob( search_str_avi ) )

            # If we are not using the full set of labels, remove examples for labels not used
            result = [ f for f in result if f.split('/')[self.label_idx] in self._labels ]
            return result

        if self.modality == "mixed":
            self._data_files = get_all_paths('datasets/visual_data'), get_all_paths('datasets/audio_data')
        else:
            self._data_files = get_all_paths(self._data_dir)

    def load_data(self, filename):

        try:
            if filename.endswith('npz'):
                return np.load(filename)['data']
            elif filename.endswith('avi'):
                return librosa.load(filename, sr=16000)[0][-19456:]
            else:
                return np.load(filename)
        except IOError:
            print( "Error when reading file: {}".format(filename) )
            sys.exit()


    def __getitem__(self, idx):
        if self.modality == "mixed":
            video_data = self.load_data(self.list[idx][0])
            audio_data = self.load_data(self.list[idx][1])
            video_preprocessing, audio_preprocessing = self.preprocessing_func
            preprocess_data = video_preprocessing(video_data), audio_preprocessing(audio_data)
        else:
            raw_data = self.load_data(self.list[idx][0])
            preprocess_data = self.preprocessing_func(raw_data)
        label = self.list[idx][-1]
        return preprocess_data, label

    def __len__(self):
        return len(self.list)


def pad_packed_collate(batch):
    if len(batch) == 1:
        data, lengths, labels_np, = zip(*[(a, a.shape[0], b) for (a, b) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)])
        data = torch.FloatTensor(data)
        lengths = [data.size(1)]

    if len(batch) > 1:
        data_list, lengths, labels_np = zip(*[(a, a.shape[0], b) for (a, b) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)])

        if data_list[0].ndim == 3:
            max_len, h, w = data_list[0].shape  # since it is sorted, the longest video is the first one
            data_np = np.zeros(( len(data_list), max_len, h, w))
        elif data_list[0].ndim == 1:
            max_len = data_list[0].shape[0]
            data_np = np.zeros( (len(data_list), max_len))
        for idx in range( len(data_np)):
            data_np[idx][:data_list[idx].shape[0]] = data_list[idx]
        data = torch.FloatTensor(data_np)
    labels = torch.LongTensor(labels_np)
    return data, lengths, labels


def mixed_pad_packed_collate(batch):
    video_batch = [(v, lbl) for (v, a), lbl in batch]
    audio_batch = [(a, lbl) for (v, a), lbl in batch]
    video_data, video_lengths, video_labels = pad_packed_collate(video_batch)
    audio_data, audio_lengths, audio_labels = pad_packed_collate(audio_batch)
    return video_data, video_lengths, audio_data, audio_lengths, video_labels
