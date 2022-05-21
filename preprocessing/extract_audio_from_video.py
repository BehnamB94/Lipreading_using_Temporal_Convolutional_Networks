#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


"""Transforms avi audio to npz. Code has strong assumptions on the dataset organization!"""

import os
import librosa
import argparse
import warnings
from multiprocessing import Pool

from utils import *

warnings.filterwarnings('ignore')

def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Extract Audio Waveforms')
    # -- utils
    parser.add_argument('--video-direc', default=None, help='raw video directory')
    parser.add_argument('--filename-path', default='./lrw500_detected_face.csv', help='list of detected video and its subject ID')
    parser.add_argument('--save-direc', default=None, help='the directory of saving audio waveforms (.npz)')
    # -- test set only
    parser.add_argument('--testset-only', default=False, action='store_true', help='process testing set only')

    args = parser.parse_args()
    return args

args = load_args()

lines = open(args.filename_path).read().splitlines()
lines = list(filter(lambda x: 'test' == x.split('/')[-2], lines)) if args.testset_only else lines

def preprocess(line):
    filename, person_id = line.split(',')
    video_pathname = os.path.join(args.video_direc, filename+'.avi')
    dst_pathname = os.path.join( args.save_direc, filename+'.npz')

    if os.path.exists(dst_pathname):
        return

    print('Processing.\t{}'.format(filename))
    assert os.path.isfile(video_pathname), "File does not exist. Path input: {}".format(video_pathname)

    max_length = 19456
    audio = librosa.load(video_pathname, sr=16000)[0]
    argmax = audio.argmax()
    start = max(0, argmax-max_length//2)
    end = min(len(audio), max(start+max_length, argmax+max_length//2))
    data = audio[start:end]
    save2npz(dst_pathname, data=data)

with Pool() as pool:
    pool.map(preprocess, lines)
print('Done.')
