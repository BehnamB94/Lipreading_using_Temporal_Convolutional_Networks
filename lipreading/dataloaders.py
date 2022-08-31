import torch
import numpy as np
from lipreading.preprocess import *
from lipreading.dataset import MyDataset, mixed_pad_packed_collate, pad_packed_collate


def get_preprocessing_pipelines(modality):
    # -- preprocess for the video stream
    # -- LRW config
    crop_size = (88, 88)
    (mean, std) = (0.421, 0.165)
    video_preprocessing = {}
    video_preprocessing["train"] = Compose(
        [
            RgbToGray(),
            Normalize(0.0, 255.0),
            RandomCrop(crop_size),
            HorizontalFlip(0.5),
            Normalize(mean, std),
        ]
    )
    video_preprocessing["val"] = Compose(
        [
            RgbToGray(),
            Normalize(0.0, 255.0),
            CenterCrop(crop_size),
            Normalize(mean, std),
        ]
    )
    video_preprocessing["test"] = video_preprocessing["val"]

    audio_preprocessing = {}
    audio_preprocessing["train"] = Compose(
        [
            AddNoise(noise=np.load("./data/babbleNoise_resample_16K.npy")),
            NormalizeUtterance(),
        ]
    )
    audio_preprocessing["val"] = NormalizeUtterance()
    audio_preprocessing["test"] = NormalizeUtterance()

    if modality == "video":
        return video_preprocessing
    elif modality == "raw_audio":
        return audio_preprocessing
    elif modality == "mixed":
        return {
            "train": (video_preprocessing["train"], audio_preprocessing["train"]),
            "val": (video_preprocessing["val"], audio_preprocessing["val"]),
            "test": (video_preprocessing["test"], audio_preprocessing["test"]),
        }


def get_data_loaders(args):
    preprocessing = get_preprocessing_pipelines( args.modality)

    # create dataset object for each partition
    dsets = {partition: MyDataset(
                modality=args.modality,
                data_partition=partition,
                data_dir=args.data_dir,
                label_fp=args.label_path,
                annonation_direc=args.annonation_direc,
                preprocessing_func=preprocessing[partition],
                data_suffix='.npz',
                merge_classes=args.merge_classes,
                ) for partition in ['train', 'val', 'test']}
    collate_fn = mixed_pad_packed_collate if args.modality == "mixed" else pad_packed_collate
    dset_loaders = {x: torch.utils.data.DataLoader(
                        dsets[x],
                        batch_size=args.batch_size,
                        shuffle=True,
                        collate_fn=collate_fn,
                        pin_memory=True,
                        num_workers=args.workers,
                        worker_init_fn=np.random.seed(1)) for x in ['train', 'val', 'test']}
    return dset_loaders
