#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

""" TCN for lipreading"""

import os
import time
import random
import argparse
from turtle import forward
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from lipreading.utils import get_save_folder
from lipreading.utils import load_json
from lipreading.utils import load_model, CheckpointSaver
from lipreading.utils import get_logger, update_logger_batch
from lipreading.utils import showLR, calculateNorm2, AverageMeter
from lipreading.model import Lipreading
from lipreading.mixup import mixed_mixup_data, mixup_criterion
from lipreading.optim_utils import get_optimizer, CosineScheduler
from lipreading.dataloaders import get_data_loaders


def load_args(default_config=None):
    parser = argparse.ArgumentParser(description="Pytorch Lipreading ")
    # -- dataset config
    parser.add_argument("--dataset", default="lrw", help="dataset selection")
    parser.add_argument(
        "--num-classes", type=int, default=500, help="Number of classes"
    )
    parser.add_argument(
        "--modality", default="mixed", choices=["mixed"], help="choose the modality"
    )
    # -- directory
    parser.add_argument(
        "--data-dir",
        default="./datasets/LRW_h96w96_mouth_crop_gray",
        help="Loaded data directory",
    )
    parser.add_argument(
        "--label-path",
        type=str,
        default="./labels/500WordsSortedList.txt",
        help="Path to txt file with labels",
    )
    parser.add_argument(
        "--annonation-direc", default=None, help="Loaded data directory"
    )
    # -- model config
    parser.add_argument(
        "--backbone-type",
        type=str,
        default="resnet",
        choices=["resnet", "shufflenet"],
        help="Architecture used for backbone",
    )
    parser.add_argument(
        "--relu-type",
        type=str,
        default="relu",
        choices=["relu", "prelu"],
        help="what relu to use",
    )
    parser.add_argument(
        "--width-mult",
        type=float,
        default=1.0,
        help="Width multiplier for mobilenets and shufflenets",
    )
    # -- TCN config
    parser.add_argument(
        "--tcn-kernel-size",
        type=int,
        nargs="+",
        help="Kernel to be used for the TCN module",
    )
    parser.add_argument(
        "--tcn-num-layers",
        type=int,
        default=4,
        help="Number of layers on the TCN module",
    )
    parser.add_argument(
        "--tcn-dropout",
        type=float,
        default=0.2,
        help="Dropout value for the TCN module",
    )
    parser.add_argument(
        "--tcn-dwpw",
        default=False,
        action="store_true",
        help="If True, use the depthwise seperable convolution in TCN architecture",
    )
    parser.add_argument(
        "--tcn-width-mult", type=int, default=1, help="TCN width multiplier"
    )
    # -- train
    parser.add_argument("--training-mode", default="tcn", help="tcn")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument(
        "--optimizer", type=str, default="adamw", choices=["adam", "sgd", "adamw"]
    )
    parser.add_argument("--lr", default=3e-4, type=float, help="initial learning rate")
    parser.add_argument("--init-epoch", default=0, type=int, help="epoch to start at")
    parser.add_argument("--epochs", default=80, type=int, help="number of epochs")
    parser.add_argument(
        "--test", default=False, action="store_true", help="training mode"
    )
    parser.add_argument("--save-features", default=False, action="store_true")
    # -- mixup
    parser.add_argument(
        "--alpha",
        default=0.0,
        type=float,
        help="interpolation strength (uniform=1., ERM=0.)",
    )
    # -- test
    parser.add_argument(
        "--mixed-model-path", type=str, default=None, help="Pretrained model pathname"
    )
    parser.add_argument(
        "--video-model-path", type=str, default=None, help="Pretrained model pathname"
    )
    parser.add_argument(
        "--audio-model-path", type=str, default=None, help="Pretrained model pathname"
    )
    parser.add_argument(
        "--allow-size-mismatch",
        default=False,
        action="store_true",
        help="If True, allows to init from model with mismatching weight tensors. Useful to init from model with diff. number of classes",
    )
    # -- feature extractor
    parser.add_argument(
        "--extract-feats", default=False, action="store_true", help="Feature extractor"
    )
    parser.add_argument(
        "--mouth-patch-path",
        type=str,
        default=None,
        help="Path to the mouth ROIs, assuming the file is saved as numpy.array",
    )
    parser.add_argument(
        "--mouth-embedding-out-path",
        type=str,
        default=None,
        help="Save mouth embeddings to a specificed path",
    )
    # -- json pathname
    parser.add_argument(
        "--video-config-path",
        type=str,
        default=None,
        help="Video model configuration with json format",
    )
    parser.add_argument(
        "--audio-config-path",
        type=str,
        default=None,
        help="Audio model configuration with json format",
    )
    # -- other vars
    parser.add_argument("--interval", default=50, type=int, help="display interval")
    parser.add_argument(
        "--workers", default=8, type=int, help="number of data loading workers"
    )
    # paths
    parser.add_argument(
        "--logging-dir",
        type=str,
        default="./train_logs",
        help="path to the directory in which to save the log file",
    )

    args = parser.parse_args()
    return args


args = load_args()

# torch.manual_seed(1)
# np.random.seed(1)
# random.seed(1)
# torch.backends.cudnn.benchmark = True


def save_features(model, dset_loaders):
    model.eval()
    with torch.no_grad():
        for dlabel, dloader in dset_loaders.items():
            for batch_idx, (
                video_input,
                video_lengths,
                audio_input,
                audio_lengths,
                labels,
            ) in enumerate(dloader):
                npz_filename = f"features/{dlabel}-{batch_idx}.npz"
                # if os.path.exists(npz_filename):
                #     continue
                logits = model(
                    video=video_input.unsqueeze(1),
                    video_lengths=video_lengths,
                    audio=audio_input.unsqueeze(1),
                    audio_lengths=audio_lengths,
                )
                np.savez_compressed(
                    npz_filename,
                    video=logits[0].numpy(),
                    audio=logits[1].numpy(),
                    labels=labels,
                )


def evaluate(model, dset_loader, criterion):

    model.eval()

    running_loss = 0.0
    running_corrects = 0.0

    with torch.no_grad():
        for (
            video_input,
            video_lengths,
            audio_input,
            audio_lengths,
            labels,
        ) in dset_loader:
            logits = model(
                video=video_input.unsqueeze(1),
                video_lengths=video_lengths,
                audio=audio_input.unsqueeze(1),
                audio_lengths=audio_lengths,
            )
            _, preds = torch.max(F.softmax(logits, dim=1).data, dim=1)
            running_corrects += preds.eq(labels.view_as(preds)).sum().item()

            loss = criterion(logits, labels)
            running_loss += loss.item() * video_input.size(0)

    print(
        "{} in total\tCR: {}".format(
            len(dset_loader.dataset), running_corrects / len(dset_loader.dataset)
        )
    )
    return running_corrects / len(dset_loader.dataset), running_loss / len(
        dset_loader.dataset
    )


def train(model, dset_loader, criterion, epoch, optimizer, logger):
    data_time = AverageMeter()
    batch_time = AverageMeter()

    lr = showLR(optimizer)

    logger.info("-" * 10)
    logger.info("Epoch {}/{}".format(epoch, args.epochs - 1))
    logger.info("Current learning rate: {}".format(lr))

    model.train()
    running_loss = 0.0
    running_corrects = 0.0
    running_all = 0.0

    end = time.time()
    for batch_idx, (
        video_input,
        video_lengths,
        audio_input,
        audio_lengths,
        labels,
    ) in enumerate(dset_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # --
        video_input, audio_input, labels_a, labels_b, lam = mixed_mixup_data(
            video_input, audio_input, labels, args.alpha
        )

        optimizer.zero_grad()
        logits = model(
            video=video_input.unsqueeze(1),
            video_lengths=video_lengths,
            audio=audio_input.unsqueeze(1),
            audio_lengths=audio_lengths,
        )

        loss_func = mixup_criterion(labels_a, labels_b, lam)
        loss = loss_func(criterion, logits)

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # -- compute running performance
        _, predicted = torch.max(F.softmax(logits, dim=1).data, dim=1)
        running_loss += loss.item() * video_input.size(0)
        running_corrects += (
            lam * predicted.eq(labels_a.view_as(predicted)).sum().item()
            + (1 - lam) * predicted.eq(labels_b.view_as(predicted)).sum().item()
        )
        running_all += video_input.size(0)
        # -- log intermediate results
        if batch_idx % args.interval == 0 or (batch_idx == len(dset_loader) - 1):
            update_logger_batch(
                args,
                logger,
                dset_loader,
                batch_idx,
                running_loss,
                running_corrects,
                running_all,
                batch_time,
                data_time,
            )

    return model


def get_model_from_json(config_path, modality):
    assert config_path.endswith(".json") and os.path.isfile(
        config_path
    ), "'.json' config path does not exist. Path input: {}".format(config_path)
    args_loaded = load_json(config_path)
    tcn_options = {
        "num_layers": args_loaded["tcn_num_layers"],
        "kernel_size": args_loaded["tcn_kernel_size"],
        "dropout": args_loaded["tcn_dropout"],
        "dwpw": args_loaded["tcn_dwpw"],
        "width_mult": args_loaded["tcn_width_mult"],
    }
    model = Lipreading(
        modality=modality,
        num_classes=args.num_classes,
        tcn_options=tcn_options,
        backbone_type=args_loaded["backbone_type"],
        relu_type=args_loaded["relu_type"],
        width_mult=args_loaded["width_mult"],
    )
    calculateNorm2(model)
    return model


def main():

    # -- logging
    save_path = get_save_folder(args)
    print("Model and log being saved in: {}".format(save_path))
    logger = get_logger(args, save_path)
    ckpt_saver = CheckpointSaver(save_path)

    # -- get model
    video_model = get_model_from_json(args.video_config_path, "video")
    audio_model = get_model_from_json(args.audio_config_path, "raw_audio")

    class MixedModel(nn.Module):
        def __init__(self, video_model, audio_model, num_classes) -> None:
            super(MixedModel, self).__init__()
            self.video_model = video_model
            self.audio_model = audio_model
            # self.norm1 = nn.LayerNorm(num_classes)
            # self.norm2 = nn.LayerNorm(num_classes)
            # self.fc1 = nn.Linear(num_classes * 2, num_classes * 2)
            # self.fc2 = nn.Linear(num_classes * 2, num_classes)

        def forward(self, video, video_lengths, audio, audio_lengths):
            video_emb = self.video_model(video, lengths=video_lengths)
            audio_emb = self.audio_model(audio, lengths=audio_lengths)
            return video_emb, audio_emb
            # video_emb = self.norm1(video_emb)
            # audio_emb = self.norm2(audio_emb)
            # concat = torch.concat([video_emb, audio_emb], dim=1)
            # return self.fc2(self.fc1(concat))

    model = MixedModel(video_model, audio_model, num_classes=args.num_classes)
    # -- get dataset iterators
    dset_loaders = get_data_loaders(args)
    # -- get loss function
    criterion = nn.CrossEntropyLoss()
    # -- get optimizer
    optimizer = get_optimizer(args, optim_policies=model.parameters())
    # -- get learning rate scheduler
    scheduler = CosineScheduler(args.lr, args.epochs)

    if (args.video_model_path and args.audio_model_path) or args.mixed_model_path:

        def load_weights(sub_model, path):
            assert path.endswith(".tar") and os.path.isfile(path), f"{path}: not exist"
            sub_model = load_model(
                path,
                sub_model,
                allow_size_mismatch=args.allow_size_mismatch,
            )
            logger.info("SubModel loaded from {}".format(path))

        if args.mixed_model_path:
            load_weights(model, args.mixed_model_path)
        else:
            load_weights(video_model, args.video_model_path)
            load_weights(audio_model, args.audio_model_path)
            for param in list(video_model.parameters())[:-2]:
                param.requires_grad = False
            for param in list(audio_model.parameters())[:-2]:
                param.requires_grad = False
        if args.test:
            acc_avg_test, loss_avg_test = evaluate(
                model, dset_loaders["test"], criterion
            )
            logger.info(
                "Test-time performance on partition {}: Loss: {:.4f}\tAcc:{:.4f}".format(
                    "test", loss_avg_test, acc_avg_test
                )
            )
            return
        if args.save_features:
            save_features(model, dset_loaders)
            return
    epoch = args.init_epoch

    while epoch < args.epochs:
        model = train(model, dset_loaders["train"], criterion, epoch, optimizer, logger)
        acc_avg_val, loss_avg_val = evaluate(model, dset_loaders["val"], criterion)
        logger.info(
            "{} Epoch:\t{:2}\tLoss val: {:.4f}\tAcc val:{:.4f}, LR: {}".format(
                "val", epoch, loss_avg_val, acc_avg_val, showLR(optimizer)
            )
        )
        # -- save checkpoint
        save_dict = {
            "epoch_idx": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        ckpt_saver.save(save_dict, loss_avg_val)
        scheduler.adjust_lr(optimizer, epoch)
        epoch += 1

    # -- evaluate best-performing epoch on test partition
    best_fp = os.path.join(ckpt_saver.save_dir, ckpt_saver.best_fn)
    _ = load_model(best_fp, model)
    acc_avg_test, loss_avg_test = evaluate(model, dset_loaders["test"], criterion)
    logger.info(
        "Test time performance of best epoch: {} (loss: {})".format(
            acc_avg_test, loss_avg_test
        )
    )


if __name__ == "__main__":
    main()
