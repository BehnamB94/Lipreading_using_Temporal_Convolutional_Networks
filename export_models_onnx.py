import os
from typing import List

import numpy as np
import torch
import torch.nn as nn

from lipreading.dataloaders import get_data_loaders
from lipreading.utils import load_model
from main_mixed import get_model_from_json


video_model = get_model_from_json("configs/lrw_snv1x_dsmstcn3x.json", "video")
audio_model = get_model_from_json("configs/lrw_snv1x_dsmstcn3x.json", "raw_audio")


def load_weights(sub_model, path):
    assert path.endswith(".tar") and os.path.isfile(path), f"{path}: not exist"
    sub_model = load_model(
        path,
        sub_model,
        allow_size_mismatch=True,
    )


load_weights(video_model, "train_logs/tcn/2022-07-03T18:27:39/ckpt.best.pth.tar")
load_weights(audio_model, "train_logs/tcn/2022-07-04T08:50:31/ckpt.best.pth.tar")


class MixModel(nn.Module):
    def __init__(self, hidden_list: List[int]) -> None:
        super().__init__()
        layers = [
            nn.Linear(hidden_list[i], hidden_list[i + 1])
            for i in range(0, len(hidden_list) - 1)
        ]
        self.fc = nn.Sequential(*layers)
        self.active = nn.Softmax(dim=1)

    def forward(self, x):
        return self.active(self.fc(x))


mix_model = MixModel([768 * 2, 256, 128, 17])
mix_model.load_state_dict(torch.load("shuffle_mix.pth.tar"))


class SuperModel(nn.Module):
    def __init__(self, video_model, audio_model, mix_model) -> None:
        super().__init__()
        self.video_model = video_model
        self.audio_model = audio_model
        self.mix_model = mix_model

    def forward(self, video, video_lengths, audio, audio_lengths):
        video_emb = self.video_model(video, lengths=video_lengths)
        audio_emb = self.audio_model(audio, lengths=audio_lengths)
        concat = torch.concat([video_emb, audio_emb], dim=1)
        return self.mix_model(concat)


model = SuperModel(video_model, audio_model, mix_model)
torch.save(model.state_dict(), "lipreading.pth.tar")
exit()

class A:
    pass


args = A()
args.modality = "mixed"
args.data_dir = "asdf"
args.label_path = "labels/labels.txt"
args.annonation_direc = ""
args.batch_size = 2
args.workers = 4
data_loaders = get_data_loaders(args)

(
    video_input,
    video_lengths,
    audio_input,
    audio_lengths,
    labels,
) = next(iter(data_loaders["test"]))

video_lengths = torch.tensor(video_lengths)
audio_lengths = torch.tensor(audio_lengths)

logits = model(
    video=video_input.unsqueeze(1),
    video_lengths=video_lengths,
    audio=audio_input.unsqueeze(1),
    audio_lengths=audio_lengths,
)

torch.onnx.export(
    model,  # model being run
    (
        video_input.unsqueeze(1),
        video_lengths,
        audio_input.unsqueeze(1),
        audio_lengths,
    ),  # model input (or a tuple for multiple inputs)
    "lipreading.onnx",  # where to save the model (can be a file or file-like object)
    export_params=True,  # store the trained parameter weights inside the model file
    do_constant_folding=True,  # whether to execute constant folding for optimization
    verbose=True,  # Causes the exporter to print out a human-readable representation of the network
    input_names=[
        "video",
        "video_lengths",
        "audio",
        "audio_lengths",
    ],  # the model's input names
    output_names=["logits"],  # the model's output names
    dynamic_axes={
        "video": {0: "batch_size"},
        "video_lengths": {0: "batch_size"},
        "audio": {0: "batch_size"},
        "audio_lengths": {0: "batch_size"},
        "logits": {0: "batch_size"},
    },
    # operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
    opset_version=10,
)
