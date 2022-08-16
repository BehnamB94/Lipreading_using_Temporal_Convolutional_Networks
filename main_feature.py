#%%
import os
from glob import glob
from typing import List

import numpy as np
from torch.utils.data import DataLoader, Dataset

with open("labels/labels.txt") as ff:
    labels = ff.readlines()
labels = list(map(str.strip, labels))

def label_change(old_lbl):
    if old_lbl == "panj" or old_lbl == "mehr":
        return "panj_mehr"
    elif old_lbl == "yek" or old_lbl == "dey" or old_lbl == "se":
        return "yek_dey_se"
    elif old_lbl == "do" or old_lbl == "noh":
        return "do_noh"
    return old_lbl
to_be_removed = "tir"
to_be_removed_ind = labels.index(to_be_removed)

mapped_labels = list(map(label_change, labels))
new_labels = list(sorted(set(mapped_labels)))
new_labels.remove(to_be_removed)

lbl_dict = dict()
for ind in range(len(labels)):
    if ind != to_be_removed_ind:
        lbl_dict[ind] = new_labels.index(mapped_labels[ind])

def get_partition(name):
    total_path = f"features/all-{name}.npz"
    if os.path.exists(total_path):
        data = np.load(total_path)
        video = data["video"]
        audio = data["audio"]
        label = data["labels"]
        return video, audio, label
    vlist = list()
    alist = list()
    llist = list()
    for path in glob(f"features/{name}-*.npz"):
        data = np.load(path)
        video = data["video"]
        audio = data["audio"]
        label = data["labels"]
        vlist.append(video)
        alist.append(audio)
        llist.append(label)
    varr = np.concatenate(vlist, axis=0)
    aarr = np.concatenate(alist, axis=0)
    larr = np.concatenate(llist, axis=0)
    valid = larr != to_be_removed_ind
    varr = varr[valid]
    aarr = aarr[valid]
    larr = larr[valid]
    larr = np.array([lbl_dict[a] for a in larr])
    np.savez_compressed(total_path, video=varr, audio=aarr, labels=larr)
    return varr, aarr, larr


train_varr, train_aarr, train_larr = get_partition("train")
valid_varr, valid_aarr, valid_larr = get_partition("val")
test_varr, test_aarr, test_larr = get_partition("test")
#%%

import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, hidden_list: List[int]) -> None:
        super().__init__()
        layres = [
            nn.Linear(hidden_list[i], hidden_list[i + 1])
            for i in range(0, len(hidden_list) - 1)
        ]
        self.fc = nn.Sequential(*layres)

    def forward(self, x):
        return self.fc(x)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    loss_list = list()
    for batch, (X, y) in enumerate(dataloader):
        # X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss)
    return sum(loss_list) / len(loss_list)


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    all_y = list()
    all_predicts = list()
    with torch.no_grad():
        for X, y in dataloader:
            # X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            all_y += y
            all_predicts += pred.numpy().tolist()
    test_loss /= num_batches
    correct /= size
    return test_loss, correct, np.array(all_predicts), np.array(all_y)


model = Model([768 * 2, 256, 128, (len(new_labels))])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
loss_fn = nn.CrossEntropyLoss()


class AVDataset(Dataset):
    def __init__(self, varr, aarr, larr) -> None:
        super().__init__()
        self.varr = varr
        self.aarr = aarr
        self.larr = larr

    def __getitem__(self, index):
        v = self.varr[index]
        a = self.varr[index]
        x = np.concatenate([v, a])
        return x, self.larr[index]

    def __len__(self):
        return self.varr.shape[0]

train_dataloader = DataLoader(
    AVDataset(train_varr, train_aarr, train_larr), batch_size=32
)
valid_dataloader = DataLoader(
    AVDataset(valid_varr, valid_aarr, valid_larr), batch_size=32
)
test_dataloader = DataLoader(AVDataset(test_varr, test_aarr, test_larr), batch_size=32)

epochs = 500
not_dec = 0
best_valid_loss = 10
for t in range(epochs):
    train_loss = train(train_dataloader, model, loss_fn, optimizer)
    valid_loss, valid_acc, _, _ = test(valid_dataloader, model, loss_fn)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        not_dec = 0
    else:
        not_dec += 1
    if not_dec > 10:
        break

    print(
        f"[{t+1:>3d}/{epochs:>3d}] train loss: {train_loss:>7f} | "
        f"valid loss: {valid_loss:>7f} | valid acc: {(100*valid_acc):>0.1f}%"
    )
print("Done!")

test_loss, test_acc, predicts, y_true = test(test_dataloader, model, loss_fn)
print(f"test loss: {valid_loss:>7f} | test acc: {(100*valid_acc):>0.1f}%")
#%%

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, predicts.argmax(axis=1))
plt.imshow(cm)

# %%
