#%%
import os
from glob import glob
from typing import List

import numpy as np
from torch.utils.data import DataLoader, Dataset

with open("labels/labels.txt") as ff:
    real_labels = ff.readlines()
real_labels = list(map(str.strip, real_labels))

def label_change(old_lbl):
    if old_lbl == "panj" or old_lbl == "mehr":
        return "panj_mehr"
    elif old_lbl == "yek" or old_lbl == "dey" or old_lbl == "se":
        return "yek_dey_se"
    elif old_lbl == "do" or old_lbl == "noh":
        return "do_noh"
    return old_lbl
to_be_removed = "tir"
to_be_removed_ind = real_labels.index(to_be_removed)

real_label_indexes = [ind for ind in range(len(real_labels)) if ind != to_be_removed_ind]

mapped_labels = list(map(label_change, real_labels))
new_labels = list(sorted(set(mapped_labels)))
new_labels.remove(to_be_removed)

lbl_dict = dict()
new_label_to_text = dict()
for ind in range(len(real_labels)):
    if ind != to_be_removed_ind:
        new_label_ind = new_labels.index(mapped_labels[ind])
        lbl_dict[ind] = new_label_ind
        label_set = new_label_to_text.setdefault(new_label_ind, set())
        label_set.add(real_labels[ind])

def get_partition(name):
    total_path = f"features/all-{name}.npz"
    if os.path.exists(total_path):
        data = np.load(total_path)
        video = data["video"]
        audio = data["audio"]
        label = data["labels"]
        real_label = data["real_labels"]
        return video, audio, label, real_label
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
    new_larr = np.array([lbl_dict[a] for a in larr])
    np.savez_compressed(total_path, video=varr, audio=aarr, labels=new_larr, real_labels=larr)
    return varr, aarr, new_larr, larr


train_varr, train_aarr, train_larr, _ = get_partition("train")
valid_varr, valid_aarr, valid_larr, _ = get_partition("val")
test_varr, test_aarr, test_larr, test_real_larr = get_partition("test")
#%%

import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, hidden_list: List[int]) -> None:
        super().__init__()
        layers = [
            nn.Linear(hidden_list[i], hidden_list[i + 1])
            for i in range(0, len(hidden_list) - 1)
        ]
        self.fc = nn.Sequential(*layers)
        self.active = nn.Softmax()

    def forward(self, x):
        return self.active(self.fc(x))


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
    all_results = np.array(all_predicts)
    all_predicts = all_results.argmax(axis=1)
    all_results.sort(axis=1)
    confidences = all_results[:, -1] - all_results[:, -2]
    return test_loss, correct, all_predicts, confidences, np.array(all_y)


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
patience = 10
not_dec = 0
best_valid_loss = 10
for t in range(epochs):
    train_loss = train(train_dataloader, model, loss_fn, optimizer)
    valid_loss, valid_acc, _, _, _ = test(valid_dataloader, model, loss_fn)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        not_dec = 0
    else:
        not_dec += 1
    if not_dec > patience:
        break

    print(
        f"[{t+1:>3d}/{epochs:>3d}] train loss: {train_loss:>7f} | "
        f"valid loss: {valid_loss:>7f} | valid acc: {(100*valid_acc):>0.1f}%"
    )
print("Done!")

#%%
test_loss, test_acc, predicts, confidences, y_true = test(test_dataloader, model, loss_fn)
print(f"test loss: {valid_loss:>7f} | test acc: {(100*valid_acc):>0.1f}%")

#%%
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true, predicts)
dsp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=new_labels)
fig, ax = plt.subplots(figsize=(10,10))
dsp.plot(ax=ax)
plt.xticks(rotation=90)
plt.show()

# %%
sequence_length_list = range(3, 8)
instances = 10000

from numpy import random
import pandas as pd


def get_genuine_predict(real_label):
    sub_list = np.where(test_real_larr == real_label)[0]
    selected_index = np.random.choice(sub_list)
    return predicts[selected_index], confidences[selected_index]

def get_impostor_predict(real_label):
    sub_list = np.where(test_real_larr != real_label)[0]
    selected_index = np.random.choice(sub_list)
    return predicts[selected_index], confidences[selected_index]

genuine_predict_func = np.vectorize(get_genuine_predict)
impostor_predict_func = np.vectorize(get_impostor_predict)


df_dict = dict()
for sequence_length in sequence_length_list:
    patterns = list()
    for i in range(instances):
        r = random.choice(real_label_indexes, replace=False, size=(sequence_length,))
        patterns.append(r)
    patterns = np.array(patterns)

    true_label_func = np.vectorize(lambda x: lbl_dict[x])
    true_labels = true_label_func(patterns)

    genuine_labels, genuine_confidence = genuine_predict_func(patterns)
    genuine_compare = genuine_labels == true_labels
    genuine_scores = (genuine_compare * genuine_confidence).mean(axis=1)

    impostor_labels, impostor_confidence = impostor_predict_func(patterns)
    impostor_compare = impostor_labels == true_labels
    impostor_scores = (impostor_compare * impostor_confidence).mean(axis=1)

    import sklearn.metrics

    label_array = [0] * instances + [1]  * instances
    score_array = np.concatenate([impostor_scores, genuine_scores])
    fmr, tpr, thr = sklearn.metrics._ranking.roc_curve(label_array, score_array)
    euc = 1 - sklearn.metrics.auc(fmr, tpr)
    fnmr = 1 - tpr

    eer_index = np.argmin(np.abs(fmr - fnmr))
    eer = (fmr[eer_index] + fnmr[eer_index]) / 2

    print(f'EUC={euc:.4} | EER={eer:.4} (at threshold = {thr[eer_index]:.3})')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(fmr, fnmr, label=f"{sequence_length} (EER={eer:.4})")
    df = pd.DataFrame({"Threshold": thr, "False Match Rate": fmr, "False Non-Match Rate": fnmr})
    df_dict[f"sequence length = {sequence_length}"] = df

writer = pd.ExcelWriter('results.xlsx', engine='xlsxwriter')
for sheet_name, df in df_dict.items():
    df.to_excel(writer, sheet_name, index=None)
writer.save()

plt.legend(title="Sequence Length")
plt.xlabel("FMR")
plt.ylabel("FNMR")
plt.show()
# %%
torch.save(model.state_dict(), "shuffle_mix.pth.tar")
