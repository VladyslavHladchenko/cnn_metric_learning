from cProfile import label
from json import load
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import heapq as hq
from dataclasses import dataclass, field
from typing import Any

from tools import *
from optparse import OptionParser

from cnn_workflow.utils import get_free_device, add_model_note
from cnn_workflow import cnn_workflow

dev = get_free_device()

# Global datasets
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# training and validation sets
train_set = Dataset_to_XY(torchvision.datasets.MNIST('../data', download=True, train=True, transform=transform))
train_set, val_set = train_set.split(valid_size=0.1)
# test set
test_set = Dataset_to_XY(torchvision.datasets.MNIST('../data', download=True, train=False, transform=transform))
test_set = test_set.fraction(fraction=0.1)
#
# dataloaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)
#

model_names = ['./models/' + name for name in ['net_class.pl', 'net_triplet.pl']]



class ConvNet(nn.Sequential):
    def __init__(self, num_classes: int = 10) -> None:
        layers = []
        layers += [nn.Conv2d(1, 32, kernel_size=3)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Conv2d(32, 32, kernel_size=3)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Conv2d(32, 64, kernel_size=3)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.AdaptiveAvgPool2d((2, 2))]
        layers += [nn.Flatten()]
        layers += [nn.Linear(64 * 2 * 2, num_classes)]
        super().__init__(*layers)
        self.layers = layers

    def features(self, x):
        f = nn.Sequential(*self.layers[:-1]).forward(x)
        f = nn.functional.normalize(f, p=2, dim=1)
        return f

def get_closest_to_queries(net, queries, loader):
    queries_features = net.features(queries)

    xs = []
    dists = []

    for x,l in loader:
        x,l = x.to(dev), l.to(dev)
        xs.append(x)
        features = net.features(x)
        dists_ = distances(features, queries_features)
        dists.append(dists_)

    xs = torch.concat(xs, dim=0)
    dists = torch.concat(dists, dim=0)
    _, indices =  torch.sort(dists,0)

    xs_sorted = xs[indices][:50]

    return xs_sorted


def plot_closest_to_queries(closest_images):
    n_queries = closest_images.shape[1]
    plt.clf()
    f, axarr = plt.subplots(n_queries, 1, gridspec_kw = {'wspace':0, 'hspace':0})

    for i in range(n_queries):
        ax = axarr[i]
        imgs = closest_images[:,i]
        imgs = torch.concat(imgs.unbind(0),2)

        ax.imshow(imgs.cpu().numpy().transpose(1, 2, 0))
        ax.axis('off')
        ax.axvline(x=27, color='w')


def new_net():
    return ConvNet().to(dev)


def load_net(filename):
    net = ConvNet()
    net.to(dev)
    net.load_state_dict(torch.load(filename,map_location=dev))
    return net



def distances(f1: torch.Tensor, f2: torch.Tensor):
    """All pairwise distancesbetween feature vectors in f1 and feature vectors in f2:
    f1: [N, d] array of N normalized feature vectors of dimension d
    f2: [M, d] array of M normalized feature vectors of dimension d
    return D [N,M] -- pairwise Euclidean distance matrix
    """
    assert (f1.dim() == 2)
    assert (f2.dim() == 2)
    assert (f1.size(1) == f2.size(1))

    return 2 - 2*torch.einsum("nd,md->nm", f1, f2)


def evaluate_AP(dist: np.array, labels: np.array, query_label: int):
    """Average Precision
    dits: [N] array of distances to all documents from the query
    labels: [N] labels of all documents
    query_label: label of the query document
    return: AP -- average precision, Prec -- Precision, Rec -- Recall
    """
    ii = np.argsort(dist)
    dist = dist[ii]
    labels = labels[ii]
    rel = np.equal(labels, query_label).astype(int)
    print(str.join('', ['.' if r > 0 else 'X' for r in rel[0:100]]))
    
    T = np.sum(rel)
    
    Prec = np.cumsum(rel,0)/np.arange(1,len(rel)+1)
    Rec = np.cumsum(rel,0)/T
    AP = np.sum(Prec*rel/T )

    return AP, Prec, Rec


def to_features(net, loader):
    features = []
    labels = []

    for x,y in loader:
        x,y = x.to(dev) ,y.to(dev)
        out = net.features(x)
        features.append(out)
        labels.append(y)
    
    features = torch.concat(features)
    labels = torch.concat(labels)
    
    return features, labels


def evaluate_mAP(net, dataset: DataXY):
    """
    Compute Mean Average Precision
    net: network with method features()
    dataset: dataset of input images and labels
    Returns: mAP -- mean average precision, mRec -- mean recall, mPrec -- mean precision
    """
    torch.manual_seed(1)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    
    """ use first 100 documents from this loader as queries, use all documents as possible items to retrive, exclude the query from the retrieved items """
    n_queries = 100
    mAP = 0
    mPrec = 0
    mRec = 0
    
    features, flabels = to_features(net, loader)

    queries, qlabels = features[:n_queries], flabels[:n_queries]
    queries, qlabels = queries.to(dev), qlabels.to(dev)

    for i, (q, ql) in enumerate(zip(queries, qlabels)):
        q, ql = q.to(dev), ql.to(dev)
        dists = distances(q[None,...], features).squeeze(0)
        AP, Prec, Rec = evaluate_AP(torch.concat( [ dists[0:i], dists[i+1:]]).detach().cpu().numpy(),
                                    torch.concat( [ flabels[0:i], flabels[i+1:]]).detach().cpu().numpy(),
                                    ql.detach().cpu().item())

        mAP += AP
        mPrec += Prec
        mRec += Rec

    mAP/=n_queries
    mPrec/=n_queries
    mRec/=n_queries

    return mAP, mPrec, mRec


def evaluate_acc(net, loss_f, loader):
    net.eval()
    with torch.no_grad():
        acc = 0
        loss = 0
        n_data = 0
        for i, (data, target) in enumerate(test_loader):
            data, target = data, target
            y = net(data)
            l = loss_f(y, target)
            loss += l.sum()
            acc += (torch.argmax(y, dim=1) == target).float().sum().item()
            n_data += data.size(0)
        acc /= n_data
        loss /= n_data
    return (loss, acc)


def train_class(net, train_loader, val_loader, epochs=20, name: str = None):
    loss_f = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        print("Epoch {}".format(epoch))
        train_acc = 0
        train_loss = 0
        n_train_data = 0
        net.train()
        for i, (data, target) in enumerate(train_loader):
            y = net(data)
            l = loss_f(y, target)
            train_loss += l.sum().item()
            train_acc += (torch.argmax(y, dim=1) == target).float().sum().item()
            n_train_data += data.size(0)
            optimizer.zero_grad()
            l.mean().backward()
            optimizer.step()
        train_loss /= n_train_data
        train_acc /= n_train_data
        #
        val_loss, val_acc = evaluate_acc(net, loss_f, val_loader)
        print(f'Epoch: {epoch} mean loss: {train_loss}')
        print("Train accuracy {}, Val accuracy: {}".format(train_acc, val_acc))
        if name is not None:
            torch.save(net.state_dict(), name)


def triplet_loss(features: torch.Tensor, labels: torch.Tensor, alpha=0.5, **kwargs):
    """
    triplet loss
    features [N, d] tensor of features for N data points
    labels [N] true labels of the data points

    Implement: max(0, d(a,p) - d(a,n) + alpha )) for a=0:10 and all valid p,n in the batch
    """
    L = 0
    
    for a_idx in range(10):
        L_a = 0
        a = features[a_idx]
        y = labels[a_idx]
        ps = features[y == labels]
        ns = features[y != labels]
        
        d_p = distances(a.unsqueeze(0), ps).squeeze()
        d_n = distances(a.unsqueeze(0), ns).squeeze()

        d_p = d_p[d_p!=0]
        
        all_pairs = torch.stack(torch.meshgrid(d_p, d_n)).reshape(2,-1).T
        
        L_a = F.relu(all_pairs[:,0] - all_pairs[:,1] + alpha).sum()
        L += L_a
    
    return L


def train_triplets(net, data_loader, epochs=20, name: str = None):
    """
    training with triplet loss
    """
    net.to(dev)
    opt = optim.Adam(net.parameters(), lr=0.001)

    add_model_note(net, name)

    results = cnn_workflow.train(net, dev, data_loader, opt, loss_fn=triplet_loss, epoch_num=epochs, save_epochs=True, no_acc = True)

    return results

    # for e in range(epochs):
    #     avg_loss = 0
    #     for x,y in train_loader:
    #         x,y = x.to(dev), y.to(dev)
    #         net.train()
            
    #         opt.zero_grad()
    #         output = net(x)
            
    #         trn_loss = triplet_loss(output, y)
    #         trn_loss.backward()
    #         opt.step()
    #         avg_loss+=trn_loss.item()

    #         net.train()

    #     avg_loss/=len(train_loader.dataset)

    #     print(f"epoch {e} avg trn loss {avg_loss}")



if __name__ == '__main__':
    op = OptionParser()
    op.add_option("--train", type=int, default=-1, help="run training: 0 -- classification loss, 1 -- triplet loss")
    op.add_option("--eval", type=int, default=-1, help="run evaluation: 0 -- classification loss, 1 -- triplet loss")
    op.add_option("-e", "--epochs", type=int, default=10, help="training epochs")
    (opts, args) = op.parse_args()
    o = dotdict(**vars(opts))

    if o.train == 0:
        net = ConvNet(10)
        net.to(dev)
        train_class(net, train_loader, val_loader, epochs=o.epochs)
        torch.save(net.state_dict(), model_names[0])

    if o.train == 1:
        net = ConvNet(10)
        net.to(dev)
        train_triplets(net, train_loader, epochs=o.epoch, name=model_names[1])

    if o.eval > -1:
        net = ConvNet(10)
        net.to(dev)
        net.load_state_dict(torch.load(model_names[o.eval]))
        # loss_f = nn.CrossEntropyLoss(reduction='none')
        # loss, acc = evaluate_acc(net, loss_f, test_loader)
        # print(f"Test accuracy: {acc*100:3.2f}%")
        mAP, mPrec, mRec = evaluate_mAP(net, test_set)
        print(f"Test mAP: {mAP:3.2f}")
