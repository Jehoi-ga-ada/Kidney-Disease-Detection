# encoding: utf-8

"""
The main CheXNet model implementation.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import KidneyDataset
from sklearn.metrics import roc_auc_score
from torchvision.models import DenseNet121_Weights
from tqdm import tqdm  # for progress bar

CKPT_PATH = 'model.pth.tar'
N_CLASSES = 4
CLASS_NAMES = ['Tumor', 'Normal', 'Cyst', 'Stone']
DATA_DIR = 'CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone'
TRAIN_IMAGE_LIST = 'CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Labels/train_list.csv'
VALID_IMAGE_LIST = 'CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Labels/valid_list.csv'
TEST_IMAGE_LIST = 'CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Labels/test_list.csv'
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 0.001


def main():
    cudnn.benchmark = True

    # Initialize and load the model
    model = DenseNet121(N_CLASSES).cpu()
    model = torch.nn.DataParallel(model).cpu()

    if os.path.isfile(CKPT_PATH):
        print("=> loading checkpoint")
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_dataset = KidneyDataset(
        data_dir=DATA_DIR,
        image_list_file=TRAIN_IMAGE_LIST,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )

    valid_dataset = KidneyDataset(
        data_dir=DATA_DIR,
        image_list_file=VALID_IMAGE_LIST,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )

    test_dataset = KidneyDataset(
        data_dir=DATA_DIR,
        image_list_file=TEST_IMAGE_LIST,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
        ])
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_auc = 0.0

    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch")
        for i, (inputs, targets) in enumerate(train_loader_tqdm):
            inputs = inputs.cpu()
            targets = targets.cpu()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=running_loss / (i + 1))

        # Validation
        model.eval()
        gt = torch.FloatTensor().cpu()
        pred = torch.FloatTensor().cpu()
        valid_loader_tqdm = tqdm(valid_loader, desc=f"Validation Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch")
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(valid_loader_tqdm):
                targets = targets.cpu()
                gt = torch.cat((gt, targets), 0)
                bs, c, h, w = inputs.size()
                inputs = inputs.cpu()
                outputs = model(inputs)
                pred = torch.cat((pred, outputs.data), 0)

        AUROCs = compute_AUCs(gt, pred)
        AUROC_avg = np.array(AUROCs).mean()
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Validation AUROC: {AUROC_avg:.3f}')

        # Save the best model
        if AUROC_avg > best_auc:
            best_auc = AUROC_avg
            torch.save({'state_dict': model.state_dict()}, CKPT_PATH)

    print('Finished Training')

    # Testing
    model.eval()
    gt = torch.FloatTensor().cpu()
    pred = torch.FloatTensor().cpu()
    test_loader_tqdm = tqdm(test_loader, desc="Testing", unit="batch")
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader_tqdm):
            targets = targets.cpu()
            gt = torch.cat((gt, targets), 0)
            bs, n_crops, c, h, w = inputs.size()
            inputs = inputs.view(-1, c, h, w).cpu()
            outputs = model(inputs)
            outputs_mean = outputs.view(bs, n_crops, -1).mean(1)
            pred = torch.cat((pred, outputs_mean.data), 0)

    AUROCs = compute_AUCs(gt, pred)
    AUROC_avg = np.array(AUROCs).mean()
    print(f'The average AUROC is {AUROC_avg:.3f}')
    for i in range(N_CLASSES):
        print(f'The AUROC of {CLASS_NAMES[i]} is {AUROCs[i]}')


def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


if __name__ == '__main__':
    main()
