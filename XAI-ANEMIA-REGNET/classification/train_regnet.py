
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import seaborn as sns
import matplotlib.pyplot as plt

from regnet import RegNetAnemiaClassifier
from focal_loss import FocalLoss
from preprocess.augment import get_transforms
import evaluation.metrics as metrics


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AnemiaDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = plt.imread(img_path)
        if image.shape[-1] == 4: 
            image = image[..., :3]
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image
        image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)

        label = int(row['label']) 
        return image, label


def train_one_epoch(model, loader, criterion, optimizer, device, writer, epoch):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [TRAIN]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader)
    acc, prec, rec, f1 = metrics.calculate_metrics(np.array(all_labels), np.array(all_preds))
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/train', acc, epoch)
    return epoch_loss, acc, prec, rec, f1


@torch.no_grad()
def validate(model, loader, criterion, device, writer, epoch):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [VAL]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader)
    acc, prec, rec, f1 = metrics.calculate_metrics(np.array(all_labels), np.array(all_preds))
    writer.add_scalar('Loss/val', epoch_loss, epoch)
    writer.add_scalar('Accuracy/val', acc, epoch)
    return epoch_loss, acc, prec, rec, f1, np.array(all_labels), np.array(all_preds)


def save_confusion_matrix(y_true, y_pred, epoch, save_dir="evaluation"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Anemia', 'Anemia'], yticklabels=['No Anemia', 'Anemia'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    path = os.path.join(save_dir, f"confusion_matrix_epoch_{epoch}.png")
    plt.savefig(path)
    plt.close()
    print(f"Matriz de confusión guardada: {path}")


def main(args):
    set_seed(42)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "augmented")
    splits_path = os.path.join(project_root, "data", "splits.csv")
    model_save_path = os.path.join(project_root, "classification", "best_regnet_anemia_classifier.pth")
    log_dir = os.path.join(project_root, "runs", "regnet")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("evaluation", exist_ok=True)

    print("Cargando splits de datos...")
    splits_df = pd.read_csv(splits_path)
    train_df = splits_df[splits_df['split'] == 'train']
    val_df = splits_df[splits_df['split'] == 'val']

    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)

    train_dataset = AnemiaDataset(train_df, data_dir, transform=train_transform)
    val_dataset = AnemiaDataset(val_df, data_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    model = RegNetAnemiaClassifier(num_classes=1).to(device)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    
    writer = SummaryWriter(log_dir)

    best_f1 = 0.0
    patience = 7
    patience_counter = 0

    print("Iniciando entrenamiento...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, writer, epoch
        )
        val_loss, val_acc, val_prec, val_rec, val_f1, y_true, y_pred = validate(
            model, val_loader, criterion, device, writer, epoch
        )

        scheduler.step()

        print(f"\nÉpoca {epoch}/{args.epochs}")
        print(f"TRAIN → Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"VAL   → Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} (P: {val_prec:.4f}, R: {val_rec:.4f})")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)
            save_confusion_matrix(y_true, y_pred, epoch)
            print(f"Nuevo mejor modelo guardado! F1: {best_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping en época {epoch}")
            break

    writer.close()
    print(f"\nEntrenamiento completado. Mejor F1 en validación: {best_f1:.4f}")
    print(f"Modelo guardado en: {model_save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Entrenar RegNet para clasificación de anemia")
    parser.add_argument('--epochs', type=int, default=50, help='Número de épocas')
    parser.add_argument('--batch_size', type=int, default=32, help='Tamaño del batch')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default=None, help='Ruta a datos aumentados (opcional)')

    args = parser.parse_args()
    main(args)