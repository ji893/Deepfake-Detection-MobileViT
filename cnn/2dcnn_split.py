import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 중인 디바이스: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 버전: {torch.version.cuda}")

# 설정값
CONFIG = {
    'image_size': 224,
    'epochs': 10,
    'batch_size': 16,
    'learning_rate': 1e-4,
    'num_workers': 4 if torch.cuda.is_available() else 0,
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
    'k_fold': 5,  # K-Fold 추가
}

# 데이터 경로 설정
DATA_DIR = Path(".")
REAL_DIR = DATA_DIR / "Real_1500"
DDIM_DIR = DATA_DIR / "DDIM_sample_500"
DIFFSWAP_DIR = DATA_DIR / "diff_500"
ADM_DIR = DATA_DIR / "processed_data_500"


class DeepfakeDataset(Dataset):
    """딥페이크 이미지 데이터셋"""
    def __init__(self, real_paths, fake_paths, transform=None):
        self.image_paths = []
        self.labels = []
        
        # Real 이미지 (label=0)
        for path in real_paths:
            self.image_paths.append(path)
            self.labels.append(0)
        
        # Fake 이미지 (label=1)
        for path in fake_paths:
            self.image_paths.append(path)
            self.labels.append(1)
        
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 이미지 로드
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class CNN2D(nn.Module):
    """2D CNN 모델 (이진 분류용 - 출력 1개 노드)"""
    def __init__(self, num_classes=1):  # 이진 분류이므로 1개 노드
        super(CNN2D, self).__init__()
        
        # 2D Convolution layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers (이진 분류용)
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)  # 출력 1개 노드
    
    def forward(self, x):
        # x shape: (batch, 3, 224, 224)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def get_image_paths(directory):
    """디렉토리에서 이미지 파일 경로 수집 (하위 디렉토리 포함)"""
    if not directory.exists():
        print(f"경고: 디렉토리가 존재하지 않습니다: {directory}")
        return []
    
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
    paths = []
    
    # 하위 디렉토리까지 재귀적으로 검색 (rglob 사용)
    for ext in extensions:
        paths.extend(list(directory.rglob(f'*{ext}')))
    
    print(f"디버그: {directory.name}에서 {len(paths)}개의 이미지를 찾았습니다.")
    return sorted(paths)  # 정렬하여 일관성 유지


def prepare_datasets(real_count=1500, ddim_count=500, diffswap_count=500, adm_count=500, random_state=42):
    """데이터셋 준비"""
    random.seed(random_state)
    np.random.seed(random_state)
    
    all_real_paths = get_image_paths(REAL_DIR)
    all_ddim_paths = get_image_paths(DDIM_DIR)
    all_diffswap_paths = get_image_paths(DIFFSWAP_DIR)
    all_adm_paths = get_image_paths(ADM_DIR)
    
    # 랜덤 샘플링
    real_paths = random.sample(all_real_paths, min(real_count, len(all_real_paths))) if len(all_real_paths) > real_count else all_real_paths
    ddim_paths = random.sample(all_ddim_paths, min(ddim_count, len(all_ddim_paths))) if len(all_ddim_paths) > ddim_count else all_ddim_paths
    diffswap_paths = random.sample(all_diffswap_paths, min(diffswap_count, len(all_diffswap_paths))) if len(all_diffswap_paths) > diffswap_count else all_diffswap_paths
    adm_paths = random.sample(all_adm_paths, min(adm_count, len(all_adm_paths))) if len(all_adm_paths) > adm_count else all_adm_paths
    
    return {
        'real': real_paths,
        'ddim': ddim_paths,
        'diffswap': diffswap_paths,
        'adm': adm_paths
    }


def split_source_data(source_paths, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """각 Source별로 8:1:1 분할"""
    # 먼저 train과 temp로 분할 (train: 80%, temp: 20%)
    train_paths, temp_paths = train_test_split(
        source_paths, test_size=(val_ratio + test_ratio), random_state=random_state
    )
    
    # temp를 val과 test로 분할 (val: 10%, test: 10%)
    val_paths, test_paths = train_test_split(
        temp_paths, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=random_state
    )
    
    return train_paths, val_paths, test_paths


def split_all_sources(datasets, dataset_size=None, random_state=42):
    """모든 Source를 각각 8:1:1로 분할"""
    random.seed(random_state)
    np.random.seed(random_state)
    
    # 데이터셋 크기에 맞게 조정
    if dataset_size is not None:
        real_size = dataset_size // 2
        fake_size = dataset_size // 2
        each_fake_size = fake_size // 3
        
        # 랜덤 샘플링
        real_paths = random.sample(datasets['real'], min(real_size, len(datasets['real']))) if len(datasets['real']) > real_size else datasets['real']
        ddim_paths = random.sample(datasets['ddim'], min(each_fake_size, len(datasets['ddim']))) if len(datasets['ddim']) > each_fake_size else datasets['ddim']
        diffswap_paths = random.sample(datasets['diffswap'], min(each_fake_size, len(datasets['diffswap']))) if len(datasets['diffswap']) > each_fake_size else datasets['diffswap']
        adm_paths = random.sample(datasets['adm'], min(each_fake_size, len(datasets['adm']))) if len(datasets['adm']) > each_fake_size else datasets['adm']
        
        # 부족한 경우 보충
        remaining = fake_size - (len(ddim_paths) + len(diffswap_paths) + len(adm_paths))
        if remaining > 0 and len(datasets['ddim']) > len(ddim_paths):
            additional_needed = min(remaining, len(datasets['ddim']) - len(ddim_paths))
            remaining_ddim = [p for p in datasets['ddim'] if p not in ddim_paths]
            if remaining_ddim:
                ddim_paths.extend(random.sample(remaining_ddim, min(additional_needed, len(remaining_ddim))))
    else:
        real_paths = datasets['real']
        ddim_paths = datasets['ddim']
        diffswap_paths = datasets['diffswap']
        adm_paths = datasets['adm']
    
    # 각 Source별로 8:1:1 분할
    real_train, real_val, real_test = split_source_data(real_paths)
    ddim_train, ddim_val, ddim_test = split_source_data(ddim_paths)
    diffswap_train, diffswap_val, diffswap_test = split_source_data(diffswap_paths)
    adm_train, adm_val, adm_test = split_source_data(adm_paths)
    
    # 학습 세트: 모든 Source 합침
    train_real = real_train
    train_fake = ddim_train + diffswap_train + adm_train
    
    # 검증 세트: 모든 Source 합침
    val_real = real_val
    val_fake = ddim_val + diffswap_val + adm_val
    
    # 테스트 세트: Source별로 분리 (Source-wise 평가용)
    test_splits = {
        'real': real_test,
        'ddim': ddim_test,
        'diffswap': diffswap_test,
        'adm': adm_test
    }
    
    # Overall 테스트 세트: 모든 Source 합침
    test_real_all = real_test
    test_fake_all = ddim_test + diffswap_test + adm_test
    
    print(f"\n데이터 분할 완료:")
    print(f"  학습: Real {len(train_real)}장, Fake {len(train_fake)}장 (총 {len(train_real) + len(train_fake)}장)")
    print(f"  검증: Real {len(val_real)}장, Fake {len(val_fake)}장 (총 {len(val_real) + len(val_fake)}장)")
    print(f"  테스트 (Source-wise):")
    print(f"    Real: {len(test_splits['real'])}장")
    print(f"    DDIM: {len(test_splits['ddim'])}장")
    print(f"    DiffSwap: {len(test_splits['diffswap'])}장")
    print(f"    ADM: {len(test_splits['adm'])}장")
    print(f"  테스트 (Overall): Real {len(test_real_all)}장, Fake {len(test_fake_all)}장 (총 {len(test_real_all) + len(test_fake_all)}장)")
    
    return {
        'train': {'real': train_real, 'fake': train_fake},
        'val': {'real': val_real, 'fake': val_fake},
        'test_splits': test_splits,  # Source-wise 평가용
        'test_all': {'real': test_real_all, 'fake': test_fake_all}  # Overall 평가용
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """한 에포크 학습"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)  # BCEWithLogitsLoss를 위해 float로 변환 및 차원 추가
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        # sigmoid를 적용하여 확률로 변환 후 0.5 기준으로 예측
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """모델 평가"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []  # ROC-AUC 계산용
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels_np = labels.numpy()  # 원본 labels 저장
            labels = labels.to(device).float().unsqueeze(1)  # BCEWithLogitsLoss를 위해 float로 변환 및 차원 추가
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            # sigmoid를 적용하여 확률로 변환
            probs = torch.sigmoid(outputs).cpu().numpy()
            # 0.5 기준으로 예측
            predicted = (probs > 0.5).astype(int)
            
            all_preds.extend(predicted.flatten().astype(int))
            all_labels.extend(labels_np)
            all_probs.extend(probs.flatten())
    
    epoch_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # ROC-AUC 계산
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc_auc = 0.0  # 클래스가 하나만 있는 경우
    
    return epoch_loss, accuracy, precision, recall, f1, roc_auc, all_preds, all_labels


def evaluate_source_wise(model, test_splits, transform, device):
    """Source-wise 성능 평가 (각 Source의 테스트 세트로 개별 평가)"""
    results = {}
    
    real_test = test_splits['real']
    
    sources = {
        'Diffusion': test_splits['ddim'],  # DDIM을 Diffusion으로 간주
        'DDIM': test_splits['ddim'],
        'DiffSwap': test_splits['diffswap'],
        'ADM': test_splits['adm'],
    }
    
    for source_name, fake_test in sources.items():
        # Real과 해당 Source의 테스트 세트로 평가
        test_dataset = DeepfakeDataset(real_test, fake_test, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], 
                                shuffle=False, num_workers=CONFIG['num_workers'])
        
        criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss
        _, accuracy, precision, recall, f1, roc_auc, _, _ = evaluate(
            model, test_loader, criterion, device
        )
        
        results[f'Real vs {source_name}'] = {
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100,
            'roc_auc': roc_auc * 100
        }
    
    return results


def train_model_kfold(data_splits, dataset_size):
    """K-Fold 교차 검증을 사용한 모델 학습 및 평가"""
    print(f"\n{'='*60}")
    print(f"데이터셋 크기: {dataset_size}장으로 K-Fold {CONFIG['k_fold']} 교차 검증 시작")
    print(f"{'='*60}\n")
    
    # Train 데이터 준비 (K-Fold용)
    train_real = data_splits['train']['real']
    train_fake = data_splits['train']['fake']
    
    # Train 데이터를 하나의 리스트로 합침 (K-Fold용)
    train_all_paths = train_real + train_fake
    train_all_labels = [0] * len(train_real) + [1] * len(train_fake)
    
    # K-Fold 설정
    kfold = KFold(n_splits=CONFIG['k_fold'], shuffle=True, random_state=42)
    
    # Data Augmentation (최소화)
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # K-Fold 결과 저장
    fold_results = []
    all_fold_val_preds = []
    all_fold_val_labels = []
    
    # Train 데이터에 K-Fold 적용
    for fold, (train_idx, fold_val_idx) in enumerate(kfold.split(train_all_paths)):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{CONFIG['k_fold']}")
        print(f"{'='*60}\n")
        
        # Fold별 train/val 분할
        fold_train_paths = [train_all_paths[i] for i in train_idx]
        fold_train_labels = [train_all_labels[i] for i in train_idx]
        fold_val_paths = [train_all_paths[i] for i in fold_val_idx]
        fold_val_labels = [train_all_labels[i] for i in fold_val_idx]
        
        # Real과 Fake로 분리
        fold_train_real = [p for p, l in zip(fold_train_paths, fold_train_labels) if l == 0]
        fold_train_fake = [p for p, l in zip(fold_train_paths, fold_train_labels) if l == 1]
        fold_val_real = [p for p, l in zip(fold_val_paths, fold_val_labels) if l == 0]
        fold_val_fake = [p for p, l in zip(fold_val_paths, fold_val_labels) if l == 1]
        
        print(f"Fold {fold + 1} 데이터 분할:")
        print(f"  학습: Real {len(fold_train_real)}장, Fake {len(fold_train_fake)}장 (총 {len(fold_train_paths)}장)")
        print(f"  검증: Real {len(fold_val_real)}장, Fake {len(fold_val_fake)}장 (총 {len(fold_val_paths)}장)")
        
        # 데이터셋 및 데이터로더 생성
        fold_train_dataset = DeepfakeDataset(fold_train_real, fold_train_fake, transform=train_transform)
        fold_val_dataset = DeepfakeDataset(fold_val_real, fold_val_fake, transform=val_transform)
        
        fold_train_loader = DataLoader(fold_train_dataset, batch_size=CONFIG['batch_size'], 
                                     shuffle=True, num_workers=CONFIG['num_workers'])
        fold_val_loader = DataLoader(fold_val_dataset, batch_size=CONFIG['batch_size'], 
                                   shuffle=False, num_workers=CONFIG['num_workers'])
        
        # 모델 초기화
        model = CNN2D(num_classes=1).to(device)  # 이진 분류
        criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
        
        # 학습
        best_val_acc = 0.0
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        
        for epoch in range(CONFIG['epochs']):
            print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
            
            train_loss, train_acc = train_epoch(model, fold_train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_prec, val_rec, val_f1, val_roc_auc, _, _ = evaluate(
                model, fold_val_loader, criterion, device
            )
            
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc * 100)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
            print(f"Val Precision: {val_prec*100:.2f}%, Val Recall: {val_rec*100:.2f}%, Val F1: {val_f1*100:.2f}%, Val ROC-AUC: {val_roc_auc*100:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                print(f"✓ Best model updated! (Val Acc: {best_val_acc*100:.2f}%)")
        
        # Best model 로드
        model.load_state_dict(best_model_state)
        
        # 검증 세트 최종 평가
        val_loss, val_acc, val_prec, val_rec, val_f1, val_roc_auc, val_preds, val_labels = evaluate(
            model, fold_val_loader, criterion, device
        )
        
        fold_results.append({
            'fold': fold + 1,
            'val_accuracy': val_acc * 100,
            'val_precision': val_prec * 100,
            'val_recall': val_rec * 100,
            'val_f1': val_f1 * 100,
            'val_roc_auc': val_roc_auc * 100,
            'train_history': {
                'loss': train_losses,
                'acc': train_accs,
                'val_loss': val_losses,
                'val_acc': val_accs
            }
        })
        
        # 모든 fold의 검증 예측 수집
        all_fold_val_preds.extend(val_preds)
        all_fold_val_labels.extend(val_labels)
        
        print(f"\nFold {fold + 1} 완료 - Val Acc: {val_acc*100:.2f}%")
    
    # K-Fold 평균 결과 계산
    avg_val_acc = np.mean([r['val_accuracy'] for r in fold_results])
    avg_val_prec = np.mean([r['val_precision'] for r in fold_results])
    avg_val_rec = np.mean([r['val_recall'] for r in fold_results])
    avg_val_f1 = np.mean([r['val_f1'] for r in fold_results])
    avg_val_roc_auc = np.mean([r['val_roc_auc'] for r in fold_results])
    
    print(f"\n{'='*60}")
    print(f"K-Fold {CONFIG['k_fold']} 평균 결과 (Train 데이터 기준)")
    print(f"{'='*60}")
    print(f"평균 Val Accuracy: {avg_val_acc:.2f}%")
    print(f"평균 Val Precision: {avg_val_prec:.2f}%")
    print(f"평균 Val Recall: {avg_val_rec:.2f}%")
    print(f"평균 Val F1: {avg_val_f1:.2f}%")
    print(f"평균 Val ROC-AUC: {avg_val_roc_auc:.2f}%")
    
    # 원래 분리한 Val 세트 평가 (마지막 fold의 best model 사용)
    print(f"\n{'='*60}")
    print("원래 분리한 Val 세트 평가")
    print(f"{'='*60}\n")
    
    val_dataset = DeepfakeDataset(data_splits['val']['real'], data_splits['val']['fake'], transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                           shuffle=False, num_workers=CONFIG['num_workers'])
    
    # 마지막 fold의 best model 사용
    model = CNN2D(num_classes=1).to(device)
    model.load_state_dict(best_model_state)
    
    val_loss, val_acc, val_prec, val_rec, val_f1, val_roc_auc, _, _ = evaluate(
        model, val_loader, criterion, device
    )
    
    print(f"Val Set Accuracy: {val_acc*100:.2f}%")
    print(f"Val Set Precision: {val_prec*100:.2f}%")
    print(f"Val Set Recall: {val_rec*100:.2f}%")
    print(f"Val Set F1: {val_f1*100:.2f}%")
    print(f"Val Set ROC-AUC: {val_roc_auc*100:.2f}%")
    
    # 테스트 세트 평가
    print(f"\n{'='*60}")
    print("테스트 세트 평가")
    print(f"{'='*60}\n")
    
    # 1. Overall 평가 (모든 Source의 테스트 세트 합쳐서)
    print("1. Overall 평가 (모든 Source의 테스트 세트 합침)")
    print("-" * 60)
    test_all_dataset = DeepfakeDataset(
        data_splits['test_all']['real'], 
        data_splits['test_all']['fake'], 
        transform=test_transform
    )
    test_all_loader = DataLoader(test_all_dataset, batch_size=CONFIG['batch_size'], 
                                shuffle=False, num_workers=CONFIG['num_workers'])
    
    test_all_loss, test_all_acc, test_all_prec, test_all_rec, test_all_f1, test_all_roc_auc, test_all_preds, test_all_labels = evaluate(
        model, test_all_loader, criterion, device
    )
    
    print(f"Test Overall Loss: {test_all_loss:.4f}")
    print(f"Test Overall Accuracy: {test_all_acc*100:.2f}%")
    print(f"Test Overall Precision: {test_all_prec*100:.2f}%")
    print(f"Test Overall Recall: {test_all_rec*100:.2f}%")
    print(f"Test Overall F1: {test_all_f1*100:.2f}%")
    print(f"Test Overall ROC-AUC: {test_all_roc_auc*100:.2f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(test_all_labels, test_all_preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    print(f"\nConfusion Matrix (Overall):")
    print(f"                Predicted")
    print(f"              Real  Fake")
    print(f"Actual Real    {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"       Fake    {cm[1][0]:4d}  {cm[1][1]:4d}")
    print(f"\nTP (True Positive): {tp}, FP (False Positive): {fp}")
    print(f"TN (True Negative): {tn}, FN (False Negative): {fn}")
    
    # 2. Source-wise 평가 (각 Source의 테스트 세트로 개별 평가)
    print(f"\n{'='*60}")
    print("2. Source-wise 성능 평가 (각 Source의 테스트 세트로 개별 평가)")
    print(f"{'='*60}\n")
    
    source_results = evaluate_source_wise(model, data_splits['test_splits'], test_transform, device)
    
    # 결과 출력
    print(f"{'Source':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC-AUC':<12}")
    print("-" * 72)
    
    for source, metrics in source_results.items():
        print(f"{source:<20} {metrics['accuracy']:>10.2f}%  {metrics['precision']:>10.2f}%  "
              f"{metrics['recall']:>10.2f}%  {metrics['f1']:>10.2f}%  {metrics['roc_auc']:>10.2f}%")
    
    return {
        'kfold_avg': {
            'val_accuracy': avg_val_acc,
            'val_precision': avg_val_prec,
            'val_recall': avg_val_rec,
            'val_f1': avg_val_f1,
            'val_roc_auc': avg_val_roc_auc
        },
        'val_set': {
            'accuracy': val_acc * 100,
            'precision': val_prec * 100,
            'recall': val_rec * 100,
            'f1': val_f1 * 100,
            'roc_auc': val_roc_auc * 100
        },
        'test_overall': {
            'loss': test_all_loss,
            'accuracy': test_all_acc * 100,
            'precision': test_all_prec * 100,
            'recall': test_all_rec * 100,
            'f1': test_all_f1 * 100,
            'roc_auc': test_all_roc_auc * 100
        },
        'source_wise': source_results,
        'fold_results': fold_results
    }


def train_model(datasets, dataset_size):
    """모델 학습 및 평가 (K-Fold + 각 Source별 8:1:1 분할)"""
    print(f"\n{'='*60}")
    print(f"데이터셋 크기: {dataset_size}장으로 학습 시작")
    print(f"{'='*60}\n")
    
    # 각 Source별로 8:1:1 분할
    data_splits = split_all_sources(datasets, dataset_size, random_state=42)
    
    # K-Fold 교차 검증으로 학습
    results = train_model_kfold(data_splits, dataset_size)
    
    return results


def main():
    """메인 함수"""
    print("="*60)
    print("2D CNN 딥페이크 탐지 학습 (K-Fold 5 교차 검증 + 각 Source별 8:1:1 분할)")
    print("="*60)
    
    # 데이터셋 준비
    print("\n데이터셋 로딩 중...")
    datasets = prepare_datasets()
    
    print(f"Real: {len(datasets['real'])}장")
    print(f"DDIM: {len(datasets['ddim'])}장")
    print(f"DiffSwap: {len(datasets['diffswap'])}장")
    print(f"ADM: {len(datasets['adm'])}장")
    
    # 데이터셋 크기별 학습
    dataset_sizes = [500, 1500, 3000]
    all_results = {}
    
    for size in dataset_sizes:
        if size > len(datasets['real']) * 2:
            print(f"\n경고: {size}장은 데이터 부족으로 건너뜁니다.")
            continue
        
        results = train_model(datasets, size)
        all_results[f'{size}장'] = results
        
        print(f"\n{'='*60}")
        print(f"{size}장 데이터셋 결과 요약")
        print(f"{'='*60}")
        print(f"K-Fold 평균 Val 정확도: {results['kfold_avg']['val_accuracy']:.2f}%")
        print(f"테스트 Overall 정확도: {results['test_overall']['accuracy']:.2f}%")
        print(f"테스트 Overall F1: {results['test_overall']['f1']:.2f}%")
    
    # 결과를 DataFrame으로 정리
    result_data = []
    for size, results in all_results.items():
        # K-Fold 평균 결과
        result_data.append({
            '데이터셋 크기': size,
            'Source': 'K-Fold 평균 (Val)',
            'Accuracy': f"{results['kfold_avg']['val_accuracy']:.2f}%",
            'Precision': f"{results['kfold_avg']['val_precision']:.2f}%",
            'Recall': f"{results['kfold_avg']['val_recall']:.2f}%",
            'F1': f"{results['kfold_avg']['val_f1']:.2f}%",
            'ROC-AUC': f"{results['kfold_avg']['val_roc_auc']:.2f}%"
        })
        
        # Val Set 결과
        result_data.append({
            '데이터셋 크기': size,
            'Source': 'Val Set',
            'Accuracy': f"{results['val_set']['accuracy']:.2f}%",
            'Precision': f"{results['val_set']['precision']:.2f}%",
            'Recall': f"{results['val_set']['recall']:.2f}%",
            'F1': f"{results['val_set']['f1']:.2f}%",
            'ROC-AUC': f"{results['val_set']['roc_auc']:.2f}%"
        })
        
        # Overall 테스트 세트 결과
        result_data.append({
            '데이터셋 크기': size,
            'Source': 'Test Set (Overall)',
            'Accuracy': f"{results['test_overall']['accuracy']:.2f}%",
            'Precision': f"{results['test_overall']['precision']:.2f}%",
            'Recall': f"{results['test_overall']['recall']:.2f}%",
            'F1': f"{results['test_overall']['f1']:.2f}%",
            'ROC-AUC': f"{results['test_overall']['roc_auc']:.2f}%"
        })
        
        # Source-wise 결과
        for source, metrics in results['source_wise'].items():
            result_data.append({
                '데이터셋 크기': size,
                'Source': source,
                'Accuracy': f"{metrics['accuracy']:.2f}%",
                'Precision': f"{metrics['precision']:.2f}%",
                'Recall': f"{metrics['recall']:.2f}%",
                'F1': f"{metrics['f1']:.2f}%",
                'ROC-AUC': f"{metrics['roc_auc']:.2f}%"
            })
    
    df = pd.DataFrame(result_data)
    
    # 결과 저장
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "2dcnn_kfold_source_wise_results_split.csv", index=False, encoding='utf-8-sig')
    
    # 시각화
    plot_results(all_results)
    
    print(f"\n결과가 {output_dir}에 저장되었습니다.")


def plot_results(all_results):
    """결과 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('2D CNN: K-Fold Cross-Validation + Source-wise Performance Comparison', fontsize=16, fontweight='bold')
    
    sources = ['Real vs Diffusion', 'Real vs DDIM', 'Real vs ADM', 'Overall']
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]
        
        x = np.arange(len(sources))
        width = 0.25
        dataset_sizes = list(all_results.keys())
        
        for i, size in enumerate(dataset_sizes):
            values = []
            for source in sources:
                if source == 'Overall':
                    # Overall은 test_overall에서 가져옴
                    values.append(all_results[size]['test_overall'][metric])
                else:
                    # Source-wise는 source_wise에서 가져옴
                    source_key = f'Real vs {source.split(" vs ")[1]}'
                    if source_key in all_results[size]['source_wise']:
                        values.append(all_results[size]['source_wise'][source_key][metric])
                    else:
                        values.append(0)
            
            ax.bar(x + i * width, values, width, label=size)
        
        ax.set_xlabel('Source')
        ax.set_ylabel(f'{metric_name} (%)')
        ax.set_title(f'{metric_name} by Dataset Size')
        ax.set_xticks(x + width)
        ax.set_xticklabels(sources, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig('results/2dcnn_kfold_source_wise_comparison_split.png', dpi=300, bbox_inches='tight')
    print("\n시각화 결과가 저장되었습니다: results/2dcnn_kfold_source_wise_comparison_split.png")


if __name__ == "__main__":
    main()






