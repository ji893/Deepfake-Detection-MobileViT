import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random
warnings.filterwarnings('ignore')

# 모든 랜덤 시드 고정 (재현성 보장)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # 멀티 GPU 사용 시
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# GPU 설정 (메인 스크립트에서만 출력)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 설정값 (Re 파일 기준)
CONFIG = {
    'image_size': 224,
    'epochs': 10,
    'batch_size': 16,
    'learning_rate': 1e-4,
    'num_workers': 4 if torch.cuda.is_available() else 0,
    'k_folds': 5,  # K-Fold Cross Validation
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


def get_image_paths(directory):
    """디렉토리에서 이미지 파일 경로 수집 (하위 디렉토리 포함)"""
    if not directory.exists():
        print(f"경고: 디렉토리가 존재하지 않습니다: {directory}")
        return []
    
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
    paths = []
    
    for ext in extensions:
        paths.extend(list(directory.rglob(f'*{ext}')))
    
    print(f"디버그: {directory.name}에서 {len(paths)}개의 이미지를 찾았습니다.")
    return sorted(paths)


def prepare_datasets(real_count=1500, ddim_count=500, diffswap_count=500, adm_count=500, random_state=42):
    """데이터셋 준비 (Source 정보 포함) - 랜덤 샘플링 사용"""
    random.seed(random_state)
    np.random.seed(random_state)
    
    # 전체 이미지 경로 가져오기
    all_real_paths = get_image_paths(REAL_DIR)
    all_ddim_paths = get_image_paths(DDIM_DIR)
    all_diffswap_paths = get_image_paths(DIFFSWAP_DIR)
    all_adm_paths = get_image_paths(ADM_DIR)
    
    # 랜덤 샘플링 (재현 가능)
    real_paths = random.sample(all_real_paths, min(real_count, len(all_real_paths))) if len(all_real_paths) > real_count else all_real_paths
    ddim_paths = random.sample(all_ddim_paths, min(ddim_count, len(all_ddim_paths))) if len(all_ddim_paths) > ddim_count else all_ddim_paths
    diffswap_paths = random.sample(all_diffswap_paths, min(diffswap_count, len(all_diffswap_paths))) if len(all_diffswap_paths) > diffswap_count else all_diffswap_paths
    adm_paths = random.sample(all_adm_paths, min(adm_count, len(all_adm_paths))) if len(all_adm_paths) > adm_count else all_adm_paths
    
    all_paths = []
    all_labels = []
    all_sources = []
    
    # Real 이미지 (label=0)
    for path in real_paths:
        all_paths.append(path)
        all_labels.append(0)
        all_sources.append('Real')
    
    # Fake 이미지 (label=1)
    for path in ddim_paths:
        all_paths.append(path)
        all_labels.append(1)
        all_sources.append('DDIM')
    
    for path in diffswap_paths:
        all_paths.append(path)
        all_labels.append(1)
        all_sources.append('DiffSwap')
    
    for path in adm_paths:
        all_paths.append(path)
        all_labels.append(1)
        all_sources.append('ADM')
    
    return {
        'paths': all_paths,
        'labels': all_labels,
        'sources': all_sources,
        'real': real_paths,
        'ddim': ddim_paths,
        'diffswap': diffswap_paths,
        'adm': adm_paths
    }


def create_dataset_splits(datasets, total_size, random_state=42):
    """데이터셋 크기에 맞게 랜덤 샘플링"""
    random.seed(random_state)
    np.random.seed(random_state)
    
    # Real은 항상 절반
    real_size = total_size // 2
    fake_size = total_size // 2
    
    # Fake 데이터를 균등 분배
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
    
    all_paths = []
    all_labels = []
    all_sources = []
    
    for path in real_paths:
        all_paths.append(path)
        all_labels.append(0)
        all_sources.append('Real')
    
    for path in ddim_paths:
        all_paths.append(path)
        all_labels.append(1)
        all_sources.append('DDIM')
    
    for path in diffswap_paths:
        all_paths.append(path)
        all_labels.append(1)
        all_sources.append('DiffSwap')
    
    for path in adm_paths:
        all_paths.append(path)
        all_labels.append(1)
        all_sources.append('ADM')
    
    return all_paths, all_labels, all_sources


class CNN2D(nn.Module):
    """2D CNN 모델 (베이스라인)"""
    def __init__(self):
        super(CNN2D, self).__init__()
        
        # Convolutional layers
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
        
        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)  # Binary classification
    
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


def train_epoch(model, dataloader, criterion, optimizer, device):
    """한 에포크 학습"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training", leave=False, ncols=80)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # (batch, 1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # 예측값 계산 (sigmoid 적용)
        preds = torch.sigmoid(outputs) > 0.5
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, accuracy * 100


def evaluate(model, dataloader, criterion, device):
    """모델 평가"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False, ncols=80)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # 예측값 계산
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return epoch_loss, accuracy, precision, recall, f1, all_preds, all_labels


def evaluate_source_wise(model, test_paths, test_labels, test_sources, transform, device):
    """Source-wise 성능 평가"""
    results = {}
    
    real_indices = [i for i, s in enumerate(test_sources) if s == 'Real']
    real_paths = [test_paths[i] for i in real_indices]
    
    sources = {
        'Diffusion': 'DDIM',
        'DDIM': 'DDIM',
        'DiffSwap': 'DiffSwap',
        'ADM': 'ADM',
    }
    
    for source_name, source_key in sources.items():
        source_indices = [i for i, s in enumerate(test_sources) if s == source_key]
        
        if not source_indices:
            continue
            
        source_paths = [test_paths[i] for i in source_indices]
        
        combined_paths = real_paths + source_paths
        combined_labels = [0] * len(real_paths) + [1] * len(source_paths)
        
        test_dataset = DeepfakeDataset(combined_paths, combined_labels, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], 
                                shuffle=False, num_workers=CONFIG['num_workers'])
        
        criterion = nn.BCEWithLogitsLoss()
        _, accuracy, precision, recall, f1, _, _ = evaluate(
            model, test_loader, criterion, device
        )
        
        results[f'Real vs {source_name}'] = {
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100
        }
    
    return results


def train_kfold(all_paths, all_labels, all_sources, dataset_size):
    """K-Fold Cross Validation으로 학습"""
    print(f"\n{'='*60}")
    print(f"데이터셋 크기: {dataset_size}장으로 K-Fold 학습 시작 (K={CONFIG['k_folds']})")
    print(f"{'='*60}\n")
    
    # Data Augmentation (최소화 - Flip, Rotation만)
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
    
    # K-Fold 설정
    skf = StratifiedKFold(n_splits=CONFIG['k_folds'], shuffle=True, random_state=42)
    
    fold_results = []
    all_fold_histories = []
    
    # NumPy 배열로 변환
    X = np.array(all_paths)
    y = np.array(all_labels)
    sources = np.array(all_sources)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{CONFIG['k_folds']}")
        print(f"{'='*60}\n")
        
        # 데이터 분할
        train_paths = X[train_idx].tolist()
        train_labels = y[train_idx].tolist()
        val_paths = X[val_idx].tolist()
        val_labels = y[val_idx].tolist()
        val_sources = sources[val_idx].tolist()
        
        print(f"학습: {len(train_paths)}장 (Real: {train_labels.count(0)}, Fake: {train_labels.count(1)})")
        print(f"검증: {len(val_paths)}장 (Real: {val_labels.count(0)}, Fake: {val_labels.count(1)})")
        
        # 데이터셋 및 데이터로더 생성
        train_dataset = DeepfakeDataset(train_paths, train_labels, transform=train_transform)
        val_dataset = DeepfakeDataset(val_paths, val_labels, transform=val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                                 shuffle=True, num_workers=CONFIG['num_workers'])
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                               shuffle=False, num_workers=CONFIG['num_workers'])
        
        # 모델 초기화
        model = CNN2D().to(device)
        criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
        
        # 학습
        best_val_acc = 0.0
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        
        for epoch in range(CONFIG['epochs']):
            print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
            
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(
                model, val_loader, criterion, device
            )
            
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc * 100)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
            print(f"Val Precision: {val_prec*100:.2f}%, Val Recall: {val_rec*100:.2f}%, Val F1: {val_f1*100:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                print(f"✓ Best model updated! (Val Acc: {best_val_acc*100:.2f}%)")
        
        # Best model 로드
        model.load_state_dict(best_model_state)
        
        # 검증 세트로 최종 평가
        val_loss, val_acc, val_prec, val_rec, val_f1, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )
        
        print(f"\nFold {fold + 1} 최종 검증 정확도: {val_acc*100:.2f}%")
        
        # Source-wise 평가
        source_results = evaluate_source_wise(
            model, val_paths, val_labels, val_sources, val_transform, device
        )
        
        fold_results.append({
            'fold': fold + 1,
            'val_accuracy': val_acc * 100,
            'val_precision': val_prec * 100,
            'val_recall': val_rec * 100,
            'val_f1': val_f1 * 100,
            'source_wise': source_results
        })
        
        all_fold_histories.append({
            'train_loss': train_losses,
            'train_acc': train_accs,
            'val_loss': val_losses,
            'val_acc': val_accs
        })
    
    # K-Fold 결과 요약
    print(f"\n{'='*60}")
    print(f"K-Fold Cross Validation 결과 요약")
    print(f"{'='*60}\n")
    
    avg_acc = np.mean([r['val_accuracy'] for r in fold_results])
    avg_prec = np.mean([r['val_precision'] for r in fold_results])
    avg_rec = np.mean([r['val_recall'] for r in fold_results])
    avg_f1 = np.mean([r['val_f1'] for r in fold_results])
    
    std_acc = np.std([r['val_accuracy'] for r in fold_results])
    std_prec = np.std([r['val_precision'] for r in fold_results])
    std_rec = np.std([r['val_recall'] for r in fold_results])
    std_f1 = np.std([r['val_f1'] for r in fold_results])
    
    print(f"평균 검증 정확도: {avg_acc:.2f}% (±{std_acc:.2f}%)")
    print(f"평균 검증 Precision: {avg_prec:.2f}% (±{std_prec:.2f}%)")
    print(f"평균 검증 Recall: {avg_rec:.2f}% (±{std_rec:.2f}%)")
    print(f"평균 검증 F1: {avg_f1:.2f}% (±{std_f1:.2f}%)")
    
    # Fold별 상세 결과
    print(f"\nFold별 상세 결과:")
    print("-" * 60)
    for result in fold_results:
        print(f"Fold {result['fold']}: Acc={result['val_accuracy']:.2f}%, "
              f"Prec={result['val_precision']:.2f}%, "
              f"Rec={result['val_recall']:.2f}%, "
              f"F1={result['val_f1']:.2f}%")
    
    # Source-wise 평균 결과
    print(f"\n{'='*60}")
    print(f"Source-wise 평균 성능")
    print(f"{'='*60}\n")
    
    all_sources = set()
    for result in fold_results:
        all_sources.update(result['source_wise'].keys())
    
    source_avg_results = {}
    for source in all_sources:
        accs = []
        precs = []
        recs = []
        f1s = []
        
        for result in fold_results:
            if source in result['source_wise']:
                accs.append(result['source_wise'][source]['accuracy'])
                precs.append(result['source_wise'][source]['precision'])
                recs.append(result['source_wise'][source]['recall'])
                f1s.append(result['source_wise'][source]['f1'])
        
        if accs:
            source_avg_results[source] = {
                'accuracy': np.mean(accs),
                'accuracy_std': np.std(accs),
                'precision': np.mean(precs),
                'precision_std': np.std(precs),
                'recall': np.mean(recs),
                'recall_std': np.std(recs),
                'f1': np.mean(f1s),
                'f1_std': np.std(f1s)
            }
    
    print(f"{'Source':<20} {'Accuracy':<20} {'Precision':<20} {'Recall':<20} {'F1':<20}")
    print("-" * 100)
    
    for source, metrics in sorted(source_avg_results.items()):
        print(f"{source:<20} "
              f"{metrics['accuracy']:>6.2f}% (±{metrics['accuracy_std']:>4.2f}%)  "
              f"{metrics['precision']:>6.2f}% (±{metrics['precision_std']:>4.2f}%)  "
              f"{metrics['recall']:>6.2f}% (±{metrics['recall_std']:>4.2f}%)  "
              f"{metrics['f1']:>6.2f}% (±{metrics['f1_std']:>4.2f}%)")
    
    return {
        'fold_results': fold_results,
        'average': {
            'accuracy': avg_acc,
            'accuracy_std': std_acc,
            'precision': avg_prec,
            'precision_std': std_prec,
            'recall': avg_rec,
            'recall_std': std_rec,
            'f1': avg_f1,
            'f1_std': std_f1
        },
        'source_wise_avg': source_avg_results,
        'histories': all_fold_histories
    }


def export_to_openvino(model, output_dir="openvino_model"):
    """PyTorch 모델을 OpenVINO IR 형식으로 변환"""
    try:
        import openvino as ov
        
        print(f"\n{'='*60}")
        print("OpenVINO로 모델 변환 중...")
        print(f"{'='*60}\n")
        
        # 모델을 평가 모드로 전환
        model.eval()
        model.cpu()
        
        # 더미 입력 생성
        dummy_input = torch.randn(1, 3, CONFIG['image_size'], CONFIG['image_size'])
        
        # OpenVINO로 변환
        ov_model = ov.convert_model(model, example_input=dummy_input)
        
        # 저장
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        ov.save_model(ov_model, output_path / "model.xml")
        
        print(f"✓ OpenVINO 모델이 {output_path}에 저장되었습니다.")
        print(f"  - model.xml (모델 구조)")
        print(f"  - model.bin (가중치)")
        
        return True
        
    except ImportError:
        print("\n경고: OpenVINO가 설치되어 있지 않습니다.")
        print("설치하려면: pip install openvino")
        return False
    except Exception as e:
        print(f"\nOpenVINO 변환 중 오류 발생: {e}")
        return False


def main():
    """메인 함수"""
    print("="*60)
    print("CNN + OpenVINO 딥페이크 탐지 (K-Fold Cross Validation)")
    print("="*60)
    
    # GPU 정보 출력 (한 번만)
    print(f"\n사용 중인 디바이스: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 버전: {torch.version.cuda}")
    print()
    
    # 경로 확인
    print(f"\n현재 작업 디렉토리: {os.getcwd()}")
    print(f"DATA_DIR: {DATA_DIR.resolve()}")
    
    dirs = {
        'Real': REAL_DIR,
        'DDIM': DDIM_DIR,
        'DiffSwap': DIFFSWAP_DIR,
        'ADM': ADM_DIR
    }
    
    for name, dir_path in dirs.items():
        exists = dir_path.exists()
        print(f"\n{name} 디렉토리:")
        print(f"  경로: {dir_path.resolve()}")
        print(f"  존재 여부: {exists}")
        if exists:
            items = list(dir_path.iterdir())
            print(f"  내부 항목 수: {len(items)}")
            if len(items) > 0:
                image_files = list(dir_path.rglob('*.jpg')) + list(dir_path.rglob('*.png'))
                print(f"  이미지 파일 수: {len(image_files)}")
    
    # 데이터셋 준비 (랜덤 시드 고정)
    print("\n데이터셋 로딩 중...")
    datasets = prepare_datasets(random_state=42)
    
    print(f"\n로딩된 데이터:")
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
        
        # 데이터 샘플링 (랜덤 시드 고정)
        all_paths, all_labels, all_sources = create_dataset_splits(datasets, size, random_state=42)
        
        # K-Fold 학습
        results = train_kfold(all_paths, all_labels, all_sources, size)
        all_results[f'{size}장'] = results
        
        print(f"\n{'='*60}")
        print(f"{size}장 데이터셋 K-Fold 결과 요약")
        print(f"{'='*60}")
        print(f"평균 검증 정확도: {results['average']['accuracy']:.2f}% (±{results['average']['accuracy_std']:.2f}%)")
        print(f"평균 검증 F1: {results['average']['f1']:.2f}% (±{results['average']['f1_std']:.2f}%)")
    
    # 결과를 DataFrame으로 정리
    result_data = []
    for size, results in all_results.items():
        # 평균 검증 결과
        result_data.append({
            '데이터셋 크기': size,
            'Source': 'Overall (K-Fold Average)',
            'Accuracy': f"{results['average']['accuracy']:.2f}% (±{results['average']['accuracy_std']:.2f}%)",
            'Precision': f"{results['average']['precision']:.2f}% (±{results['average']['precision_std']:.2f}%)",
            'Recall': f"{results['average']['recall']:.2f}% (±{results['average']['recall_std']:.2f}%)",
            'F1': f"{results['average']['f1']:.2f}% (±{results['average']['f1_std']:.2f}%)"
        })
        
        # Source-wise 평균 결과
        for source, metrics in results['source_wise_avg'].items():
            result_data.append({
                '데이터셋 크기': size,
                'Source': source,
                'Accuracy': f"{metrics['accuracy']:.2f}% (±{metrics['accuracy_std']:.2f}%)",
                'Precision': f"{metrics['precision']:.2f}% (±{metrics['precision_std']:.2f}%)",
                'Recall': f"{metrics['recall']:.2f}% (±{metrics['recall_std']:.2f}%)",
                'F1': f"{metrics['f1']:.2f}% (±{metrics['f1_std']:.2f}%)"
            })
    
    df = pd.DataFrame(result_data)
    
    # 결과 저장
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "kfold_results.csv", index=False, encoding='utf-8-sig')
    
    # 시각화
    plot_kfold_results(all_results)
    
    # OpenVINO 변환 (마지막 모델 사용)
    print(f"\n{'='*60}")
    print("OpenVINO 모델 변환")
    print(f"{'='*60}")
    
    # 새 모델 생성 후 변환 (예시)
    final_model = CNN2D()
    export_to_openvino(final_model, output_dir="openvino_model")
    
    print(f"\n결과가 {output_dir}에 저장되었습니다.")


def plot_kfold_results(all_results):
    """K-Fold 결과 시각화"""
    if not all_results:
        print("시각화할 결과가 없습니다.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('K-Fold Cross Validation Results (CNN + OpenVINO)', fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]
        
        dataset_sizes = list(all_results.keys())
        values = []
        errors = []
        
        for size in dataset_sizes:
            values.append(all_results[size]['average'][metric])
            errors.append(all_results[size]['average'][f'{metric}_std'])
        
        x = np.arange(len(dataset_sizes))
        ax.bar(x, values, yerr=errors, capsize=5, alpha=0.7, color='steelblue')
        
        ax.set_xlabel('Dataset Size')
        ax.set_ylabel(f'{metric_name} (%)')
        ax.set_title(f'{metric_name} across Dataset Sizes (K-Fold Average)')
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_sizes)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])
        
        # 값 표시
        for i, (v, e) in enumerate(zip(values, errors)):
            ax.text(i, v + e + 2, f'{v:.1f}%\n±{e:.1f}%', 
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/kfold_comparison.png', dpi=300, bbox_inches='tight')
    print("\n시각화 결과가 저장되었습니다: results/kfold_comparison.png")
    
    # Source-wise 시각화
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle('Source-wise Performance (K-Fold Average)', fontsize=16, fontweight='bold')
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes2[idx // 2, idx % 2]
        
        # 첫 번째 데이터셋 크기의 source 추출
        first_size = list(all_results.keys())[0]
        sources = list(all_results[first_size]['source_wise_avg'].keys())
        
        x = np.arange(len(sources))
        width = 0.25
        
        for i, size in enumerate(all_results.keys()):
            values = []
            errors = []
            
            for source in sources:
                if source in all_results[size]['source_wise_avg']:
                    values.append(all_results[size]['source_wise_avg'][source][metric])
                    errors.append(all_results[size]['source_wise_avg'][source][f'{metric}_std'])
                else:
                    values.append(0)
                    errors.append(0)
            
            ax.bar(x + i * width, values, width, yerr=errors, 
                  capsize=3, label=size, alpha=0.8)
        
        ax.set_xlabel('Source')
        ax.set_ylabel(f'{metric_name} (%)')
        ax.set_title(f'{metric_name} by Source (K-Fold Average)')
        ax.set_xticks(x + width)
        ax.set_xticklabels([s.replace('Real vs ', '') for s in sources], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig('results/source_wise_kfold.png', dpi=300, bbox_inches='tight')
    print("시각화 결과가 저장되었습니다: results/source_wise_kfold.png")


if __name__ == "__main__":
    main()

