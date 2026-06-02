import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split

#  НАСТРОЙКИ 
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001
IMG_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f" Устройство: {DEVICE}")

# ПОДГОТОВКА ДАННЫХ 
data_dir = "dataset"

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class CastDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {"OK": 0, "DEFECT": 1}

        for label in ["OK", "DEFECT"]:
            folder = os.path.join(root_dir, label)
            if not os.path.exists(folder):
                continue
            for img_name in os.listdir(folder):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(folder, img_name))
                    self.labels.append(self.class_to_idx[label])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Загружаем датасет
full_dataset = CastDataset(data_dir, transform=transform)
print(f" Всего изображений: {len(full_dataset)}")

# Считаем классы
labels = [item[1] for item in full_dataset]
count_ok = labels.count(0)
count_def = labels.count(1)
print(f"OK: {count_ok} |  DEFECT: {count_def}")

# Делим на train/val
train_idx, val_idx = train_test_split(
    list(range(len(full_dataset))),
    test_size=0.2,
    stratify=labels,
    random_state=42
)

train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
val_dataset = torch.utils.data.Subset(full_dataset, val_idx)

#  БАЛАНСИРОВКА КЛАССОВ 
# Считаем веса: чем меньше класс, тем больше вес
total = len(labels)
weight_ok = total / (2 * count_ok)
weight_def = total / (2 * count_def)
class_weights = torch.tensor([weight_ok, weight_def]).to(DEVICE)
print(f"&  Вес OK: {weight_ok:.2f} | Вес DEFECT: {weight_def:.2f}")

# Создаём сэмплер с весами
train_labels = [labels[i] for i in train_idx]
sample_weights = [weight_ok if l == 0 else weight_def for l in train_labels]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

#  МОДЕЛЬ 
model = models.resnet18(pretrained=True)

# Замораживаем веса
for param in model.parameters():
    param.requires_grad = False

# Меняем последний слой
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 2)
)
model = model.to(DEVICE)

#  ОБУЧЕНИЕ 
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

best_acc = 0.0

print("\nНачинаем обучение...\n")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total

    # Валидация
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total

    print(f"Эпоха [{epoch+1}/{EPOCHS}] | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"Модель сохранена! (Val Acc: {val_acc:.2f}%)")

print(f"\nОбучение завершено! Лучшая точность: {best_acc:.2f}%")
print("Модель сохранена в best_model.pth")
