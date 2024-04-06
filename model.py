import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve

# CNN model definition
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(4*4*64, 128)  # Adjusted for 4x4 input feature map
        self.fc2 = nn.Linear(128, 10)  # 10 output classes

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 4*4*64)  # Adjust flattening according to your input
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Loading the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.USPS('data', train=True, download=True, transform=transform)
test_dataset = datasets.USPS('data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize models, loss function, optimizers, and TensorBoard writer
cnn = CNN()  # Now using the CNN model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters())  # Optimizer for CNN
writer = SummaryWriter()

# Training function
def train(model, optimizer, train_loader, epochs, model_name):
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'{model_name} Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                writer.add_scalar(f'Loss/{model_name}', loss.item(), epoch * len(train_loader) + batch_idx)

# Evaluation function with metrics and TensorBoard logging
def evaluate(model, test_loader, model_name):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=False)
            all_preds.extend(pred.tolist())
            all_targets.extend(target.tolist())

    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')

    print(f'{model_name} Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')

    all_preds_tensor = torch.tensor(all_preds, dtype=torch.float32)
    all_targets_tensor = torch.tensor(all_targets, dtype=torch.float32)

    # Log precision-recall curve for each class in TensorBoard
    for i in range(10):
        mask = all_targets_tensor == i
        preds_class = all_preds_tensor[mask]
        targets_class = all_targets_tensor[mask]

        writer.add_pr_curve(f'{model_name} class {i}', targets_class, preds_class, global_step=0)

# Training and evaluating CNN
train(cnn, optimizer, train_loader, epochs=10, model_name='CNN')
evaluate(cnn, test_loader, model_name='CNN')

writer.close()

