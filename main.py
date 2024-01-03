import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CustomDataset
from deeplabv3_model import DeepLabV3
from deeplabv3_loss import DeepLabV3Loss

# 参数设置
num_classes = 21
batch_size = 8
learning_rate = 0.001
num_epochs = 100

# 数据加载
train_dataset = CustomDataset('train.txt', transform=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = CustomDataset('val.txt', transform=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 模型创建
model = DeepLabV3(num_classes)
criterion = DeepLabV3Loss(num_classes)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练和测试函数
def train():
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0

def test():
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == labels).sum().item() / labels.size(0)
            print(f'Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train()
    test()
