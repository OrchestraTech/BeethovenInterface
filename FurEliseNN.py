import os
import json
from random import sample
from PIL import Image

from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs  
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

class EyeDataset(data.Dataset):
        def __init__(self, path, train=True, validation=False, transform=None):
            self.path = os.path.join(path, "train" if train else "test")
            if validation:
                self.path = os.path.join(path, "val")
            self.transform = transform
            self.samples = [] 

            open_path = os.path.join(self.path, "open")
            if os.path.exists(open_path):
                for img_name in os.listdir(open_path):
                    self.samples.append((os.path.join(open_path, img_name), 1)) # 1 = открыт 0 = закрыт

            closed_path = os.path.join(self.path, "closed")
            if os.path.exists(closed_path):
                for img_name in os.listdir(closed_path):
                    self.samples.append((os.path.join(closed_path, img_name), 0))

        def __getitem__(self, index):
            img_path, label = self.samples[index]

            image = Image.open(img_path).convert("RGB")
            
            if self.transform:
                image = self.transform(image)

            return image, label

        def __len__(self):
            return len(self.samples)

transforms = tfs.Compose([tfs.ToImage(), tfs.ToDtype(torch.float32, scale=True)])
path_to_my_dataset = r"C:\Users\Timofei\Desktop\CV\OrchestraModels\Datasets\OpenClosedDatased"
d_train = EyeDataset(path_to_my_dataset, transform=transforms)
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)

v_train = EyeDataset(path_to_my_dataset, validation=True, transform=transforms)
train_data_val = data.DataLoader(v_train, batch_size=32, shuffle = False)

model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 8, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(8, 4, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(484, 128),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(128, 1),
    nn.Sigmoid()
    
    )

optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',       
    patience=4,       
    factor=0.5,       
)

loss_function = nn.MSELoss()


epochs = 100
model.train()

#for graphics
loss_lst_val = [] #val
loss_lst = [] #train

for _e in range(epochs):
    model.train()

    loss_mean = 0
    lm_count = 0
    
    train_tqdm = tqdm(train_data)
    for x_train, y_train in train_tqdm:
        predict = model(x_train)
        loss = loss_function(predict, y_train.float().unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count += 1
        loss_mean = 1/lm_count * loss.item() + (1 - 1/lm_count) * loss_mean
        train_tqdm.set_description(f"Epoch [{_e+1}/{epochs}], loss_mean={loss_mean:.3f}")

    #model validation
    model.eval()
    Q_val = 0
    count_val = 0

    for x_val, y_val in train_data_val:
        with torch.no_grad():
            p = model(x_val)
            loss = loss_function(p, y_val.float().unsqueeze(1))
            Q_val += loss.item()
            count_val += 1

    Q_val /= count_val

    loss_lst.append(loss_mean)
    loss_lst_val.append(Q_val)

    scheduler.step(loss_mean)


st = model.state_dict()
torch.save(st, "FurEliseNN.tar")

#Testing
print("Тестирование")
d_test = EyeDataset(path_to_my_dataset, train=False, transform=transforms)
test_data = data.DataLoader(d_test, batch_size=50, shuffle=False)

Q = 0
count = 0
model.eval()

correct = 0
total = 0
y_pred = []
y_true = []


test_tqdm = tqdm(test_data)
for x_test, y_test in test_tqdm:
    with torch.no_grad():
        p = model(x_test)

        predicted = (p > 0.5).float() 
        total += y_test.size(0) 
        correct += (predicted.squeeze(1) == y_test).sum().item() 
        
        y_pred.extend(predicted.numpy().flatten())
        y_true.extend(y_test.numpy().flatten())

        Q += loss_function(p, y_test.float().unsqueeze(1)).item()
        count += 1

Q /= count
print(Q)

accuracy = 100 * correct / total
print(f"Точность на тестовой выборке: {accuracy:.2f}%")
print(f"Правильно: {correct}/{total}")

print("\nДетальный отчет по классам:")
print(classification_report(y_true, y_pred, target_names=['Открытые', 'Закрытые']))

plt.plot(loss_lst)
plt.plot(loss_lst_val)
plt.grid()
plt.show()

model.eval()

sample_batch, _ = next(iter(test_data)) 

with torch.no_grad():
    for _ in range(5):
        _ = model(sample_batch)


start_time = time.time()
with torch.no_grad():
    for _ in range(100):  
        _ = model(sample_batch)
end_time = time.time()


total_samples = sample_batch.size(0) * 100 
total_time = end_time - start_time

print(f"Общее время: {total_time:.4f} сек")
print(f"Размер батча: {sample_batch.size(0)}")
print(f"Инференс на 1 батч: {total_time / 100:.4f} сек")
print(f"Инференс на 1 изображение: {total_time / total_samples:.4f} сек")
print(f"FPS (кадров в секунду): {total_samples / total_time:.1f}")
