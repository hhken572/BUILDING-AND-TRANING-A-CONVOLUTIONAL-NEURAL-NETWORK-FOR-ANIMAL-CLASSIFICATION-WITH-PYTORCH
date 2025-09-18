from torch.utils.data import Dataset, DataLoader
import os
import pickle
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torchvision.transforms import ToTensor, Resize, Compose, RandomHorizontalFlip, RandomRotation, Normalize
from tqdm import tqdm

class AnimalDataset(Dataset):
    def __init__(self, root, train=True, transform = None):
        self.image_paths = []
        self.labels = []
        self.categories = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']
        self.transform = transform

        data_path = os.path.join(root, "animals")

        if train:
            data_path = os.path.join(data_path, "train")
        else:
            data_path = os.path.join(data_path, "test")
        
        for i, category in enumerate(self.categories):
            data_files = os.path.join(data_path, category)
            for item in os.listdir(data_files):
                path = os.path.join(data_files, item)
                self.image_paths.append(path)
                self.labels.append(i)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = self.make_block(in_channels=3, out_channels=8)
        self.conv2 = self.make_block(in_channels=8, out_channels=16)
        self.conv3 = self.make_block(in_channels=16, out_channels=32)
        self.conv4 = self.make_block(in_channels=32, out_channels=64)
        self.conv5 = self.make_block(in_channels=64, out_channels=128)      

        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=6272, out_features=512),
            # nn.Linear(in_features=784, out_features=512),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=num_classes)
        ) 


    def make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    num_epochs = 100
    batch_size =8
    # transform = Compose([
    #     Resize((224, 224)),
    #     ToTensor(),
    # ])
    transform = Compose([
        Resize((224, 224)),
        RandomHorizontalFlip(),
        RandomRotation(10),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5])
    ])

    train_dataset = AnimalDataset(root="./data", train=True, transform=transform)
    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
        drop_last = True
    )

    test_dataset = AnimalDataset(root="./data", train=False, transform=transform)
    test_dataloader = DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = 4,
        drop_last = False
    )

    model = SimpleCNN(num_classes = 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    num_iters = len(train_dataloader)
    if torch.cuda.is_available():
        model.cuda()
    for epoch in range(num_epochs):    
        model.train()
        progress_bar = tqdm(train_dataloader)
        for iter, (images, labels) in enumerate(progress_bar):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            
            #forward
            outputs = model(images)
            loss_value = criterion(outputs, labels)
            # if (iter + 1)%10 == 0:
            #     print("Epoch {}/{}. Iteration {}/{}. Loss {}".format(epoch+1, num_epochs, iter+1, num_iters, loss_value))
            progress_bar.set_description("Epoch {}/{}. Iteration {}/{}. Loss {:.2}".format(epoch+1, num_epochs, iter+1, num_iters, loss_value))

            #backward
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
        
        model.eval()
        all_predictions = []
        all_labels = []
        for iter, (images, labels) in enumerate(test_dataloader):
            all_labels.extend(labels)
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            with torch.no_grad():
                predictions = model(images)
                indices = torch.argmax(predictions.cpu(), dim=1)
                all_predictions.extend(indices)
                loss_value = criterion(predictions, labels)
        
        all_labels = [label.item() for label in all_labels]
        all_predictions = [prediction.item() for prediction in all_predictions]
        print("Epoch {}".format(epoch+1))
        print(classification_report(all_labels, all_predictions))




