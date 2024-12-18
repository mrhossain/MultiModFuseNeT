import pandas as pd
import numpy as np
import  torch
from torch.utils.data import Dataset, DataLoader
import os
import re
from torchvision import transforms, utils
import cv2

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from transformers import ViTModel, ElectraModel, ElectraTokenizer, get_linear_schedule_with_warmup

label_map = {"Business": 0, "Crime": 1, "Entertainment": 2, "Environment": 3, "Science-Tech": 4, "Others": 5}

class CustomImageDataLoader(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.label = self.data["Label"].tolist()
        self.image_path = self.data["Image_path"].tolist()
        self.samples = len(self.data)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.image_size = 224

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        image = cv2.imread(self.image_path[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = self.transform(image)
        label = self.label[idx]
        label = label_map[label]
        return {'image': image, 'label': torch.tensor(label,dtype=torch.long)}

class CustomImageClassification(torch.nn.Module):
    def __init__(self, num_labels=6):
        super(CustomImageClassification, self).__init__()
        # self.config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=6)
        # # Return CLS token
        # self.config.return_dict = True
        self.hidden_size = 768
        self.num_labels = 6
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", return_dict=True)
        self.linear = torch.nn.Linear(self.hidden_size, self.num_labels)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, pixel_values):
        output = self.model(pixel_values=pixel_values)
        output = self.dropout(output.last_hidden_state[:, 0, :])
        output = self.linear(output)
        output = self.softmax(output)
        return output



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)




def TrainViTImageModel():
    datasets = CustomImageDataLoader("train-mmtc-6.csv")
    dataloader = DataLoader(datasets, batch_size=16, shuffle=True)
    num_labels = 6
    model = CustomImageClassification(num_labels=num_labels)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * 10)
    epochs = 30
    certiation_loss = torch.nn.CrossEntropyLoss()
    model.train()
    best_loss = 10000000000
    best_model = None
    for epoch in range(epochs):
        epoch_loss = 0
        predict_labels = []
        actual_labels = []
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label']
            output = model(images)
            labels = torch.nn.functional.one_hot(labels, num_classes=6).to(device)
            loss = certiation_loss(output, labels.float())
            epoch_loss += loss.item()/batch["image"].shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            output = output.argmax(dim=1).cpu().numpy()
            labels = labels.argmax(dim=1).cpu().numpy()
            predict_labels.extend(output)
            actual_labels.extend(labels)
        scheduler.step()
        print("Epoch: ", epoch, "Loss: ", epoch_loss)
        print(classification_report(actual_labels, predict_labels))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = model
    torch.save(best_model, "./mmbtc-6-model/best_model_ViT_image.pt")

def TestViTModel():
    dataset = CustomImageDataLoader("test-mmtc-6.csv")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = torch.load("./mmbtc-6-model/best_model_ViT_image.pt")
    model = model.to(device)
    model.eval()
    predict_labels = []
    actual_labels = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label']
            output = model(images)
            output = output.argmax(dim=1).cpu().numpy()
            labels = labels.numpy()
            predict_labels.extend(output)
            actual_labels.extend(labels)

    print(classification_report(actual_labels, predict_labels))
    print(confusion_matrix(actual_labels, predict_labels))
    acc = accuracy_score(actual_labels, predict_labels)
    print("Accuracy: ", acc)

    # #Write csv file for misclassified images
    # with open("./mmbtc-6-model/misclassified_images_for_ViT.csv", "w") as f:
    #     f.write("Image_path,Actual_label,Predicted_label\n")
    #     for i in range(len(actual_labels)):
    #         if actual_labels[i] != predict_labels[i]:
    #             f.write(f"{dataset.image_path[i]},{actual_labels[i]},{predict_labels[i]}\n")
    #

if __name__ == "__main__":
    TestViTModel()

    #TrainViTImageModel()
