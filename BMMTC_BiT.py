import pandas as pd
import numpy as np
import  torch
from torch.utils.data import Dataset, DataLoader
import os
import re
from torchvision import transforms, utils
import cv2

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from transformers import BitModel, AdamW, get_linear_schedule_with_warmup, AutoConfig, BitConfig

label_map = {"Business": 0, "Crime": 1, "Entertainment": 2, "Environment": 3, "Science-Tech": 4, "Others": 5}



class CustomImageDataLoader(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.label = self.data["Label"].tolist()
        self.image_path = self.data["Image_path"].tolist()
        self.samples = len(self.data)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.image_size = 112

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


class CustomBiTImageClassification(torch.nn.Module):
    def __init__(self, num_labels=6):
        super(CustomBiTImageClassification, self).__init__()
        self.config = AutoConfig.from_pretrained("google/bit-50", num_labels=6)
        self.config.return_dict = True
        self.model = BitModel.from_pretrained("google/bit-50", return_dict=True)
        #print(self.model)
        #2048 x 4 x 4
        self.flatten_dim = 32768
        self.hidden_size = 2048
        self.num_labels = 6
        self.fc = torch.nn.Linear(self.flatten_dim, self.hidden_size)
        self.linear = torch.nn.Linear(self.hidden_size, self.num_labels)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, pixel_values):
        output = self.model(pixel_values=pixel_values)
        #Extract 2048 from the last hidden state
        output = output.last_hidden_state
        output = output.mean(dim=[2,3])
        #print(output.shape)
        output = self.dropout(output)
        #output = self.fc(output)
        output = self.linear(output)
        output = self.softmax(output)
        return output

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def TrainBiTModel():
    datasets = CustomImageDataLoader("./MMMHC_train.csv")
    dataloader = DataLoader(datasets, batch_size=16, shuffle=True)
    num_labels = 6
    model = CustomBiTImageClassification()
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * 10)
    certiation_loss = torch.nn.CrossEntropyLoss()
    epochs = 30
    best_accuracy = 0
    best_loss = 100000000
    best_model = None
    for epoch in range(epochs):
        running_loss = 0.0
        predict_labels = []
        actual_labels = []
        for i, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            labels = batch["label"]
            optimizer.zero_grad()
            labels = torch.nn.functional.one_hot(labels, num_classes=num_labels).to(device)
            outputs = model(images)
            #print(outputs)
            #print(labels.float())
            loss = certiation_loss(outputs, labels.float())
            loss.backward()
            optimizer.step()
            running_loss +=  loss.item()/batch["image"].shape[0]
            outputs = outputs.argmax(dim=1).cpu().numpy()
            labels = labels.argmax(dim=1).cpu().numpy()
            predict_labels.extend(outputs)
            actual_labels.extend(labels)
        scheduler.step()
        print("Epoch: ", epoch, "Loss: ", running_loss)
        print(classification_report(actual_labels, predict_labels))
        if running_loss < best_loss:
            best_loss = running_loss
            best_model = model
            torch.save(best_model.state_dict(), "./Model/best_model_BiT_image.pt")



def TestBiTModel():
    dataset = CustomImageDataLoader("test-mmtc-6.csv")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    num_labels = 6
    model = CustomBiTImageClassification()
    model = model.to(device)
    model.load_state_dict(torch.load("./mmbtc-6-model/best_model_BiT_image.pt"))
    model.eval()

    predict_labels = []
    actual_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"]
            outputs = model(images)
            outputs = outputs.argmax(dim=1).cpu().numpy()
            labels = labels.numpy()
            predict_labels.extend(outputs)
            actual_labels.extend(labels)

    print(classification_report(actual_labels, predict_labels))
    print(confusion_matrix(actual_labels, predict_labels))
    acc = accuracy_score(actual_labels, predict_labels)
    print("Accuracy: ", acc)

    #Write a csv file for the misclassified images
    #
    # with open("./mmbtc-6-model/misclassified_images_for_BiT.csv", "w") as file:
    #     file.write("Image_path, Predicted_label, Actual_label\n")
    #     for i in range(len(actual_labels)):
    #         if actual_labels[i] != predict_labels[i]:
    #             file.write(f"{dataset.image_path[i]}, {predict_labels[i]}, {actual_labels[i]}\n")
    #
    #
    #



if __name__ == "__main__":
    TrainBiTModel()
    #TestBiTModel()

