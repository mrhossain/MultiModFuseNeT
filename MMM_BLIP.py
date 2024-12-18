import pandas as pd
import numpy as np
import  torch
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn as nn
import re
import torch.nn.functional as F
#from torch.xpu import device
from torchvision import transforms, utils
import cv2
from torch.cuda.amp import GradScaler, autocast
from PIL import Image

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from transformers import CLIPModel, CLIPProcessor, AdamW, get_linear_schedule_with_warmup, AutoConfig, BlipProcessor, BlipModel, Blip2Model, Blip2Processor

label_map = {"Business": 0, "Crime": 1, "Entertainment": 2, "Environment": 3, "Science-Tech": 4, "Others": 5}

class CustomBLIPDataloder(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.label = self.data["Label"].tolist()
        self.image_path = self.data["Image_path"].tolist()
        self.samples = len(self.data)
        self.text = self.data["Text"].tolist()
        self.tokenizer = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
    def __len__(self):
        return self.samples
    def __getitem__(self,idx):
        image_path = self.image_path[idx]
        label = label_map[self.label[idx]]
        #image = cv2.imread(image_path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.open(image_path)
        image = self.transform(image)
        text = self.text[idx]
        encoded_input = self.tokenizer(text=text, images=image, padding='max_length', truncation=True, max_length=16, return_tensors='pt')
        return {'pixel_values': image, 'input_ids': encoded_input['input_ids'].flatten(), 'attention_mask': encoded_input['attention_mask'].flatten(), 'image_path': self.image_path[idx],'label': torch.tensor(label,dtype=torch.long)}



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Print Device',device)



class MFBFusion(nn.Module):
    def __init__(self, image_embed_dim, text_embed_dim, fact_dim, output_dim):
        super(MFBFusion, self).__init__()
        self.image_embed_dim = image_embed_dim
        self.text_embed_dim = text_embed_dim
        self.fact_dim = fact_dim
        self.output_dim = output_dim
        self.image_linear = nn.Linear(self.image_embed_dim, self.fact_dim * self.output_dim)
        self.text_linear = nn.Linear(self.text_embed_dim, self.fact_dim * self.output_dim)
    def forward(self, text_embeddings, image_embeddings):
        image_proj = self.image_linear(image_embeddings)
        text_proj = self.text_linear(text_embeddings)
        mfb_output = image_proj * text_proj
        mfb_output = mfb_output.view(-1,self.fact_dim, self.output_dim)
        mfb_output = mfb_output.mean(dim=1)
        mfb_output = F.normalize(mfb_output, p=2, dim=1)
        return mfb_output



class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            targets = targets.long()  # Ensure targets are long tensor for indexing
            alpha = self.alpha[targets]  # Index alpha using targets
            alpha = alpha.view(-1, 1)  # Adjust dimensions for broadcasting
            #print(f"alpha: {alpha.shape}, focal_loss:{focal_loss.shape}")

            focal_loss = alpha * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CustomBLIP(torch.nn.Module):
    def __init__(self, num_labels=6):
        super(CustomBLIP, self).__init__()
        self.config = AutoConfig.from_pretrained("Salesforce/blip-image-captioning-base", num_labels=6)
        self.config.return_dict = True
        self.model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base", return_dict=True)
        self.text_embedding_size = 512
        self.image_embedding_size = 512
        self.input_size =  50271


        self.image_embed_dim = 512
        self.text_embed_dim = 512
        self.fact_dim = 16
        self.output_dim = 1000
        self.num_labels = num_labels
        self.mfb_fusion = MFBFusion(image_embed_dim=self.image_embed_dim, text_embed_dim=self.text_embed_dim,
                                    fact_dim=self.fact_dim, output_dim=self.output_dim)

        self.linear = torch.nn.Linear(self.output_dim, self.num_labels)


        self.fc = torch.nn.Linear(self.input_size+1, self.text_embedding_size)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, pixel_values, attention_mask,input_ids):
        output = self.model(pixel_values=pixel_values, attention_mask=attention_mask, input_ids=input_ids)
        #For CLIP and BLIP
        image_embeds = output["image_embeds"]
        text_embeds = output["text_embeds"]
        image_embeds = image_embeds.to(dtype=torch.float32)
        text_embeds = text_embeds.to(dtype=torch.float32)
        fused = self.mfb_fusion(image_embeds, text_embeds)
        #fused = torch.cat((image_embeds, text_embeds), dim=1)
        res = self.linear(fused)
        res = self.softmax(res)
        return res



def TestBMMTC_BLIP():
    model = CustomBLIP()
    model.load_state_dict(torch.load('./Model/BLIP.pth'))
    model.to(device)
    model.eval()
    datasets = CustomBLIPDataloder("./BMMTC6-Final/test.csv")
    dataloader = DataLoader(datasets, batch_size=16, shuffle=True)
    predictions = []
    actual = []
    image_path_list = []
    for batch in dataloader:
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label = batch['label']
        image_path = batch['image_path']
        output = model(pixel_values=pixel_values, attention_mask=attention_mask,input_ids=input_ids)
        output = output.argmax(dim=1).cpu().numpy()
        image_path_list.extend(image_path)
        predictions.extend(output)
        label = label.numpy()
        actual.extend(label)
    acc = accuracy_score(actual, predictions)
    print("Accuracy:", accuracy_score(actual, predictions))
    print("Precision:", precision_score(actual, predictions, average='macro'))
    print("Recall:", recall_score(actual, predictions, average='macro'))
    print("F1 Score:", f1_score(actual, predictions, average='macro'))
    print("Classification Report:", classification_report(actual, predictions))
    print("Confusion Matrix:\n", confusion_matrix(actual, predictions))
    return acc



def TrainBMMTC_BLIP():
    model = CustomBLIP()
    model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    class_counts = [2335, 1000, 3667, 762, 1134, 1784]
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()  # Normalize weights
    class_weights = class_weights.to(device)
    certian_loss = FocalLoss(alpha=class_weights, gamma=2, reduction='mean')


    #certian_loss = torch.nn.CrossEntropyLoss()
    datasets = CustomBLIPDataloder("./BMMTC6-Final/train.csv")
    dataloader = DataLoader(datasets, batch_size=16,shuffle=True)
    epochs = 30
    best_model = None
    criterion = certian_loss
    globalAcc = 0
    for epoch in range(epochs):
        print("Epoch:", epoch)
        predictions = []
        actual = []
        epoch_loss = 0
        for batch in dataloader:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label']
            optimizer.zero_grad()
            output = model(pixel_values=pixel_values, attention_mask=attention_mask,input_ids=input_ids)
            label = torch.nn.functional.one_hot(label, num_classes=6).to(device)
            loss = certian_loss(output, label.float()) #certian_loss(output, label.float())
            loss.backward()
            optimizer.step()
            output = output.argmax(dim=1).cpu().numpy()
            predictions.extend(output)
            label = label.argmax(dim=1).cpu().numpy()
            actual.extend(label)
            epoch_loss += loss.item()/batch["label"].shape[0]
        scheduler.step()
        accuracyscore = TestBMMTC_BLIP(model)
        print("Accuracy:", accuracyscore)
        print("Loss:", epoch_loss)
        if globalAcc < accuracyscore:
            globalAcc = accuracyscore
            torch.save(model.state_dict(), './Model/BLIP.pth')
if __name__ == "__main__":
    #TrainBMMTC_BLIP()
    TestBMMTC_BLIP()

