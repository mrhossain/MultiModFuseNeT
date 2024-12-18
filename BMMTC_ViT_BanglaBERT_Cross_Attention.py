import pandas as pd
import numpy as np
import  torch
from torch.utils.data import Dataset, DataLoader
import os
import re
from torchvision import transforms, utils
import cv2
import torch.nn.functional as F
import torch.nn as nn

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from transformers import ViTModel, ElectraModel, ElectraTokenizer, get_linear_schedule_with_warmup

label_map = {"Business": 0, "Crime": 1, "Entertainment": 2, "Environment": 3, "Science-Tech": 4, "Others": 5}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.embed_dim = 768
        self.num_heads = 1
        self.text_attention = nn.MultiheadAttention(self.embed_dim, self.num_heads)
        self.image_attention = nn.MultiheadAttention(self.embed_dim, self.num_heads)

    def forward(self, text_embeddings, image_embeddings):
        # text_embeddings and image_embeddings shape: (batch_size, seq_length, embed_dim)

        # Apply text-to-image attention
        text_to_image_attn, _ = self.text_attention(text_embeddings, image_embeddings, image_embeddings)

        # Apply image-to-text attention
        image_to_text_attn, _ = self.image_attention(image_embeddings, text_embeddings, text_embeddings)

        return text_to_image_attn

        # Combine the attended features
        #print(f"text_to_image_attn shape: {text_to_image_attn.shape}")
        #print(f"image_to_text_attn shape: {image_to_text_attn.shape}")
        #combined_features = torch.cat((text_to_image_attn, image_to_text_attn), dim=-1)

        #return combined_features


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


class CustomTextDataLoader(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.label = self.data["Label"].tolist()
        self.sentences = self.data["Text"].tolist()
        self.samples = len(self.data)
        self.tokenizer = ElectraTokenizer.from_pretrained("csebuetnlp/banglabert")

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        text = self.sentences[idx]
        label = self.label[idx]
        label = label_map[label]
        encoded_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=16, return_tensors='pt')
        return {'input_ids': encoded_input['input_ids'].flatten(), 'attention_mask': encoded_input['attention_mask'].flatten(), 'label': torch.tensor(label,dtype=torch.long)}




class FocalLoss(nn.Module):
    def __init__(self, alpha=[.5, .5, 1], gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).to(device)  # Move alpha to the correct device here
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets)
        at = self.alpha[targets.data.view(-1).long()]  # Now alpha and targets are on the same device
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
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
        #output = self.linear(self.activation(output))
        #output = self.softmax(output)
        #return output
        #print(output.shape)
        #output = self.linear(output)
        #output = self.softmax(output)
        #print(output)
        return output



class CustomTextClassification(torch.nn.Module):
    def __init__(self, num_labels=6):
        super(CustomTextClassification, self).__init__()
        self.hidden_size = 768
        self.num_labels = num_labels
        self.model = ElectraModel.from_pretrained("csebuetnlp/banglabert", return_dict=True)
        self.linear = torch.nn.Linear(self.hidden_size, self.num_labels)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        output = self.dropout(output.last_hidden_state[:, 0, :])
        return output
        #output = self.linear(output)
        #output = self.softmax(output)
        #return output




def TestImageModel(temp_model):
    num_labels = 6
    #model = CustomImageClassification(num_labels=num_labels)
    #model = model.to(device)
    best_model_path = './mmbtc-6-model/best_Image_model.pt'
    dataset = CustomImageDataLoader("./BMMTC6-Final/BMMTC6-Test/test.csv")
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)
    model = temp_model #torch.load(best_model_path).to(device)
    #model = model.to(device)
    model.eval()
    predict_labels = []
    actual_labels = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label']
            output = model(images)
            output = output.argmax(dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            predict_labels.extend(output)
            actual_labels.extend(labels)
    acc = accuracy_score(actual_labels, predict_labels)
    print(classification_report(actual_labels, predict_labels))
    print(confusion_matrix(actual_labels, predict_labels))
    return acc



def TrainImageModel():
    datasets = CustomImageDataLoader("./BMMTC6-Final/BMMTC6-Test/train.csv")
    dataloader = DataLoader(datasets, batch_size=64, shuffle=True)
    num_labels = 6
    model = CustomImageClassification(num_labels=num_labels)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * 10)
    epochs = 20
    focal_loss =FocalLoss(alpha=[0.66, .67, .77], gamma=2) #FocalLoss() #torch.nn.CrossEntropyLoss()
    model.train()
    best_loss = 10000000000
    best_model = None
    globalAccuracy = 0
    for epoch in range(epochs):
        epoch_loss = 0
        predict_labels = []
        actual_labels = []
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label']
            labels = torch.nn.functional.one_hot(labels, num_classes=6).to(device)
            output = model(images)
            labels = labels.float()
            # output = torch.argmax(output, dim=1)
            # labels = torch.argmax(labels, dim=1)
            loss = focal_loss(output,labels)
            epoch_loss += loss.item()/batch["image"].shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            output = output.argmax(dim=1).cpu().numpy()
            labels = labels.argmax(dim=1).cpu().numpy()
            predict_labels.extend(output)
            actual_labels.extend(labels)
        scheduler.step()

        acc = TestImageModel(model)
        print("Epoch: ", epoch, "Loss: ", epoch_loss)
        #print(classification_report(actual_labels, predict_labels))
        if acc > globalAccuracy:
            best_model = model
            globalAccuracy = acc
            print(f"Save at Epoch: {epoch}")
            torch.save(best_model, "./Model/Image-based-Sentiment/best_Image_MSA_Focal_loss.pt")
    #torch.save(best_model, "./Model/Image-based-Sentiment/best_Image_MSA.pt")





def TrainTextModel():
    datasets = CustomTextDataLoader("./BMMTC6-Final/BMMTC6-Test/train.csv")
    dataloader = DataLoader(datasets, batch_size=16, shuffle=True)
    num_labels = 6
    model = CustomTextClassification(num_labels=num_labels)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * 10)
    epochs = 20
    certiation_loss = torch.nn.CrossEntropyLoss()
    model.train()
    best_loss = 10000000000
    best_model = None
    for epoch in range(epochs):
        epoch_loss = 0
        predict_labels = []
        actual_labels = []
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            output = model(input_ids, attention_mask)
            labels = torch.nn.functional.one_hot(labels, num_classes=6).to(device)
            loss = certiation_loss(output, labels.float())
            epoch_loss += loss.item()/batch["input_ids"].shape[0]
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
    torch.save(best_model, "./mmbtc-6-model/best_Text_model.pt")


def TestTextModel():
    best_model_path = './mmbtc-6-model/best_Text_model.pt'
    dataset = CustomTextDataLoader("test-mmtc-6.csv")
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)
    model = torch.load(best_model_path).to(device)
    model.eval()
    predict_labels = []
    actual_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            output = model(input_ids, attention_mask)
            output = output.argmax(dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            predict_labels.extend(output)
            actual_labels.extend(labels)
    print(classification_report(actual_labels, predict_labels))
    print(confusion_matrix(actual_labels, predict_labels))


#Label,Text,Image_path
class CustomDataLoaderMMTC(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.label = self.data["Label"].tolist()
        self.image_path = self.data["Image_path"].tolist()
        self.sentences = self.data["Text"].tolist()
        self.samples = len(self.data)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.image_size = 224
        self.tokenizer = ElectraTokenizer.from_pretrained("csebuetnlp/banglabert")

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        image = cv2.imread(self.image_path[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = self.transform(image)
        text = self.sentences[idx]
        label = self.label[idx]
        label = label_map[label]
        encoded_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=16, return_tensors='pt')
        return {'image': image, 'input_ids': encoded_input['input_ids'].flatten(), 'attention_mask': encoded_input['attention_mask'].flatten(), 'image_path':self.image_path[idx], 'label': torch.tensor(label,dtype=torch.long)}
class CustoMultiModalImgaeTextClassification(torch.nn.Module):
    def __init__(self, num_labels=3, image_model=None, text_model=None):
        super(CustoMultiModalImgaeTextClassification, self).__init__()
        self.hidden_size = 768
        self.num_labels = num_labels
        self.image_model = image_model
        self.text_model = text_model
        self.linear = torch.nn.Linear(self.hidden_size*2, self.num_labels)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, pixel_values, input_ids, attention_mask):
        image_output = self.image_model(pixel_values)
        text_output = self.text_model(input_ids, attention_mask)
        output = torch.cat((image_output, text_output), dim=1)
        output = self.dropout(output)
        output = self.linear(output)
        output = self.softmax(output)
        return output


class MultimodalImageTextCrossAttentionClassificationModel(nn.Module):
    def __init__(self, image_model=None, text_model=None):
        super(MultimodalImageTextCrossAttentionClassificationModel, self).__init__()
        self.embed_dim = 768
        self.num_heads = 1
        self.num_classes = 6
        self.image_model = image_model
        self.text_model = text_model
        self.cross_attention = CrossAttention(self.embed_dim, self.num_heads)
        self.fc1 = nn.Linear(self.embed_dim * 2, 512)  # Assuming combined_features has twice the embed_dim
        self.fc2 = nn.Linear(512, self.num_classes)
        self.linear = torch.nn.Linear(self.embed_dim*2, self.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pixel_values, input_ids, attention_mask):
        # text_embeddings and image_embeddings shape: (batch_size, seq_length, embed_dim)

        # Apply cross-attention
        text_embeddings = self.text_model(input_ids, attention_mask)
        image_embeddings = self.image_model(pixel_values)
        combined_features = self.cross_attention(text_embeddings, image_embeddings)

        combined_features = torch.cat((text_embeddings, combined_features), dim=1)

        #print(f"combined_features shape: {combined_features.shape}")

        # Aggregate features by mean pooling over the sequence length
        #combined_features = combined_features.mean(dim=1)

        # Classification layers
        #x = F.relu(self.fc1(combined_features))
        #x = self.fc2(x)
        x = self.linear(combined_features)
        x = self.softmax(x)
        return x


def TestMMTC(temp_model):
    #best_model_path = './mmbtc-6-model/best_mmtc_model_withoutfreez.pt'
    dataset = CustomDataLoaderMMTC("./BMMTC6-Final/BMMTC6-Test/test.csv")
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = temp_model #torch.load(best_model_path).to(device)
    model.eval()
    predict_labels = []
    actual_labels = []
    Image_path_list = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            image_path = batch['image_path']
            output = model(images, input_ids, attention_mask)
            output = output.argmax(dim=1).cpu().numpy()
            labels = labels.cpu().numpy()

            predict_labels.extend(output)
            actual_labels.extend(labels)

            Image_path_list.extend(image_path)


    #toal_miss = 0

    #Write a csv file with image path and predicted labels, actual labels only for the missclassified images
    # with open('./mmbtc-6-model/best_mmtc_model_withoutfreez_missclassified.csv', 'w') as f:
    #     f.write('Image_path, Predicted_Label, Actual_Label\n')
    #     for i in range(len(predict_labels)):
    #         if predict_labels[i] != actual_labels[i]:
    #             toal_miss += 1
    #             f.write(f'{Image_path_list[i]}, {predict_labels[i]}, {actual_labels[i]}\n')
    acc = accuracy_score(actual_labels, predict_labels)
    print(f"Accuracy: {acc}")
    print(classification_report(actual_labels, predict_labels))
    print(confusion_matrix(actual_labels, predict_labels))
    return acc




def TrainMMTC():
    datasets = CustomDataLoaderMMTC("./BMMTC6-Final/BMMTC6-Train/train.csv")
    dataloader = DataLoader(datasets, batch_size=32, shuffle=True)
    num_classes = 6
    image_model = CustomImageClassification(num_labels=num_classes)
    image_model = image_model.to(device)
    text_model = CustomTextClassification(num_labels=num_classes)
    text_model = text_model.to(device)
    # best_image_model_path = './mmbtc-6-model/best_Image_model.pt'
    # best_text_model_path = './mmbtc-6-model/best_Text_model.pt'
    # image_model = torch.load(best_image_model_path)
    # text_model = torch.load(best_text_model_path)
    #
    # for param in image_model.parameters():
    #     param.requires_grad = False
    # for param in text_model.parameters():
    #     param.requires_grad = False

    model = MultimodalImageTextCrossAttentionClassificationModel(image_model=image_model, text_model=text_model)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * 10)
    epochs = 20
    certiation_loss = FocalLoss() #torch.nn.CrossEntropyLoss()
    model.train()
    best_loss = 10000000000
    best_model = None
    globAlaccuracy = 0
    for epoch in range(epochs):
        epoch_loss = 0
        predict_labels = []
        actual_labels = []
        for batch in dataloader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            #image_path = batch['image_path']
            #print(image_path)
            output = model(images, input_ids, attention_mask)
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
        acc = TestMMTC(model)
        #print(classification_report(actual_labels, predict_labels))
        if acc > globAlaccuracy:
            globAlaccuracy = acc
            best_model = model
            torch.save(best_model, "./mmbtc-6-model/BMMTC_Focal_loss_Text_guided_Cross_Attention.pt")



if __name__ == "__main__":
    #TestMMTC()
    TrainMMTC()
    #TrainTextModel()
    #TestTextModel()
    #TrainImageModel()
    #TestImageModel()