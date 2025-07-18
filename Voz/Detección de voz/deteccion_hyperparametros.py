import torch
import torch.nn as nn
import librosa
import numpy as np
import os
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Extraer MFCC con data augmentation
def extract_mfcc(audio_path, n_mfcc=30, augment=False):
    audio, sr = librosa.load(audio_path, sr=16000)
    
    if augment:
        if random.random() < 0.3:
            audio = librosa.effects.time_stretch(audio, rate=random.uniform(0.8, 1.2))
        if random.random() < 0.3:
            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=random.randint(-2, 2))
        if random.random() < 0.3:
            noise = np.random.randn(len(audio)) * 0.005
            audio = audio + noise
    
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

# Dataset personalizado
class SpeakerDataset(Dataset):
    def __init__(self, my_audio_dir, other_audio_dir, augment=False):
        self.audio_files = []
        self.labels = []
        self.augment = augment
        
        for audio_file in os.listdir(my_audio_dir):
            if audio_file.endswith('.wav'):
                self.audio_files.append(os.path.join(my_audio_dir, audio_file))
                self.labels.append(0)
        
        for audio_file in os.listdir(other_audio_dir):
            if audio_file.endswith('.wav'):
                self.audio_files.append(os.path.join(other_audio_dir, audio_file))
                self.labels.append(1)
        
    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        label = self.labels[idx]
        mfcc = extract_mfcc(audio_file, augment=self.augment)
        return torch.tensor(mfcc, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Red neuronal basada en x-vectors
class XVectorNet(nn.Module):
    def __init__(self, input_dim=30, hidden_dims=[128, 128, 64, 64, 32], dropout=0.2):
        super(XVectorNet, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, 2))  # 2 clases
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Función de entrenamiento
def train(model, dataloader, criterion, optimizer, epochs=30):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for mfcc, label in dataloader:
            optimizer.zero_grad()
            output = model(mfcc)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}')

# Búsqueda de hiperparámetros
def hyperparameter_search():
    best_acc = 0
    best_params = {}
    
    for _ in range(10):  # 10 combinaciones aleatorias
        hidden_dims = random.choice([[128, 128, 64, 64, 32], [256, 128, 64], [128, 64, 32]])
        dropout = random.uniform(0.1, 0.4)
        lr = random.choice([0.001, 0.0005, 0.0001])
        batch_size = random.choice([16, 32, 64])
        
        dataset = SpeakerDataset(my_audio_dir, other_audio_dir, augment=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = XVectorNet(hidden_dims=hidden_dims, dropout=dropout)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        train(model, dataloader, criterion, optimizer, epochs=20)
        
        accuracy = evaluate(model, test_my_audio_dir, test_label=0)
        accuracy += evaluate(model, test_other_audio_dir, test_label=1)
        accuracy /= 2
        
        print(f"Acc: {accuracy:.2f}% | Layers: {hidden_dims} | Dropout: {dropout} | LR: {lr} | Batch: {batch_size}")
        
        if accuracy > best_acc:
            best_acc = accuracy
            best_params = {'hidden_dims': hidden_dims, 'dropout': dropout, 'lr': lr, 'batch_size': batch_size}
    
    print("Mejores parámetros:", best_params)
    return best_params

# Verificación de hablante
def verify(model, audio_path):
    model.eval()  # Modo evaluación
    mfcc = extract_mfcc(audio_path)
    mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # Añadir dimensión de batch
    with torch.no_grad():
        output = model(mfcc)
    _, predicted = torch.max(output, 1)
    return predicted.item()


# Evaluación
def evaluate(model, test_audio_dir, test_label):
    model.eval()
    correct = 0
    total = 0
    for audio_file in os.listdir(test_audio_dir):
        if audio_file.endswith('.wav'):
            audio_path = os.path.join(test_audio_dir, audio_file)
            result = verify(model, audio_path)
            if result == test_label:
                correct += 1
            total += 1
    accuracy = correct / total * 100
    return accuracy

# Configuración
my_audio_dir = "C:/Users/gerar/Desktop/tfg/Control_de_veu/audios_deteccio_veu/meva_veu_aug_2"
other_audio_dir = "C:/Users/gerar/Desktop/tfg/Control_de_veu/audios_deteccio_veu/altres_veu_aug_2"
test_my_audio_dir = "C:/Users/gerar/Desktop/tfg/Control_de_veu/audios_deteccio_veu/meva_veu_test"
test_other_audio_dir = "C:/Users/gerar/Desktop/tfg/Control_de_veu/audios_deteccio_veu/altres_veu_test"

# Búsqueda de mejores hiperparámetros
best_params = hyperparameter_search()
