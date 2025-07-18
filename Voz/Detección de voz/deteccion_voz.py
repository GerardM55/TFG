import torch
import torch.nn as nn
import librosa
import numpy as np
import os
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

# Red neuronal basada en x-vectors con los mejores parámetros
class XVectorNet(nn.Module):
    def __init__(self, input_dim=30, hidden_dims=[128, 64, 32], dropout=0.248, output_dim=2):
        super(XVectorNet, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Configurar los hiperparámetros
hidden_dims = [128, 64, 32]
dropout = 0.24829483371480324
lr = 0.0001
batch_size = 64

# Cargar los datos
my_audio_dir = "C:/Users/gerar/Desktop/tfg/Control_de_veu/audios_deteccio_veu/meva_veu_aug_2"
other_audio_dir = "C:/Users/gerar/Desktop/tfg/Control_de_veu/audios_deteccio_veu/altres_veu_aug_2"

dataset = SpeakerDataset(my_audio_dir, other_audio_dir, augment=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Inicializar el modelo
model = XVectorNet(hidden_dims=hidden_dims, dropout=dropout)

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Entrenamiento con early stopping y guardado del modelo completo
def train_with_early_stopping(model, dataloader, criterion, optimizer, epochs=100, patience=5, save_path="best_model_complete.pth"):
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        total_loss = 0
        for mfcc, label in dataloader:
            optimizer.zero_grad()
            output = model(mfcc)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model, save_path)  # Guardar modelo completo
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break
    
    print(f'Training completed. Best model saved to {save_path}')

# Entrenar el modelo
train_with_early_stopping(model, dataloader, criterion, optimizer, epochs=100, patience=5)

# Evaluación
def evaluate_with_confusion_matrix(model, test_my_audio_dir, test_other_audio_dir):
    model.eval()
    true_labels = []
    predicted_labels = []
    
    for audio_file in os.listdir(test_my_audio_dir):
        if audio_file.endswith('.wav'):
            audio_path = os.path.join(test_my_audio_dir, audio_file)
            result = verify(model, audio_path)
            true_labels.append(0)
            predicted_labels.append(result)
    
    for audio_file in os.listdir(test_other_audio_dir):
        if audio_file.endswith('.wav'):
            audio_path = os.path.join(test_other_audio_dir, audio_file)
            result = verify(model, audio_path)
            true_labels.append(1)
            predicted_labels.append(result)
    
    cm = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix (combined for both voices):\n")
    print(cm)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0 (my voice)', 'Predicted 1 (other voice)'], yticklabels=['Actual 0 (my voice)', 'Actual 1 (other voice)'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Combined')
    plt.show()
    
    accuracy = np.trace(cm) / np.sum(cm) * 100
    return accuracy

# Verificación de hablante
def verify(model, audio_path):
    model.eval()
    mfcc = extract_mfcc(audio_path)
    mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(mfcc)
    _, predicted = torch.max(output, 1)
    return predicted.item()

# Evaluar el modelo en los datos de prueba
test_my_audio_dir = "C:/Users/gerar/Desktop/tfg/Control_de_veu/audios_deteccio_veu/meva_veu_test"
test_other_audio_dir = "C:/Users/gerar/Desktop/tfg/Control_de_veu/audios_deteccio_veu/altres_veu_test"

total_accuracy = evaluate_with_confusion_matrix(model, test_my_audio_dir, test_other_audio_dir)
print(f'Total Accuracy (combined): {total_accuracy:.2f}%')

# Cargar el modelo guardado en otro programa
# model = torch.load("best_model_complete.pth")
# model.eval()
