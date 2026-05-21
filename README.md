# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
This program develops a Convolutional Denoising Autoencoder using PyTorch to restore clean images from noisy inputs in the MNIST dataset. The model follows an encoder–decoder architecture, where the encoder extracts compressed feature representations of the input image and the decoder reconstructs the image back to its original form. To simulate real-world distortions, random noise is added to the input images during training, and the network learns to remove this noise effectively. The training process minimizes the reconstruction error between the original and output images using the Mean Squared Error (MSE) loss function, while the Adam optimizer is used for efficient weight updates. The model is trained over multiple epochs using batch processing for improved performance. Finally, the effectiveness of the autoencoder is evaluated by visually comparing the original, noisy, and reconstructed images.



## DESIGN STEPS
### STEP 1: 
Problem Understanding and Dataset Selection

### STEP 2: 
 Preprocessing the Dataset
 
### STEP 3: 
Design the Convolutional Autoencoder Architecture

### STEP 4: 
Compile and Train the Model

### STEP 5: 
Evaluate the Model

### STEP 6: 
Visualization and Analysis 

## PROGRAM

### Name: AKSHAY KARTHICK ASR

### Register Number: 212224230015

```
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()

train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128)

def noise(x):
    return torch.clamp(x + 0.5 * torch.randn_like(x), 0., 1.)

class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1,16,3,2,1),
            nn.ReLU(),
            nn.Conv2d(16,32,3,2,1),
            nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32,16,3,2,1,1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,1,3,2,1,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.dec(self.enc(x))

model = AE().to(device)
loss_fn = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr=1e-3)

summary(model, (1,28,28))

for e in range(5):
    model.train()
    total = 0
    for x,_ in train_loader:
        x = x.to(device)
        y = noise(x)
        out = model(y)
        loss = loss_fn(out, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    print(f"Epoch [{e+1}/5], Loss: {total/len(train_loader):.4f}")

model.eval()
with torch.no_grad():
    for x,_ in test_loader:
        x = x.to(device)
        y = noise(x)
        out = model(y)
        break

x = x.cpu().numpy()
y = y.cpu().numpy()
out = out.cpu().numpy()

plt.figure(figsize=(18,6))
n = 10

for i in range(n):
    plt.subplot(3,n,i+1)
    plt.imshow(x[i].squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(3,n,i+1+n)
    plt.imshow(y[i].squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(3,n,i+1+2*n)
    plt.imshow(out[i].squeeze(), cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()

```

### OUTPUT

### Model Summary

<img width="537" height="351" alt="Screenshot 2026-05-21 at 11 49 41 PM" src="https://github.com/user-attachments/assets/74770c01-4269-44dc-93e4-ad99449e97cf" />


### Training loss

<img width="203" height="88" alt="Screenshot 2026-05-21 at 11 50 04 PM" src="https://github.com/user-attachments/assets/63997de2-d83d-4ef7-8625-759720017440" />


## Original vs Noisy Vs Reconstructed Image
<img width="1045" height="325" alt="Screenshot 2026-05-21 at 11 50 38 PM" src="https://github.com/user-attachments/assets/111456e9-8e58-4985-bc3e-d922b0840a39" />

## RESULT
Therefore, To develop a convolutional autoencoder for image denoising application executed successfully.
