import torch
from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import os

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize images to range [-1, 1]
])

# Download and load the training dataset
train_dataset = datasets.MNIST(
    root='/home/prabhav/ml_tut',  # Directory to store the dataset
    train=True,     # Load training data
    transform=transform,
    download=True   # Download if dataset is not present
)

# Download and load the test dataset
test_dataset = datasets.MNIST(
    root='/home/prabhav/ml_tut',
    train=False,    # Load test data
    transform=transform,
    download=True
)
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def visulaize_dataset():

    import torchvision.transforms as T

    images, labels = next(iter(train_loader))
    transform_to_image = T.ToPILImage()
    # print(images.shape)
    images = images[0].reshape(1,28,28)
    print("Corresponding Label: ", labels[0])
    img = transform_to_image(images)
    img.show()

# Define image dimensions
img_rows = 28
img_cols = 28
channels = 1
img_shape = (channels, img_rows, img_cols)
latent_dim = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        noise_shape = (100,)

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=100, out_features=256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(256,momentum=0.8))   
        
        self.fc2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(512,momentum=0.8)
            )    

        self.fc3 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(1024,momentum=0.8)
            )       
        
        self.fc4 = nn.Sequential(
            nn.Linear(1024, np.prod(img_shape)),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = x.reshape(x.size(0), 28,28)
        return x


class Discriminator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(np.prod(img_shape),512)
        self. act1 = nn.LeakyReLU(negative_slope=0.2)
        self.fc2 = nn.Linear(512,256)
        self.act2 = nn.LeakyReLU(negative_slope=0.2)
        self.fc3 = nn.Linear(256,1)
        self.act_output = nn.Sigmoid()
    
    def forward(self,x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act_output(self.fc3(x))
        return x

generator_model = Generator().to(device)
discriminator_model = Discriminator().to(device)

'''
images = generator_model.forward(torch.rand(batch_size,100).to(device))
import torchvision.transforms as T
transform_to_image = T.ToPILImage()
img = transform_to_image(images[0])
img.show()
'''
lr = 0.001
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator_model.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator_model.parameters(), lr=lr, betas=(0.5, 0.999))
epochs = 100
noise_dim = 100
save_interval = 10
os.makedirs("images", exist_ok=True)

def train():
    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(train_loader):
            # Ground truths
            
            valid = torch.ones((imgs.size(0), 1), device=device, dtype=torch.float32)
            fake = torch.zeros((imgs.size(0), 1), device=device, dtype=torch.float32)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            real_imgs = imgs.to(device)
            # print(i, batch_size, real_imgs.size())
            real_imgs = real_imgs.reshape(imgs.size(0),784)

            # Generate fake images
            noise = torch.randn((imgs.size(0), noise_dim), device=device)
            gen_imgs = generator_model(noise)
            gen_imgs = gen_imgs.reshape(imgs.size(0), 784)
            # print(gen_imgs.shape)
            
            # Discriminator loss
            real_loss = adversarial_loss(discriminator_model(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator_model(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            # Optimize Discriminator
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            # ---------------------
            #  Train Generator
            # ---------------------
            # Generator loss
            g_loss = adversarial_loss(discriminator_model(gen_imgs), valid)

            # Optimize Generator
            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            # Print progress
            if i % 100 == 0:
                print(
                    f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(train_loader)}] "
                    f"[D loss: {d_loss.item()}] [G loss: {g_loss.item()}]"
                )

        # Save generated images at intervals
        if epoch % save_interval == 0:
            save_imgs(epoch)

def save_imgs(epoch):
    r, c = 5, 5
    noise = torch.randn((r * c, noise_dim), device=device)
    gen_imgs = generator_model(noise).detach().cpu()
    gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale to [0, 1]

    fig, axs = plt.subplots(r, c, figsize=(10, 10))
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt].squeeze(0), cmap="gray")
            axs[i, j].axis("off")
            cnt += 1
    fig.savefig(f"images/mnist_{epoch}.png")
    plt.close()


# Train the GAN
train()

# Save the generator model
torch.save(generator_model.state_dict(), "generator_model.pth")
