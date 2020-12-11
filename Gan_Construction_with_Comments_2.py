# on python 3.8.5 AKA python38

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions
import math

from torch.autograd import Variable


from torch.utils.tensorboard import SummaryWriter

# cuda = True if torch.cuda.is_available() else False
# Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Paper: https://arxiv.org/abs/1406.2661


#import the MNIST database
transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))
                              ])

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
print(len(mnist_trainset))
print(len(mnist_testset))


# transforms.ToTensor() — converts the image into numbers
# transforms.Normalize() — normalizes the tensor with a mean and standard deviation 

# Understanding the shapes of the data
#dataiter = iter(trainloader)
#images, labels = dataiter.next()

# print("image shapes: ", images.shape)
# print("label shapes: ", labels.shape)

# 64 images in each batch and each image has a dimension of 28 x 28 pixels
# 64 images should have 64 labels respectively.


from datetime import datetime

def get_curr_time():
    return datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

def get_start_time():
    return get_curr_time()


class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)
    
    # forward method
    def forward(self, x): 
        
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))
    
class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)
    
    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))


# Tensorboard Logging
writer = SummaryWriter(log_dir=TBLOGDIR)

bceloss = nn.BCELoss()
mnist_dim = mnist_trainset.train_data.size(1) * mnist_trainset.train_data.size(2)


def discrim_loss_d(discrim_images,num_images):
    # for images yn = 1 for bceloss
    true = torch.ones(num_images)
    return bceloss(discrim_images, true)

def gener_loss_d(discrim_gen_samples, num_images):
    # for fake images yn = 0 for bceloss
    false = torch.zeros(num_images)
    return bceloss(discrim_gen_samples, false)

def gener_loss_g(discrim_gen_samples, num_images):
    # for fake images yn = 1 for bceloss
    true = torch.ones(num_images)
    return bceloss(discrim_gen_samples, true)


def train(num_samples, trainloader):
    # Initialize discriminator and generator
    D = Discriminator(mnist_dim)
    G = Generator(g_input_dim=num_samples, g_output_dim= mnist_dim)

    # optimizers
    lr_G = 0.00022
    lr_D = 0.00025 
    G_optimizer = optim.Adam(G.parameters(), lr = lr_G)
    D_optimizer = optim.Adam(D.parameters(), lr = lr_D)

    # Training Loop
    for ts, (images, _ ) in enumerate(trainloader):
        # Discriminator loss for D
        test_image = images
        images = Variable(images.view(-1, mnist_dim))

        disc_tensors = D(images)
        # print(disc_tensors)p
        print("iteration is: ", ts)
        disc_loss = discrim_loss_d(disc_tensors, num_samples)
        # Generator Loss for D
        
        m_samples = Variable(torch.normal(mean=torch.zeros(num_samples, num_samples), std=torch.ones(num_samples, num_samples) ))
        # print(m_samples)
        x_fake = G(m_samples)
        # y_fake = torch.zeros(num_samples, 1)

        gen_tensors = D ( x_fake )
        gen_loss = gener_loss_d(gen_tensors, num_samples)

        # Updating Discriminator
        D.zero_grad()
        total_disc = (disc_loss + gen_loss)/2
        total_disc.backward()
        D_optimizer.step()

        # Updating Generator
        G.zero_grad()
        x_fake_2 = G(m_samples)
        gen_tensors_2 = D ( x_fake_2 )
        G_loss = gener_loss_g(gen_tensors_2, num_samples)
        G_loss.backward()
        G_optimizer.step()

        writer.add_scalar("AverageDiscrimFalse", torch.mean(D(G(m_samples))).detach(), ts ) #want to be 50%, feed bad image, but says it is true
        writer.add_scalar("AverageDiscrimTrue", torch.mean(D(images)), ts ) #want to be 1, feed an image and checks that it is true

        if ts %20 == 0:
            test_z = Variable(torch.randn(num_samples, num_samples))
            generated = G(test_z)
            writer.add_image("gan forward", generated.view(num_samples, 1, 28, 28)[0], ts )
            writer.add_image("sample image", test_image[0], ts )

if __name__ == "__main__":  
    print(mnist_dim)

    true_bs = 32
    trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=true_bs, shuffle=True)
    testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=true_bs, shuffle=True)

    train(true_bs, trainloader)