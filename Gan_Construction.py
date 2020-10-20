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

# Paper: https://arxiv.org/abs/1406.2661


#import the MNIST database
transform = transforms.Compose([transforms.ToTensor()
                            #   transforms.Normalize((0.5,), (0.5,)),
                              ])

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
print(len(mnist_trainset))
print(len(mnist_testset))


# transforms.ToTensor() — converts the image into numbers
# transforms.Normalize() — normalizes the tensor with a mean and standard deviation 

trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(mnist_testset, batch_size=64, shuffle=True)

# Understanding the shapes of the data
#dataiter = iter(trainloader)
#images, labels = dataiter.next()

# print("image shapes: ", images.shape)
# print("label shapes: ", labels.shape)

# 64 images in each batch and each image has a dimension of 28 x 28 pixels
# 64 images should have 64 labels respectively.


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Discriminator network
        # We will do a CNN
        # nn.Conv2d
        # in_channels(3 = red, blue green), out_channels (whatever tf you want), kernel_size (# of shared weights)
        
        # Convolutional layer, Pooling layer, Fully-Connected Layer
        # Cifar (32 by 32)
        #(32 - 4)/2 + 1 = 15
        #(15 - 5)/2 + 1 = 6*6*16 = 96

        # in-channel = 1, out-channel = 6, kernel_size = 4
        # batch size is automatic
        # receptive field is 5*5
        '''
        (w) = input volume size 
        (f) = the receptive field size of the Conv Layer neurons
        (s) = the stride with which they are applied
        (p) = the amount of zero padding used (P) on the border
        '''
        # (w-f+2p)/s +1 = (28-4)/2+1 = 13
        self.conv1 = nn.Conv2d(1, 6, 4, stride = 2) # output is of size (13*13*6) 
        # (w-f+2p)/s +1 = (13-5)/1+1 = 9
        self.conv2 = nn.Conv2d(6, 2, 5) # output is (9*9)*2
        self.fc1 = nn.Linear(9 * 9 * 2, 40) # batch size is 64
        self.fc2 = nn.Linear(40, 20)
        self.fc3 = nn.Linear(20, 2)
        self.very_soft = nn.Softmax()

        #Optimizer
        self.adam = optim.Adam(self.parameters(), lr = 3*10**(-3))

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    #def forward(self)??? (let's see what happens)
    def discriminate(self, x): #def discriminate
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.very_soft(x)
        return x[0] # make sure this doesnt mean taking the first in the batch


class Generator(nn.Module):
    def __init__(self, n):
        super(Generator, self).__init__()
        # Generator network
        self.loc = torch.zeros(n) #tensor filled with the scalar value 0 #AKA the mean
        self.scale = torch.ones(n) #tensor filled with the scalar value 1 #AKA the variance
        self.prior = torch.distributions.multivariate_normal.MultivariateNormal(self.loc, torch.eye(n))
            # Creates a multivariate normal (also called Gaussian) distribution parameterized by a mean vector and a covariance matrix. 
            # scale_tril = identity is used to increase the numerically computation stability for Cholesky decomposition
        self.g1 = nn.Linear(n ,25*25 )

        self.g2 = nn.Linear(25*25 ,28*28)

        self.adam = optim.Adam(self.parameters(), lr = 3*10**(-3))

    def generate(self, m): 
        sample = self.prior.sample(sample_shape=torch.Size([m]))# .detach() # does the distribution get updated?
            # detach() detaches the output from the computationnal graph. So no gradient will be backproped along this variable.
        o1 = F.relu(self.g1(sample))
        o2 = F.relu(self.g2(o1))
        return torch.reshape(o2, (m,1,28,28))


# Training Loop
number_of_iters = 50
discrim_steps = 1
m = 64 #number of samples in minibatch
learning_rate = 3*10**(-3)

# Construct framework
discriminator = Discriminator()
print(Discriminator)
generator = Generator(15)
# Create iterator for sampling data dist
dataiter = iter(trainloader)

# for name, param in generator.named_parameters():
#     print(name)
print(generator.parameters())


for ts in range(number_of_iters):
    for k in range(discrim_steps):
        # Sample minibatch of m noise samples from p_g(z)
        m_smp = generator.generate(m)

        # Sample minibatch of m examples from p_data(X)
        images, _ = dataiter.next() # images is the batch needed

        # Update discriminator by ascending stoch grad
        # 1/m sum ( log D(xi) + log(1-D(G(zi))) )
        #print(images.size())
        dis_loss = 1/m *  torch.sum( torch.log(discriminator.discriminate(images)) + torch.log(1-discriminator.discriminate( m_smp.detach() )) , dim=0 )
        
        discriminator.zero_grad()
        dis_loss.backward()
        discriminator.adam.step()
        



    # Sample minibatch of m noise samples {z^(1),...,z^(m)} from noise prior p_g(z)
    # Update the generator by descending its stochastic gradient:
    # 1/m sum ( log(1-D(G(zi))) )
    gen_loss = -1/m *  torch.sum(torch.log(1-discriminator.discriminate(generator.generate( m ) ) ) , dim=0 )
    discriminator.zero_grad()
    generator.zero_grad()
    gen_loss.backward()
    generator.adam.step()

# figure = plt.figure()
# num_of_images = 60
# for index in range(1, num_of_images + 1):
#     plt.subplot(6, 10, index)
#     plt.axis('off')
#     plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
# plt.show()

arr = np.ndarray((1,80,80,1))#This is your tensor
arr_ = np.squeeze(arr) # you can give axis attribute if you wanna squeeze in specific dimension
plt.imshow(arr_)
plt.show()