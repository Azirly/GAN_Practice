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
from torchvision.utils import save_image
from torch.autograd import Variable


from torch.utils.tensorboard import SummaryWriter


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=100, shuffle=True)
valloader = torch.utils.data.DataLoader(mnist_testset, batch_size=100, shuffle=True)


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

class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()

        self.fc1 = nn.Linear(d_input_dim, 1024) # batch size is 64
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):   #def forward
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))

class Generator(nn.Module):
    def __init__(self,g_input_dim, g_output_dim):
        super(Generator, self).__init__()
        # Generator network
        
            # Creates a multivariate normal (also called Gaussian) distribution parameterized by a mean vector and a covariance matrix. 
            # scale_tril = identity is used to increase the numerically computation stability for Cholesky decomposition
        
        self.g1 = nn.Linear(g_input_dim ,256)

        self.g2 = nn.Linear(self.g1.out_features, self.g1.out_features * 2)

        self.g3 = nn.Linear(self.g2.out_features, self.g2.out_features * 2)

        self.g4 = nn.Linear(self.g3.out_features,g_output_dim) #mnist_trainset.train_data.size(1) * mnist_trainset.train_data.size(2))

    def forward(self, m): 
        loc = torch.zeros(m) #tensor filled with the scalar value 0 #AKA the mean
        scale = torch.ones(m) #tensor filled with the scalar value 1 #AKA the variance
        prior = torch.distributions.multivariate_normal.MultivariateNormal(loc, torch.eye(m))
        sample = prior.sample(sample_shape=torch.Size([m]))#  # does the distribution get updated?
        # detach() detaches the output from the computationnal graph. So no gradient will be backproped along this variable.

        x = F.leaky_relu(self.g1(sample), 0.2)
        x = F.leaky_relu(self.g2(x), 0.2)
        x = F.leaky_relu(self.g3(x), 0.2)
        x = torch.tanh(self.g4(x))

        return x


# Tensorboard Logging
writer = SummaryWriter(log_dir=TBLOGDIR)

# Training Loop
number_of_iters = 20000
discrim_steps = 1



mnist_dim = mnist_trainset.train_data.size(1) * mnist_trainset.train_data.size(2)
# Construct framework
discriminator = Discriminator(mnist_dim).to(device)
# print(Discriminator)
generator = Generator(g_input_dim = 100, g_output_dim = mnist_dim).to(device)
# Create iterator for sampling data dist
dataiter = iter(trainloader)

# for name, param in generator.named_parameters():
#     print(name)
print(generator.parameters())

disc_opt = optim.Adam(discriminator.parameters(), lr = 0.0002 )
# gen_opt = optim.Adam(generator.parameters(), lr = 10**(-3))
gen_opt = optim.Adam(generator.parameters(), lr = 0.0002 )


# for each_val in generator.named_parameters():
#     print("generator.named_parameters : ", each_val)
# raise 4

adversarial_loss = torch.nn.BCELoss()

bs = 100

def D_train(images):
    discriminator.zero_grad()

    m_smp = generator(bs)

    images = images.view(-1, mnist_dim)

    valid = Variable( torch.ones( bs , 1).to(device) )
    fake = Variable( torch.zeros( bs, 1).to(device) )

    real_loss_tensor = discriminator(images)
    fake_loss_tensor = discriminator(m_smp )
    real_loss = adversarial_loss(real_loss_tensor, valid)
    fake_loss = adversarial_loss(fake_loss_tensor, fake)
    print(real_loss)

    dis_loss = (real_loss + fake_loss) / 2

    dis_loss.backward()
    
    disc_opt.step()

    writer.add_scalar("AverageDiscrimTrue", torch.mean(discriminator(images)).detach(), ts ) #want to be 1, feed an image and checks that it is true

    return dis_loss.data.item()

def G_train(images):
    
    generator.zero_grad()
    
    #m = Variable(torch.randn(bs, 100).to(device))
    fake = Variable(torch.ones(bs, 1).to(device))

    # Sample minibatch of m noise samples {z^(1),...,z^(m)} from noise prior p_g(z)
    # Update the generator by descending its stochastic gradient:
    # 1/m sum ( log(1-D(G(zi))) )
    writer.add_scalar("AverageDiscrimFalse", torch.mean(discriminator( generator( bs ) ) ).detach(), ts ) #want to be 50%, feed bad image, but says it is true

    gen_loss = adversarial_loss(discriminator(generator( bs ) ), fake)

    gen_loss.backward()
    gen_opt.step()

    return gen_loss.data.item()

# for ts in range(number_of_iters):
for ts in range(1, 200+1):           
    D_losses, G_losses = [], []
    new_image = None
    for batch_idx, (images, _) in enumerate(mnist_trainset):
        D_train(images)
        G_train(images)
        new_image = images

    if ts %20 == 0:
        writer.add_image("gan forwardd", generator(1)[0].detach(), ts )
        writer.add_image("sample image", new_image[0], ts )

        save_image(generator(100).view(generator(100).size(0), 1, 28, 28), "C:/Users/J Lin/Documents/CS Side Projects/Machine Learning Chris/Gan_Project/GAN_Practice/sample_" +str(ts) + ".png")



# Close the writer
writer.close()


# figure = plt.figure()
# num_of_images = 60
# for index in range(1, num_of_images + 1):
#     plt.subplot(6, 10, index)
#     plt.axis('off')
#     plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
# plt.show()

# arr = np.ndarray(generator(1)[0])

# arr = np.ndarray(generator)#This is your tensor
# arr_ = np.squeeze(arr) # you can give axis attribute if you wanna squeeze in specific dimension
# plt.imshow(arr_)
# plt.show()
