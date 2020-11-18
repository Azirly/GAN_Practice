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

trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=16, shuffle=True)
valloader = torch.utils.data.DataLoader(mnist_testset, batch_size=16, shuffle=True)

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
        self.fc3 = nn.Linear(20, 1)
        self.tanh = nn.Tanh()
        self.lk1 = nn.LeakyReLU(0.2, inplace=True)
        self.lk2 = nn.LeakyReLU(0.2, inplace=True)
        self.lk3 = nn.LeakyReLU(0.2, inplace=True)
        self.sg = nn.Sigmoid()


        for each_weight in self.parameters():
            nn.init.normal(each_weight)

        #Optimizer
        # self.adam = optim.Adam(self.parameters(), lr = 3)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    #def forward(self)??? (let's see what happens)
    def forward(self, x):   #def forward
        x = self.conv1(x)
        x = self.conv2(x) 
        x = x.view(-1, self.num_flat_features(x))
        x = self.lk1(self.fc1(x))
        x = self.lk2(self.fc2(x))
        x = self.lk3(self.fc3(x))
        # print(x)
        #x = x + 10**(-3)
        # x = self.very_soft(x)
        # x = torch.abs(self.tanh(x) + 10**(-3) )
        x = self.sg(x)
        # x = torch.clamp(x+0.001, min = 1, max = 3)
        # x= torch.add(x, .001)
        # print(x)
        return x[:,0]
        # return x[0] # make sure this doesnt mean taking the first in the batch


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

        nn.init.normal(self.g1.weight)
        nn.init.normal(self.g2.weight)

        # self.adam = optim.Adam(self.parameters(), lr = 3*10**(-2))
        self.drop1 = nn.Dropout(p=0.2) #p – probability of an element to be zeroed
        self.drop2 = nn.Dropout(p=0.2) #p – probability of an element to be zeroed

    def forward(self, m): 
        sample = self.prior.sample(sample_shape=torch.Size([m]))# .detach() # does the distribution get updated?
            # detach() detaches the output from the computationnal graph. So no gradient will be backproped along this variable.

        # o1 = F.tanh(self.g1(sample))
        # o2 = F.tanh(self.g2(o1))

        o1 = self.drop1(F.relu(self.g1(sample)))
        o2 = self.drop2(F.relu(self.g2(o1)))

        return torch.reshape(o2, (m,1,28,28))


TBLOGDIR=r"C:\Users\J Lin\Documents\CS Side Projects\Machine Learning Chris\Gan_Project\GAN_Practice\GAN_Training_Details\{}".format(get_start_time())

# Tensorboard Logging
writer = SummaryWriter(log_dir=TBLOGDIR)

# Training Loop
number_of_iters = 20000
discrim_steps = 1
m = 16 #number of samples in minibatch

# Construct framework
discriminator = Discriminator()
# print(Discriminator)
generator = Generator(10)
# Create iterator for sampling data dist
dataiter = iter(trainloader)

# for name, param in generator.named_parameters():
#     print(name)
print(generator.parameters())

g_list = [generator.g2.weight, generator.g2.bias, generator.g1.weight, generator.g1.bias]

disc_opt = optim.Adam(discriminator.parameters(), lr = 10**(-4))
# gen_opt = optim.Adam(generator.parameters(), lr = 10**(-3))
gen_opt = optim.Adam(params=g_list, lr = 5*10**(-2))


# for each_val in generator.named_parameters():
#     print("generator.named_parameters : ", each_val)
# raise 4

adversarial_loss = torch.nn.BCELoss()

for ts in range(number_of_iters):
    print("ts level is :", ts)
    for k in range(discrim_steps):
        # Sample minibatch of m noise samples from p_g(z)

        

        m_smp = generator(m)

        # Sample minibatch of m examples from p_data(X)
        images, _ = dataiter.next() # images is the batch needed

        # valid = Variable(Tensor(images.size(0), 1).fill_(1.0), requires_grad=False)
        # fake = Variable(Tensor(images.size(0), 1).fill_(0.0), requires_grad=False)

        valid = torch.full((images.size(0),), 1.0, dtype=torch.float)
        fake = torch.full((images.size(0),), 0.0, dtype=torch.float)

        # real_imgs = Variable(images.type(Tensor))
        # Update discriminator by ascending stoch grad
        # 1/m sum ( log D(xi) + log(1-D(G(zi))) )
        #print(images.size())
        discriminator.zero_grad()

        # test_loss = -torch.log(discriminator(images) )[0] 
        # print("test loss is :", test_loss)
        # print("test loss", test_loss)

        # dis_loss = 1/m *  torch.sum( torch.log(discriminator(images)) + torch.log(1-discriminator( m_smp.detach() )) , dim=0 )
        # print("torch.log(discriminator(images) is: ", torch.log(discriminator(images)))
        # print("discriminator(images) is: ", discriminator(images))
        # print("discriminator(images).shape is: ", discriminator(images).shape)
        # print("images is: ", images)
        # print("torch.log(1-discriminator( m_smp.detach() ))) is: ", torch.log(1-discriminator( m_smp.detach() )))
        # print("discriminator( m_smp.detach() )) is: ", discriminator( m_smp.detach() ))
        # print("m_smp.detach() ) is: ", m_smp.detach() )
        # print("m_smp ) is: ", m_smp )

        # for first_layer in images:
        #     for second_layer in first_layer:
        #         plt.imshow(second_layer)
        #         plt.show()

        # dis_loss = 1/2 * (torch.nn.BCELoss(discriminator(images), valid) + torch.nn.BCELoss( discriminator( m_smp.detach() ), fake) ) 
        
        # print("discriminator(images) reshape before: ", discriminator(images).size())
        # print("discriminator(images) reshape after: ", torch.reshape(discriminator(images), (16,1)).size())
        # print("discriminator(m_smp.detach() ).size() is: ", discriminator(m_smp.detach() ), (16,1))
        
        real_loss_tensor = torch.reshape(discriminator(images), (16,1))
        fake_loss_tensor = torch.reshape(discriminator(m_smp.detach() ), (16,1))
        real_loss = adversarial_loss(real_loss_tensor, valid)
        fake_loss = adversarial_loss(fake_loss_tensor, fake)
        # real_loss = adversarial_loss(real_loss_tensor)
        # fake_loss = adversarial_loss(fake_loss_tensor)
        print(real_loss)

        dis_loss = (real_loss + fake_loss) / 2
        # print("dis_loss is ", dis_loss)
        # print(discriminator(images))

        print (1-discriminator( m_smp.detach() ))

        print("fc3 weight before zero_grad() is: ", discriminator.fc3.weight)
        discriminator.zero_grad()

        # print('discriminator.fc3.grad before backward')
        # print(discriminator.fc3.weight.grad)

        dis_loss.backward()
        
        print("discriminator loss requires gradient?: ", dis_loss.requires_grad)
        # test_loss.backward()

        # print('discriminator.fc3.grad after backward')
        # print(discriminator.fc3.weight.grad)
        # print("fc3 weight after backwards is: ", discriminator.fc3.weight)
        disc_opt.step()
        # print("fc3 weight adam.step() is: ", discriminator.fc3.weight)

    writer.add_scalar("AverageDiscrimTrue", torch.mean(discriminator(images)), ts ) #want to be 1
    #Printing out the tensor as an image
    #print()
    #arr_ = np.squeeze(generator(1)[0].detach().numpy())
    
    # plt.axis('off')
    # plt.imshow(arr_, cmap='gray_r')
    if ts %20 == 0:
        writer.add_image("gan forwardd", generator(1)[0].detach(), ts )
        writer.add_image("sample image", images[0], ts )

        # print("TENSOR: ", generator(1)[0].detach())
        # print("IMAGE: ", images[0])
        #plt.show()


    fake = torch.full((images.size(0),), 0.0, dtype=torch.float)

    # Sample minibatch of m noise samples {z^(1),...,z^(m)} from noise prior p_g(z)
    # Update the generator by descending its stochastic gradient:
    # 1/m sum ( log(1-D(G(zi))) )
    writer.add_scalar("AverageDiscrimFalse", torch.mean(discriminator( generator( m ).detach() ) ), ts ) #want to be 50%
    #print()
    #??? Should BCELoss be used instead ???
    discriminator.zero_grad()
    generator.zero_grad()

    # gen_loss = -1/m *  torch.sum(torch.log(1-discriminator(generator( m ) ) ) , dim=0 )
    # print("torch.log(1-discriminator(generator( m ) )) is: ", torch.log(1-discriminator(generator( m ) )))
    # print("1- discriminator(generator( m ) ) is: ", 1 - discriminator(generator( m ) ))
    # print("log(0.5): ", math.log(0.5))
    # print("discriminator(generator( m ) ) is: ", discriminator(generator( m ) ))
    # print("discriminator(generator( m ) ).shape is: ", discriminator(generator( m ) ).shape)
    # print("m is: ", m)
    # print("torch.sum(torch.log(1-discriminator(generator( m ) ) ) is: ", torch.sum(torch.log(1-discriminator(generator( m ) ) )))
    # print("generator( m ) is: ", generator( m ))
    # print("generator( m ).shape is: ", generator( m ).shape)

    gen_loss = adversarial_loss(discriminator(generator( m ) ), fake)

    print('generator.g2.grad before backward: ',generator.g2.weight.grad)

    print("generator loss requires gradient?: ", gen_loss.requires_grad)
    print("type of gen_loss: ", gen_loss.type())

    gen_loss.backward()

    print('generator.g2.grad after backward: ', generator.g2.weight.grad)
    print("type of gen_loss: ", generator.g2.weight.grad.type())
    gen_opt.step()


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


