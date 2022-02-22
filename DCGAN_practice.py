#!/usr/bin/env python
# coding: utf-8

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import argparse
import random

# Parsing Arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
    parser.add_argument('--lr', type=float, default=5e-4, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
    parser.add_argument('--img_size', type=int, default=48, help='size of each image dimension')
    parser.add_argument('--channels', type=int, default=3, help='number of image channels')
    parser.add_argument('--sample_interval', type=int, default=400, help='interval between image samples')
    parser.add_argument('--seed', type=int, default=777, help='seed number')
    parser.add_argument('--object', type=str, default='person', help='which object to generate')
    
    args = parser.parse_args()
    
    return args

def set_dataloader(args):
    dataset = datasets.ImageFolder(root=f'./data/{args.object}',
                                   transform=transforms.Compose([
                                       transforms.Resize((args.img_size, args.img_size)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5], [0.5])
                                   ]))
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Generator(nn.Module):
    def __init__(self, latent_dim, image_shape):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        
        def block(input_fea, output_fea, normalize=True):
            layers = [nn.Linear(input_fea, output_fea)]
            if normalize:
                layers.append(nn.BatchNorm1d(output_fea, 0.5))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(image_shape))),
            nn.Tanh()
        )
    
    def forward(self, z):
        image = self.model(z)
        image = image.view(image.size(0), *self.image_shape)
        return image

class Discriminator(nn.Module):
    def __init__(self, image_shape):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(image_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, image):
        image_flat = image.view(image.size(0), -1)
        validity = self.model(image_flat)
        
        return validity

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
def main():
    set_seed(args)
    dataloader = set_dataloader(args)
    
    image_shape = (args.channels, args.img_size, args.img_size)
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        Tensor = torch.cuda.FloatTensor
    else:
        device = torch.device('cpu')
        Tensor = torch.FloatTensor
        
    print(f'Current Device: {device}\t Base Tensor: {Tensor}')
    
    criterion = torch.nn.BCELoss().to(device)
    generator = Generator(args.latent_dim, image_shape).to(device)
    discriminator = Discriminator(image_shape).to(device)
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    
    train(args, device, dataloader, criterion, generator, discriminator, optimizer_G, optimizer_D, Tensor)

import os 
from torch.autograd import Variable
from torchvision.utils import save_image
from datetime import datetime

def train(args, device, dataloader, criterion, generator, discriminator, optimizer_G, optimizer_D, Tensor):
    experiment_time = datetime.today().strftime('%Y%m%d_%H_%M')
    result_dir = f'images/{experiment_time}'
    model_dir = f'trained_models/{experiment_time}'
    
    os.makedirs(result_dir, exist_ok=False)
    os.makedirs(model_dir, exist_ok=False)
    
    for epoch in range(args.n_epochs):
        for idx, data in enumerate(dataloader):
            images, labels = data[0].to(device), data[1].to(device)
            
            # Adversarial
            valid = Variable(Tensor(images.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(images.size(0), 1).fill_(0.0), requires_grad=False)
            
            #
            real_images = Variable(images.type(Tensor))
            
            # Train Generator
            optimizer_G.zero_grad()
            
            z = Variable(Tensor(np.random.normal(0, 1, (images.size(0), args.latent_dim))))
            
            gen_images = generator(z)
            
            loss_G = criterion(discriminator(gen_images), valid)
            
            loss_G.backward()
            optimizer_G.step()
            
            # Train Discriminator
            optimizer_D.zero_grad()
            
            loss_real = criterion(discriminator(real_images), valid)
            loss_fake = criterion(discriminator(gen_images.detach()), fake)
            loss_D = (loss_real + loss_fake) / 2
            
            loss_D.backward()
            optimizer_D.step()
            
            if idx%20 == 0:
                print(f'[Epoch {epoch:d}/{args.n_epochs:d}] \t [Batch {idx:d}/{len(dataloader):d}] \t [Loss_G : {loss_G.item():.4f}] \t [Loss_D : {loss_D.item():0.4f}]')
                
            batches_done = epoch * len(dataloader) + idx
            
            if batches_done % args.sample_interval == 0:
                print('Save sample Image')
                save_image(gen_images.data[:25], f'{result_dir}/{batches_done:d}.png')
                
    print('Everything Done.. Saving Model')
    
    PATH_G = model_dir + '/generator.pth'
    PATH_D = model_dir + '/discriminator.pth'
    
    torch.save(generator.state_dict(), PATH_G)
    torch.save(discriminator.state_dict(), PATH_D)


if __name__ == '__main__':
    global args
    args = parse_args()
    main()