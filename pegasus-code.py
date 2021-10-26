# Implementation of a Noise Conditional Score Matching Network
# Based on https://github.com/ermongroup/ncsnv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import os
import time
import math
import random


# hyperparameters
batch_size  = 64
n_channels  = 3
latent_size = 512
dataset = 'stl10'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def cycle(iterable):
    while True:
        for x in iterable:
            yield x



# Linear Attention code based on https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/images.py
class LinearAttention(nn.Module):
    def __init__(self, dim, kernel_size = 1, padding = 0, stride = 1, key_dim = 64, value_dim = 64, heads = 4, norm_queries = True):
        super().__init__()
        self.dim = dim

        self.key_dim = key_dim
        self.value_dim = value_dim
        self.heads = heads

        self.norm_queries = norm_queries

        conv_kwargs = {'padding': padding, 'stride': stride}
        self.to_q = nn.Conv2d(dim, key_dim * heads, kernel_size, **conv_kwargs)
        self.to_k = nn.Conv2d(dim, key_dim * heads, kernel_size, **conv_kwargs)
        self.to_v = nn.Conv2d(dim, value_dim * heads, kernel_size, **conv_kwargs)

        out_conv_kwargs = {'padding': padding}
        self.to_out = nn.Conv2d(value_dim * heads, dim, kernel_size, **out_conv_kwargs)

    def forward(self, x):
        b, c, h, w, k_dim, heads = *x.shape, self.key_dim, self.heads
        q, k, v = (self.to_q(x), self.to_k(x), self.to_v(x))
        q, k, v = map(lambda t: t.reshape(b, heads, -1, h * w), (q, k, v))
        q, k = map(lambda x: x * (self.key_dim ** -0.25), (q, k))

        k = k.softmax(dim=-1)

        if self.norm_queries:
            q = q.softmax(dim=-2)

        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhdn,bhde->bhen', q, context)
        out = out.reshape(b, -1, h, w)
        out = self.to_out(out)
        return out


class InstanceNorm2dPlus(nn.Module):
    def __init__(self, num_features, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
        self.alpha = nn.Parameter(torch.zeros(num_features))
        self.gamma = nn.Parameter(torch.zeros(num_features))
        self.alpha.data.normal_(1, 0.02)
        self.gamma.data.normal_(1, 0.02)
        if bias:
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        means = torch.mean(x, dim=(2, 3))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))
        h = self.instance_norm(x)

        if self.bias:
            h = h + means[..., None, None] * self.alpha[..., None, None]
            out = self.gamma.view(-1, self.num_features, 1, 1) * h + self.beta.view(-1, self.num_features, 1, 1)
        else:
            h = h + means[..., None, None] * self.alpha[..., None, None]
            out = self.gamma.view(-1, self.num_features, 1, 1) * h
        return out


# ReZero code from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/master/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
class Rezero(nn.Module):
    def __init__(self, f):
        super(Rezero, self).__init__()
        self.f = f
        self.g = nn.Parameter(torch.zeros(1))
    def forward(self, X):
        return self.f(X) * self.g



class ConvBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            InstanceNorm2dPlus(n_in),
            nn.ELU(),
            nn.Conv2d(n_in, n_out, 3, padding=1, stride=1)
        )
    def forward(self, X):
        return self.block(X)


# Resnet Block based on code from https://github.com/ermongroup/ddim/blob/main/models/diffusion.py
class ResnetBlock(nn.Module):
    def __init__(self, n_in, n_out, n_t=256):
        super(ResnetBlock, self).__init__()
        self.block1 = ConvBlock(n_in, n_out)
        self.block2 = ConvBlock(n_out, n_out)
        self.t_reshape = nn.Linear(n_t, n_out)
        self.conv = nn.Conv2d(n_in, n_out, 1)
        self.t_act = nn.ELU()

    def forward(self, X, t):
        h = X
        h = self.block1(h)
        t = self.t_act(self.t_reshape(t))[:,:,None,None]
        h = h + t
        h = self.block2(h)
        return h + self.conv(X)

# Fourier Embedding code from https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing
class FourierEmbedding(nn.Module):
    def __init__(self, embedding_dim=64, scale_factor=16):
        super(FourierEmbedding, self).__init__()
        self.embedding = nn.Parameter(torch.randn(embedding_dim // 2)* scale_factor, requires_grad=False) 
    def forward(self,X):
        X = X[:,None] * self.embedding[None,:] * 2 * np.pi
        return torch.cat([torch.sin(X), torch.cos(X)],dim=1)


class ScoreNet(nn.Module):
    def __init__(self):
        super(ScoreNet, self).__init__()
        self.sigmas = torch.tensor(np.exp(np.linspace(np.log(50), np.log(0.01), 232))).float().to(device)
        self.embedding = FourierEmbedding()
        self.y_mlp = nn.Sequential(
            nn.Linear(64,256),
            nn.ELU(),
            nn.Linear(256,256)
        )
        self.conv_in = nn.Conv2d(3,32,3,1,1)
        self.conv_down_1 = ResnetBlock(32,64)
        self.attn_1 = Rezero(LinearAttention(64))
        self.conv_down_2 = ResnetBlock(64,128)
        self.attn_2 = Rezero(LinearAttention(128))
        self.conv_down_3 = ResnetBlock(128,256)
        self.attn_3 = Rezero(LinearAttention(256))

        
        self.middle_conv1 = ResnetBlock(256,256)
        self.middle_attn = Rezero(LinearAttention(256))
        self.middle_conv2 = ResnetBlock(256,256)
        
        self.conv_up_3 = ResnetBlock(256+256,128)
        self.attn_6 = Rezero(LinearAttention(128))
        self.conv_up_2 = ResnetBlock(128+128,64)
        self.attn_7 = Rezero(LinearAttention(64))
        self.conv_up_1 = ResnetBlock(64+64,32)
        self.attn_8 = Rezero(LinearAttention(32))
        self.out = nn.Conv2d(32,3,1,1)
    
    def forward(self, x, y=None):
      yemb = self.embedding(y)
      yemb = self.y_mlp(yemb)
      h = 2 * x - 1
      h = self.conv_in(h)
      h = self.conv_down_1(h,yemb)
      attn_1 = self.attn_1(h)
      h = F.avg_pool2d(attn_1, 2)
      h = self.conv_down_2(h, yemb)
      attn_2 = self.attn_2(h)
      h = F.avg_pool2d(attn_2, 2)
      h = self.conv_down_3(h, yemb)
      attn_3 = self.attn_3(h)
      h = F.avg_pool2d(attn_3, 2)
      h = self.middle_conv1(h, yemb)
      h = self.middle_attn(h)
      h = self.middle_conv2(h,yemb)
      h = F.interpolate(h, scale_factor=2, mode='nearest')
      h = torch.cat([h, attn_3], dim=1)
      h = self.conv_up_3(h, yemb)
      h = self.attn_6(h)
      h = F.interpolate(h, scale_factor=2, mode='nearest')
      h = torch.cat([h, attn_2], dim=1)
      h = self.conv_up_2(h, yemb)
      h = self.attn_7(h)
      h = F.interpolate(h, scale_factor=2, mode='nearest')
      h = torch.cat([h, attn_1], dim=1)
      h = self.conv_up_1(h, yemb)
      h = self.attn_8(h)
      X = self.out(h)
      sigmas = self.sigmas[y].view(X.shape[0], *([1] * len(X.shape[1:])))
      X = X / sigmas
      return X


def anneal_dsm_score_estimation(scorenet, samples, labels, sigmas, anneal_power=2.):
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
    target = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)
    scores = scorenet(perturbed_samples, labels)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    return loss.mean(dim=0)

@torch.no_grad()
def anneal_Langevin_dynamics(x_mod, scorenet, sigmas, n_steps_each=200, step_lr=0.000008,
                             final_only=True, denoise=True):
    images = []

    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            for s in range(n_steps_each):
                grad = scorenet(x_mod, labels)

                noise = torch.randn_like(x_mod)
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                x_mod = x_mod + step_size * grad + noise * np.sqrt(step_size * 2)

                image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
                snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
                grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

                if not final_only:
                    images.append(x_mod.to('cpu')) 

        if denoise:
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
            images.append(x_mod.to('cpu'))

        if final_only:
            return [x_mod.to('cpu')]
        else:
            return images

@torch.no_grad()
def interpolate_annealing_langevin_dynamics(x_mod, scorenet, sigmas, n_interpolations=15, T=5, step_lr=0.0000062, k=4):
    images = []
    n_rows = x_mod.shape[0]

    for c, sigma in enumerate(sigmas):
        labels = torch.ones(x_mod.shape[0],device=x_mod.device) * c
        labels = labels.long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        for step in range(T):
            grad = scorenet(x_mod, labels)
            noise_p = torch.randn(n_rows, x_mod.shape[1], x_mod.shape[2], x_mod.shape[3],device=x_mod.device)
            noise_q = torch.randn(n_rows, x_mod.shape[1], x_mod.shape[2], x_mod.shape[3],device=x_mod.device)
            angles = torch.linspace(0, np.pi/2.0, n_interpolations, device=x_mod.device)
            noise = noise_p[:,None, ...] * torch.cos(angles[k]) + noise_q[:,None, ...] * torch.sin(angles[k])
            noise = noise.reshape(-1, *noise.shape[2:])
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
            image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
            x_mod = x_mod + step_size * grad + noise * np.sqrt(step_size * 2)
            snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
    final = [x_mod.to("cpu")]
    return final


    
def train(net, optimiser, data_loader, n_epochs, device, start_epoch=0):
    start = time.time()
    sigmas = torch.tensor(np.exp(np.linspace(np.log(50), np.log(0.01), 232))).float().to(device)
    step = 0
    for epoch in range(start_epoch, n_epochs):
        for loader in data_loader:
            for i, (X,_) in enumerate(loader):
                X = X.to(device)
                X = X / 256. * 255. + torch.rand_like(X) / 256.
                labels = torch.randint(0, len(sigmas), (X.shape[0],), device=device)
                loss = anneal_dsm_score_estimation(net, X, labels, sigmas, 2)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                if step % 1000 == 0:
                    print(anneal_dsm_score_estimation(net, X, labels, sigmas, 2), step / 1000)
                step += 1
            torch.save({'A':net.state_dict(), 'optimiser':optimiser.state_dict(), 'epoch':epoch}, './model.pt')
    print(time.time() - start)


def sample(net, device, test_iterator):
    samples, _ = next(test_iterator)
    samples = torch.rand_like(samples).to(device)
    sigmas = np.exp(np.linspace(np.log(50), np.log(0.01), 232))
    interpolated_samples = interpolate_annealing_langevin_dynamics(samples, net, sigmas, 15, 5, 0.0000062, 11)
    return interpolated_samples

def sample_horses_birds(net, device, test_iterator):
    samples, _ = next(test_iterator)
    samples = torch.rand_like(samples).to(device)
    sigmas = np.exp(np.linspace(np.log(50), np.log(0.01), 232))
    generated_samples = anneal_Langevin_dynamics(samples, net, sigmas, 5, 0.0000062)
    return generated_samples




def main(load=True, cifar=True):
  score_net = ScoreNet().to(device)
  optimiser = torch.optim.Adam(score_net.parameters(), lr=0.0001)
  epoch = 0
  if load:
    params = torch.load('./model.pt')
    score_net.load_state_dict(params['A'])
    optimiser.load_state_dict(params['optimiser'])
    epoch = params['epoch']
  print(epoch)
  score_net.train()
  print(f'> Number of scorenet parameters {len(torch.nn.utils.parameters_to_vector(score_net.parameters()))}')
  if cifar:
    dataset = torchvision.datasets.CIFAR10('./cifar10', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
    ]))
    labels = torch.Tensor(dataset.targets)
    idxs = []
    for i in range(len(labels)):
        if labels[i] == 7:
            idxs.append(i)
        elif labels[i] == 2:
            idxs.append(i)

        
    sampler = torch.utils.data.SubsetRandomSampler(idxs)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, sampler=sampler, drop_last=True)

        
    train_iterator = iter(train_loader)
    x, t = next(train_iterator)
    x = x.to(device), t.to(device)
    plt.rcParams['figure.dpi'] = 175
    plt.grid(False)
    plt.imshow(torchvision.utils.make_grid(x[0]).cpu().data.permute(0,2,1).contiguous().permute(2,1,0),
    cmap=plt.cm.binary)
    plt.show()
  
    train(score_net, optimiser, [train_loader], 5000, device, epoch)
  else:
    dataset = torchvision.datasets.STL10('./stl10', split='train', download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Resize(48),
        torchvision.transforms.ToTensor(),
    ]))
    idxs = []
    for i in range(len(dataset)):
        if dataset[i][1] == 6:
            idxs.append(i)
        elif dataset[i][1] == 1:
            idxs.append(i)


    print(len(idxs))

        
    sampler = torch.utils.data.SubsetRandomSampler(idxs)

    loader_1 = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, sampler=sampler, drop_last=True)

    dataset = torchvision.datasets.STL10('./stl10', split='test', download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Resize(48),
        torchvision.transforms.ToTensor(),
    ]))
    idxs = []
    for i in range(len(dataset)):
    
        if dataset[i][1] == 6:
            idxs.append(i)
        elif dataset[i][1] == 1:
            idxs.append(i)

    print(len(idxs))
    sampler = torch.utils.data.SubsetRandomSampler(idxs)

    loader_2 = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, sampler=sampler, drop_last=True)
    train_iterator = iter(loader_2)
    x, t = next(train_iterator)
    x = x.to(device), t.to(device)
    plt.rcParams['figure.dpi'] = 175
    plt.grid(False)
    plt.imshow(torchvision.utils.make_grid(x[0]).cpu().data.permute(0,2,1).contiguous().permute(2,1,0),
    cmap=plt.cm.binary)
    plt.show()
    loader = [loader_1, loader_2]
    train(score_net, optimiser, loader, 5000, device, epoch)



def load(resize=48):
    score_net = ScoreNet().to(device)
    optimiser = torch.optim.Adam(score_net.parameters(), lr=0.001)
    epoch = 0
    params = torch.load('./model.pt')
    score_net.load_state_dict(params['A'])
    optimiser.load_state_dict(params['optimiser'])
    epoch = params['epoch']
    print(epoch)
    score_net.eval()
    print(f'> Number of scorenet parameters {len(torch.nn.utils.parameters_to_vector(score_net.parameters()))}')
    dataset = torchvision.datasets.CIFAR10('./cifar10', train=True, download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Resize(resize),
            torchvision.transforms.ToTensor(),
    ]))
    labels = torch.Tensor(dataset.targets)
    idxs = []
    for i in range(len(labels)):
        if labels[i] == 7:
            idxs.append(i)
        elif labels[i] == 2:
            idxs.append(i)

        
    sampler = torch.utils.data.SubsetRandomSampler(idxs)

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, sampler=sampler, drop_last=True)
    test_iterator = iter(test_loader)
    samples = sample(score_net, device, test_iterator)
    plt.rcParams['figure.dpi'] = 175
    plt.grid(False)
    plt.imshow(torchvision.utils.make_grid(samples[0]).cpu().data.permute(0,2,1).contiguous().permute(2,1,0),
    cmap=plt.cm.binary)
    plt.show()




if __name__ == "__main__":
    main()
    load()