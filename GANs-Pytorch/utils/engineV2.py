import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch
from tqdm import tqdm
import sys



def train_one_epoch(g_model: torch.nn.Module,
                    d_model: torch.nn.Module,
                    dataloader,
                    optimizer_G: torch.optim.Optimizer,
                    optimizer_D: torch.optim.Optimizer,
                    epoch: int,
                    args,
                    set_training_mode=True
                    ):

    g_model.train(set_training_mode)
    d_model.train(set_training_mode)

    d_model_loss, g_model_loss = 0, 0


    train_bar = tqdm(dataloader, file=sys.stdout, colour='red')
    for step, (imgs, _) in enumerate(train_bar):

        # Configure input
        real_imgs = Variable(imgs.type(torch.cuda.FloatTensor))

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))))

        # Generate a batch of images
        fake_imgs = g_model(z).detach()
        # Adversarial loss
        loss_D = -torch.mean(d_model(real_imgs)) + torch.mean(d_model(fake_imgs))

        d_model_loss += loss_D.item()
        loss_D.backward()
        optimizer_D.step()

        # Clip weights of discriminator
        for p in d_model.parameters():
            p.data.clamp_(-args.clip_value, args.clip_value)

        # Train the generator every n_critic iterations
        if step % args.n_critic == 0:
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = g_model(z)
            # Adversarial loss
            loss_G = -torch.mean(d_model(gen_imgs))

            g_model_loss += loss_G.item()
            loss_G.backward()
            optimizer_G.step()

        train_bar.desc = f'[Train Epoch {epoch}/{args.n_epochs}], [D loss: {(d_model_loss/(step+1)):.6f}], [G loss: {(g_model_loss/(step+1)):.6f}]'

    return gen_imgs


@torch.no_grad()
def evaluate(data_loader, model, device, scalar=None):
    ...