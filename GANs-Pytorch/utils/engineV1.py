import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch
from tqdm import tqdm
import sys



def train_one_epoch(g_model: torch.nn.Module,
                    d_model: torch.nn.Module,
                    criterion: nn.BCEWithLogitsLoss,
                    data_loader,
                    g_optimizer: torch.optim.Optimizer,
                    d_optimizer: torch.optim.Optimizer,
                    epoch: int,
                    loss_scaler,
                    max_norm,
                    args,
                    set_training_mode=True
                    ):

    g_model.train(set_training_mode)
    d_model.train(set_training_mode)

    d_model_loss, g_model_loss = 0, 0


    train_bar = tqdm(data_loader, file=sys.stdout, colour='red')
    for step, (imgs, _) in enumerate(train_bar):

        # Adversarial ground truths
        valid = Variable(torch.cuda.FloatTensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(torch.cuda.FloatTensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(torch.cuda.FloatTensor))

        g_optimizer.zero_grad()

        # Sample noise as generator input
        z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))))

        with torch.cuda.amp.autocast():
            # Generate a batch of images
            gen_imgs = g_model(z)
            # Loss measures generator's ability to fool the discriminator
            gen_outs = d_model(gen_imgs)
            g_loss = criterion(gen_outs, valid)

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(g_optimizer, 'is_second_order') and g_optimizer.is_second_order

        g_model_loss += g_loss.item()

        with torch.cuda.amp.autocast():
            loss_scaler(g_loss, g_optimizer, clip_grad=max_norm,
                        parameters=g_model.parameters(), create_graph=is_second_order)
        # g_loss.backward()
        # g_optimizer.step()


        # ---------------------
        #  Train Discriminator
        # ---------------------

        d_optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            # Measure discriminator's ability to classify real from generated samples
            real_loss = criterion(d_model(real_imgs), valid)
            fake_loss = criterion(d_model(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

        is_second_order = hasattr(d_optimizer, 'is_second_order') and d_optimizer.is_second_order

        d_model_loss += d_loss.item()
        with torch.cuda.amp.autocast():
            loss_scaler(d_loss, d_optimizer, clip_grad=max_norm,
                        parameters=d_model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        train_bar.desc = f'[Train Epoch {epoch}/{args.n_epochs}], [D loss: {(d_model_loss/(step+1)):.6f}], [G loss: {(g_model_loss/(step+1)):.6f}]'

        # d_loss.backward()
        # d_optimizer.step()
        # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return gen_imgs


@torch.no_grad()
def evaluate(data_loader, model, device, scalar=None):
    ...