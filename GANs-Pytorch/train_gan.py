import argparse
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
from timm.utils import NativeScaler
from torchvision import datasets
from utils.engineV1 import train_one_epoch
from models.gan import Generator, Discriminator
import torch


def get_args():
    parser = argparse.ArgumentParser(description='Train the gan-model on images')
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--model", type=str, default='GAN', help="name of model")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--pin_mem", type=bool, default=True, help="dataloader pin memory")
    parser.add_argument("--device", type=str, default='cuda:0', help="device (cuda:0 or cpu)")
    parser.add_argument("--save_images", type=str, default='./save_images', help="directory of saving images")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--nw", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")

    return parser.parse_args()


def main(args):
    print(args)

    img_shape = (args.channels, args.img_size, args.img_size)
    device = args.device

    train_loader = DataLoader(
        datasets.MNIST(
            "../../data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(args.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=args.batch_size,
        pin_memory=args.pin_mem,
        num_workers=args.nw,
        shuffle=True,
    )

    # Loss function
    adversarial_loss = torch.nn.BCEWithLogitsLoss()

    # Initialize generator and discriminator
    generator = Generator(img_shape, args.latent_dim).to(device)
    discriminator = Discriminator(img_shape).to(device)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    loss_scaler = NativeScaler()

    for epoch in range(args.n_epochs):
        gen_imgs = train_one_epoch(generator,
                                   discriminator,
                                   adversarial_loss,
                                   train_loader,
                                   optimizer_G,
                                   optimizer_D,
                                   epoch,
                                   loss_scaler,
                                   args.clip_grad,
                                   args,
                                   set_training_mode=True)
        batches_done = epoch * len(train_loader)
        if batches_done % args.sample_interval == 0:
            save_image(gen_imgs.data[:25], f"{args.save_images}/{args.model}/{batches_done}.png", nrow=5, normalize=True)


if __name__ == '__main__':
    args = get_args()
    if not os.path.exists(f'{args.save_images}/{args.model}'):
        os.makedirs(f'{args.save_images}/{args.model}', exist_ok=True)
    main(args)
