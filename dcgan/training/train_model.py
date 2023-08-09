import os
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dcgan.data.dataset import GANDataset
from dcgan.model.discriminator import Discriminator
from dcgan.model.generator import Generator


def train():
    config_path = os.path.join(Path.cwd(), "dcgan/model_params/params.yaml")
    # Read YAML file
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    epochs = config["epochs"]
    batch_size = config["batch_size"]
    # file_path = config["file_path"]
    lr = config["lr"]
    beta1 = config["beta1"]
    workers = config["workers"]
    latent_dims = config["latent_dims"]

    dataset = GANDataset(
        file_path=r"D:\img_align_celeba",
    )
    # Create the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
    )

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    netD = Discriminator().to(device)
    netG = Generator(latent_dims=latent_dims).to(device)

    # Initialize the ``BCELoss`` function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    # fixed_noise = torch.randn(64, 100, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.0
    fake_label = 0.0

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    G_losses = []
    D_losses = []

    batch_iter = 0

    for epoch in range(1, epochs + 1):
        print(f"Epoch, #{epoch}")

        with tqdm(dataloader, unit="Batch") as t_batches:
            for batch_idx, (data) in enumerate(t_batches):
                t_batches.set_description(f"Batch: #{batch_idx}")

                ## Train with all-real batch
                netD.zero_grad()
                # Format batch
                real_cpu = data.to(device)
                b_size = real_cpu.size(0)
                label = torch.full(
                    (b_size,),
                    real_label,
                    dtype=torch.float,
                    device=device,
                )
                # Forward pass real batch through D
                output = netD(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, latent_dims, 1, 1, device=device)
                # Generate fake image batch with G
                fake = netG(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                t_batches.set_postfix(gen_loss=errG.item(), disc_loss=errD.item())

                batch_iter += 1

        # Save model.
        torch.save(
            netG.state_dict(),
            os.path.join(Path.cwd(), f"dcgan/model_dicts/generator_{epoch}.pt"),
        )

    print(batch_iter)


if __name__ == "__main__":
    train()
