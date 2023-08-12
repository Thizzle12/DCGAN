from torch import nn


class Generator(nn.Module):
    def __init__(
        self,
        latent_dims: int = 100,
        n_features: int | tuple[int, int] = 64,
        n_channels: int = 3,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.main = nn.Sequential(
            # Latent space dimension for generating new images.
            nn.ConvTranspose2d(
                in_channels=latent_dims,
                out_channels=n_features * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(n_features * 8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                in_channels=n_features * 8,
                out_channels=n_features * 8,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            nn.ConvTranspose2d(
                in_channels=n_features * 8,
                out_channels=n_features * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(n_features * 4),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(
                in_channels=n_features * 4,
                out_channels=n_features * 4,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            nn.ConvTranspose2d(
                in_channels=n_features * 4,
                out_channels=n_features * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(n_features * 2),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(
                in_channels=n_features * 2,
                out_channels=n_features * 2,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            nn.ConvTranspose2d(
                in_channels=n_features * 2,
                out_channels=n_features,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(n_features),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(
                in_channels=n_features,
                out_channels=n_features,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            nn.ConvTranspose2d(
                in_channels=n_features,
                out_channels=n_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Tanh(),
            # nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)
