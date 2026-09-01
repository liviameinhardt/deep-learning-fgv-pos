"""Pre-treino do DCGAN do notebook aulas_praticas/GAN.ipynb.

Arquitetura e loop de treino copiados verbatim do notebook, para que os
state_dicts salvos carreguem sem nenhuma adaptacao. Muda apenas: mais dados,
mais epocas e checkpoints/amostras periodicos.
"""
import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __name__ != "__main__":
    raise ImportError(
        "train_gan.py e um script de treino: rode pela linha de comando, nao importe. "
        "Importar dispara um treino inteiro."
    )

p = argparse.ArgumentParser()
p.add_argument("--data", default="/u00/livia.meinhardt/gan_work/faces64")
p.add_argument("--out", default="/u00/livia.meinhardt/gan_work/checkpoints")
p.add_argument("--epochs", type=int, default=120)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--dist", default="normal")
p.add_argument("--save-every", type=int, default=5)
args = p.parse_args()

INPUT_SHAPE = 64          # no notebook e usado tanto como lado da imagem quanto como dim do z
LEAKY_SLOPE = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)
np.random.seed(42)
os.makedirs(args.out, exist_ok=True)


# ========================== Dados ==========================
class FacesInMemory(Dataset):
    """Imagens ja em 64x64; normalizadas para [-1,1] como no notebook."""

    def __init__(self, root):
        files = sorted(f for f in os.listdir(root) if f.endswith(".png"))
        arr = np.stack([imageio.imread(os.path.join(root, f)) for f in files])
        self.data = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # uint8 (N,3,64,64)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return (self.data[idx].float() - 127.5) / 127.5


# ========================== Modelos (identicos ao notebook) ==========================
class Generator(nn.Module):
    def __init__(self, input_dim=128, leaky_slope=0.2):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.leaky_slope = leaky_slope

        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024 * 4 * 4),
            nn.LeakyReLU(leaky_slope, inplace=True),

            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.Dropout(0.1),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.Dropout(0.1),

            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.Dropout(0.1),

            nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.net[0](z)
        x = self.net[1](x)
        x = x.view(-1, 1024, 4, 4)
        x = self.net[2:](x)
        return x


class Discriminator(nn.Module):
    def __init__(self, leaky_slope=0.2):
        super(Discriminator, self).__init__()
        self.leaky_slope = leaky_slope

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(leaky_slope, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(leaky_slope, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(leaky_slope, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(leaky_slope, inplace=True),

            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def rand_vector(distribution, size, device="cpu"):
    if distribution == "normal":
        return torch.randn(size, device=device)
    elif distribution == "uniform":
        return (torch.rand(size, device=device) * 2 - 1)
    elif distribution == "weibull":
        q = torch.from_numpy(np.random.weibull(50, size)).float().to(device)
        q = (q - q.mean()) / q.std()
        return q
    else:
        raise ValueError("Specify a valid distribution: normal, uniform, weibull")


def save_grid(path, generator, dist, samples=16, dim=(4, 4)):
    z = rand_vector(dist, (samples, INPUT_SHAPE), device=DEVICE)
    generator.eval()
    with torch.no_grad():
        imgs = generator(z)
    imgs = ((imgs + 1) * 127.5).clamp(0, 255).cpu().numpy().astype("uint8").transpose(0, 2, 3, 1)
    plt.figure(figsize=(6, 6))
    for i in range(samples):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(imgs[i]); plt.axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=80); plt.close()


dataset = FacesInMemory(args.data)
dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                        num_workers=4, drop_last=True, persistent_workers=True)
print(f"Dataset: {len(dataset)} imagens | device={DEVICE} | epochs={args.epochs}", flush=True)

generator = Generator(INPUT_SHAPE, LEAKY_SLOPE).to(DEVICE)
discriminator = Discriminator(LEAKY_SLOPE).to(DEVICE)
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

d_loss_history, g_loss_history = [], []
start_time = time.time()

for e in range(1, args.epochs + 1):
    d_running, g_running, nb = 0.0, 0.0, 0
    for real_batch in dataloader:
        batch_size = real_batch.size(0)
        real_batch = real_batch.to(DEVICE, non_blocking=True)

        # ---------------------  Train Discriminator  ---------------------
        discriminator.train(); generator.eval()

        real_labels = torch.ones(batch_size, 1, device=DEVICE) - 0.05 * torch.rand(batch_size, 1, device=DEVICE)
        fake_labels = torch.zeros(batch_size, 1, device=DEVICE) + 0.05 * torch.rand(batch_size, 1, device=DEVICE)

        z = rand_vector(args.dist, (batch_size, INPUT_SHAPE), device=DEVICE)
        fake_batch = generator(z)

        all_imgs = torch.cat([real_batch, fake_batch], dim=0)
        all_labels = torch.cat([real_labels, fake_labels], dim=0)
        idx = torch.randperm(all_imgs.size(0))
        all_imgs = all_imgs[idx][:batch_size]
        all_labels = all_labels[idx][:batch_size]

        optimizer_D.zero_grad()
        d_output = discriminator(all_imgs)
        d_loss = criterion(d_output, all_labels)
        d_loss.backward()
        optimizer_D.step()

        # -----------------  Train Generator  -----------------
        discriminator.eval(); generator.train()
        for p_ in discriminator.parameters():
            p_.requires_grad = False

        z = rand_vector(args.dist, (batch_size, INPUT_SHAPE), device=DEVICE)
        optimizer_G.zero_grad()
        g_output = discriminator(generator(z))
        g_loss = criterion(g_output, torch.ones(batch_size, 1, device=DEVICE))
        g_loss.backward()
        optimizer_G.step()

        for p_ in discriminator.parameters():
            p_.requires_grad = True

        d_running += d_loss.item(); g_running += g_loss.item(); nb += 1

    d_loss_history.append(d_running / nb); g_loss_history.append(g_running / nb)
    print(f"Epoch {e}/{args.epochs} | {time.time()-start_time:.0f}s | "
          f"g_loss {g_loss_history[-1]:.4f} | d_loss {d_loss_history[-1]:.4f}", flush=True)

    if e == 1 or e % args.save_every == 0 or e == args.epochs:
        torch.save({k: v.cpu() for k, v in generator.state_dict().items()},
                   os.path.join(args.out, f"generator_epoch{e}.pt"))
        torch.save({k: v.cpu() for k, v in discriminator.state_dict().items()},
                   os.path.join(args.out, f"discriminator_epoch{e}.pt"))
        save_grid(os.path.join(args.out, f"samples_epoch{e}.png"), generator, args.dist)

        plt.figure(figsize=(6, 5))
        plt.plot(d_loss_history, color="blue", label="Discriminator")
        plt.plot(g_loss_history, color="red", label="Generator")
        plt.legend(); plt.grid(True); plt.title("Learning curves")
        plt.ylabel("Loss"); plt.xlabel("Epochs"); plt.tight_layout()
        plt.savefig(os.path.join(args.out, "Learning_curves.png"), dpi=120); plt.close()

print("done", flush=True)
