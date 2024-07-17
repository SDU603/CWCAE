import torch
import torch.nn as nn


class CWCAE(nn.Module):
    def __init__(self, dataDim=512, hiddenDim=512, latentDim=32):
        super(CWCAE, self).__init__()
        self.dataDim = dataDim
        self.latentDim = latentDim

        # Encoder
        self.encLayers1 = nn.Sequential(
            LaplaceFast(16, 15),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2, return_indices=True),
        )
        self.encLayers2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, return_indices=True),
        )
        self.encLayers3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, return_indices=True),
        )
        self.encLayers4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, return_indices=True),
        )
        self.encLayers5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, return_indices=True),
        )
        self.encLayers6 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 64, hiddenDim),
            nn.BatchNorm1d(hiddenDim),
            nn.ReLU(),
        )

        self.latent = nn.Sequential(
            nn.Linear(hiddenDim, latentDim),
            nn.Tanh()
        )

        # Decoder
        self.decLayers6 = nn.Sequential(
            nn.Linear(latentDim, hiddenDim),
            nn.ReLU(),
            nn.BatchNorm1d(hiddenDim),
            nn.Linear(hiddenDim, 16 * 64),
        )
        self.unpooling5 = nn.MaxUnpool1d(2)
        self.decLayers5 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.ConvTranspose1d(64, 64, kernel_size=5, stride=1, padding=2),
        )
        self.unpooling4 = nn.MaxUnpool1d(2)
        self.decLayers4 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.ConvTranspose1d(64, 64, kernel_size=5, stride=1, padding=2),
        )
        self.unpooling3 = nn.MaxUnpool1d(2)
        self.decLayers3 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=1, padding=2),
        )
        self.unpooling2 = nn.MaxUnpool1d(2)
        self.decLayers2 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=1, padding=2),
        )
        self.unpooling1 = nn.MaxUnpool1d(2)
        self.decLayers1 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.ConvTranspose1d(16, 1, kernel_size=15, stride=1, padding=7)
        )

    def encode(self, x):
        x, _ = self.encLayers1(x)
        x, _ = self.encLayers2(x)
        x, _ = self.encLayers3(x)
        x, _ = self.encLayers4(x)
        x, _ = self.encLayers5(x)
        x = self.encLayers6(x)
        return self.latent(x)

    def forward(self, x):
        x, ind1 = self.encLayers1(x)
        x, ind2 = self.encLayers2(x)
        x, ind3 = self.encLayers3(x)
        x, ind4 = self.encLayers4(x)
        x, ind5 = self.encLayers5(x)
        x = self.encLayers6(x)
        x = self.latent(x)
        z = self.decLayers6(x)
        z = z.reshape(z.shape[0], 64, -1)
        z = self.unpooling5(z, ind5)
        z = self.decLayers5(z)
        z = self.unpooling4(z, ind4)
        z = self.decLayers4(z)
        z = self.unpooling3(z, ind3)
        z = self.decLayers3(z)
        z = self.unpooling2(z, ind2)
        z = self.decLayers2(z)
        z = self.unpooling1(z, ind1)
        z = self.decLayers1(z)
        return x, z


def contraLoss(featureL, featureU, label, pLabel, pValue, threshold=0.9, dMax=16.0):
    lossDir = (label == pLabel).float()
    lossP = torch.where(pValue > threshold, torch.tensor(1).cuda(), torch.tensor(0).cuda())
    distance = nn.functional.l1_loss(featureL, featureU, reduction='none')
    distance = torch.sum(distance, dim=1)
    loss = lossDir * distance + (1 - lossDir) * torch.clamp(dMax - distance, min=0.0)
    loss = loss * lossP
    return lossP.sum(), torch.sum(loss)


class Classifier(nn.Module):
    def __init__(self, inputNum: int = 32, outputNum: int = 10):
        super().__init__()
        self.output = nn.Linear(in_features=inputNum, out_features=outputNum)

    def forward(self, x):
        return self.output(x)

    def calcPseudoLabel(self, x):
        label = torch.argmax(self.output(x), dim=1)
        softmax = torch.softmax(self.output(x), dim=1)
        softValue = softmax[torch.arange(x.shape[0]), label]
        return label, softValue


def Laplace(p, A=0.08, ep=0.03, tal=0.1, f=20):
    w = 2 * torch.pi * f
    q = torch.tensor(1 - pow(ep, 2))
    y = A * torch.exp((-ep / (torch.sqrt(q))) * (w * (p - tal))) * (-torch.sin(w * (p - tal)))
    return y


class LaplaceFast(nn.Module):
    def __init__(self, out_channels, kernel_size):
        super(LaplaceFast, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # Can't view() here, otherwise the parameters can't be learned
        self.a_ = nn.Parameter(torch.linspace(1, 10, out_channels))
        self.b_ = nn.Parameter(torch.linspace(0, 10, out_channels))

    def forward(self, waveforms):
        time_disc = torch.linspace(0, 1, steps=int(self.kernel_size))
        p1 = time_disc.cuda() - self.b_.view(-1, 1).cuda() / self.a_.view(-1, 1).cuda()
        laplace_filter = Laplace(p1)
        filters = laplace_filter.view(self.out_channels, 1, self.kernel_size).cuda()
        return nn.functional.conv1d(waveforms, filters, stride=1, padding=self.kernel_size//2, dilation=1, bias=None, groups=1)
