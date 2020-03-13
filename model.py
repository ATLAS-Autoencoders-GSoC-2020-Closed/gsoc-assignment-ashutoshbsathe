import torch.nn as nn 
import torch.nn.functional as F 

class AutoEncoder(nn.Module):
    def __init__(self, connections=12, neg_slope=0.01):
        super(AutoEncoder, self).__init__()
        encoder = [
            nn.Linear(4, 4*connections),
            nn.LeakyReLU(neg_slope),
            nn.Linear(4*connections, 2*connections),
            nn.LeakyReLU(neg_slope),
            nn.Linear(2*connections, connections),
            nn.LeakyReLU(neg_slope),
            nn.Linear(connections, 3),
            nn.LeakyReLU(neg_slope)
        ]
        self.encoder = nn.Sequential(*encoder)
        decoder = [
            nn.Linear(3, connections),
            nn.LeakyReLU(neg_slope),
            nn.Linear(connections, 2*connections),
            nn.LeakyReLU(neg_slope),
            nn.Linear(2*connections, 4*connections),
            nn.LeakyReLU(neg_slope),
            nn.Linear(4*connections, 4)
        ]
        self.decoder = nn.Sequential(*decoder)
    
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

class AE_3D_200(nn.Module):
    def __init__(self, n_features=4):
        super(AE_3D_200, self).__init__()
        self.en1 = nn.Linear(n_features, 200)
        self.en2 = nn.Linear(200, 100)
        self.en3 = nn.Linear(100, 50)
        self.en4 = nn.Linear(50, 3)
        self.de1 = nn.Linear(3, 50)
        self.de2 = nn.Linear(50, 100)
        self.de3 = nn.Linear(100, 200)
        self.de4 = nn.Linear(200, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

    def decode(self, x):
        return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-200-100-50-3-50-100-200-out'

class AE_3D_200_Modified(nn.Module):
    def __init__(self, n_features=4):
        super(AE_3D_200_Modified, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 200),
            nn.Tanh(),
            nn.Linear(200, 100),
            nn.Tanh(),
            nn.Linear(100, 50),
            nn.Tanh(),
            nn.Linear(50, 3)
        )
        self.decoder = nn.Sequential(
            nn.Tanh(),
            nn.Linear(3, 50),
            nn.Tanh(),
            nn.Linear(50, 100),
            nn.Tanh(),
            nn.Linear(100, 200),
            nn.Tanh(),
            nn.Linear(200, n_features)
        )
        self.en1 = nn.Linear(n_features, 200)
        self.en2 = nn.Linear(200, 100)
        self.en3 = nn.Linear(100, 50)
        self.en4 = nn.Linear(50, 3)
        self.de1 = nn.Linear(3, 50)
        self.de2 = nn.Linear(50, 100)
        self.de3 = nn.Linear(100, 200)
        self.de4 = nn.Linear(200, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

    def decode(self, x):
        return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def describe(self):
        return 'in-200-100-50-3-50-100-200-out'