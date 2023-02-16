from torch import nn

from . import mininet


class Model(mininet.Model):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(40, 1024)
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512, 256)
        self.layer4 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        x = nn.functional.relu(x)
        x = self.layer3(x)
        x = nn.functional.relu(x)
        logits = self.layer4(x)
        return logits
