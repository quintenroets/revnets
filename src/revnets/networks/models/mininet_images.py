from . import mininet


class Model(mininet.Model):
    def __init__(self, input_size: int = 784, hidden_size: int = 40, **kwargs) -> None:
        super().__init__(input_size=input_size, hidden_size=hidden_size, **kwargs)

    def forward(self, x):
        x = x.view(len(x), -1)
        return super().forward(x)
