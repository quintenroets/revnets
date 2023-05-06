from . import mediumnet


class Model(mediumnet.Model):
    def __init__(self, input_size=784, hidden_size1=512, hidden_size2=256, **kwargs):
        super().__init__(
            input_size=input_size,
            hidden_size1=hidden_size1,
            hidden_size2=hidden_size2,
            **kwargs
        )

    def forward(self, x):
        x = x.view(len(x), -1)
        return super().forward(x)
