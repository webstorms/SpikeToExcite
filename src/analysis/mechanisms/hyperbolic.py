import torch
import torch.nn as nn
import torch.nn.functional as F
import devtorch


class ContrastResponseDataset(torch.utils.data.Dataset):

    def __init__(self, c, y):
        self.c = c
        self.y = y

    def __getitem__(self, i):
        return self.c, self.y

    def __len__(self):
        return 1


class HyperbolicFunction(devtorch.DevModel):

    def __init__(self):
        super().__init__()
        self.r_max = nn.Parameter(torch.Tensor([1]))
        self.n = nn.Parameter(torch.Tensor([5]))
        self.c50 = nn.Parameter(torch.Tensor([10]))

    def forward(self, c):
        return self.r_max * (c ** self.n) / (self.c50 ** self.n + c ** self.n)


class Trainer(devtorch.Trainer):

    def __init__(self, model, train_dataset):
        super().__init__(model=model, train_dataset=train_dataset, n_epochs=600, batch_size=1, lr=0.1, optimizer_func=torch.optim.Adam, device="cpu")

    def loss(self, output, target, model):
        return F.mse_loss(output, target, reduction="mean")

    def train(self, save=True):
        super().train(save)


def get_fitted_hyperbolic_function(c, y):
    train_dataset = ContrastResponseDataset(c, y)
    hyperbolic_func = HyperbolicFunction()
    train = Trainer(hyperbolic_func, train_dataset)
    train.train(False)

    with torch.no_grad():
        c = torch.linspace(train_dataset.c.min(), train_dataset.c.max(), 400)
        pred_y = hyperbolic_func(c)

    return c, pred_y
