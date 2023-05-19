from robustbench.utils import load_model
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Cifar10Model:
    def __init__(self, idx):
        # assert type(idx) is int and 2 > idx > -1
        self.models = ["Gowal2021Improving_28_10_ddpm_100m",
                       "Gowal2020Uncovering_28_10_extra", "Standard"]
        self.l_norm = ["Linf", "Linf", "Linf"]
        self.model = load_model(model_name=self.models[idx], dataset="cifar10", threat_model=self.l_norm[idx])
        self.model.to(device)
        self.model.eval()

        self.max = 0
        self.counter = 0

    def enough(self):
        return True if self.counter >= self.max else False

    def set_maximum(self, max):
        self.max = max

    def predict(self, x):
        return self.model(x)

    def __call__(self, x):
        self.counter += 1
        return self.model(x)

    def zero_grad(self):
        self.model.zero_grad()


class RNDCifar10Model:
    def __init__(self, idx):
        self.model = Cifar10Model(idx)
        self.v = 0.02

    def predict(self, x):
        # add noise to the image
        x_ = x + torch.normal(torch.zeros_like(x), 1.) * self.v
        return self.model.predict(x_)

