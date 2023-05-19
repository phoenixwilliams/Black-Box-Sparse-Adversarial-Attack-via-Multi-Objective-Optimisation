from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input as mn_pi

from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input as rn_pi

from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.applications.nasnet import preprocess_input as nn_pi

import numpy as np

from robustbench.utils import load_model
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ImageNetModel:
    def __init__(self, idx: int):
        assert type(idx) is int and 4 > idx > -1

        self.models = [MobileNet, ResNet50V2, NASNetMobile]
        self.preprocesses = [mn_pi, rn_pi, nn_pi]

        self.model = self.models[idx](weights='imagenet')
        self.preprocess_input = self.preprocesses[idx]

    def predict(self, x):
        # y = x.permute(1, 2, 0).numpy()
        # y = np.expand_dims(x, axis=0)
        y = self.preprocess_input(x)
        pred = self.model.predict(y, verbose=0)
        return pred


class RNDImageNet:
    def __init__(self, idx):
        self.model = ImageNetModel(idx)
        self.v = 0.02

    def predict(self, x):
        x_ = x + np.random.normal(0, 1, size=x.shape) * self.v
        return self.model.predict(x_)




from robustbench.data import get_preprocessing
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from torchvision import transforms


class AdversarialImageNetModel:
    def __init__(self, idx: int):
        self.models = ["Salman2020Do_50_2", "Salman2020Do_R50"]
        self.l_norm = ["Linf", "Linf", "Linf"]

        self.model = load_model(model_name=self.models[idx], dataset="imagenet", threat_model=self.l_norm[idx])
        self.model.to(device)
        self.model.eval()

        dataset_: BenchmarkDataset = BenchmarkDataset("imagenet")
        threat_model_: ThreatModel = ThreatModel("Linf")

        #self.preprocess_input = get_preprocessing(dataset_, threat_model_, self.models[idx], preprocessing=None)
        self.preprocess_input = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
        #print(prepr)

    def predict(self, x):
        # y = x.permute(1, 2, 0).numpy()
        # y = np.expand_dims(x, axis=0)
        #y = self.preprocess_input(x)
        pred = self.model(x)
        return pred

    def __call__(self, x):
        #self.counter += 1
        return self.model(x)
