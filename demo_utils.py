# --------- To load demo data --------------
from torchvision import datasets, transforms

# --------- Torch libs ---------------------
from models.torch_model import TorchModel
import models.resnet as resnet
import torch
import torch.nn.functional as F
import torch.nn as nn

# --------- SciPy libs ---------------------
from sklearn.datasets import load_digits
from sklearn import tree

import numpy as np


TORCH_WEIGHTS = "models/pytorch_resnet_cifar10-master/pretrained_models/resnet20-12fca82f.th"
CIFAR_NORM = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

def make_cifar_dataset():
	transform = transforms.Compose([
		transforms.ToTensor(),
     	transforms.Normalize(*CIFAR_NORM)])
	testset = datasets.CIFAR10(root='data', train=False,
		download=True, transform=transform)
	return testset

def make_digit_dataset(flatten=True):
	data, target = load_digits(return_X_y=True)
	data = data.reshape(len(data), -1) / 255.
	return data, target

def create_torch_model(device):
	model = resnet.resnet20()
	weights = torch.load(TORCH_WEIGHTS, map_location=device)
	model = nn.DataParallel(model)
	model.load_state_dict(weights['state_dict'])
	model = model.module
	model.eval()
	model.to(device)
	print('Created PyTorch model:\t', model._get_name())
	print(' * Best Test prec:\t', weights['best_prec1'])
	return model

def create_decision_tree(train_x, train_y, max_depth=8):
	d_tree = tree.DecisionTreeClassifier(max_depth=max_depth)
	d_tree = d_tree.fit(train_x, train_y)
	test_score = d_tree.score(train_x, train_y)
	print('Created decision tree')
	print(' * Depth:\t', d_tree.get_depth())
	print(' * Test score:\t', test_score)
	return d_tree