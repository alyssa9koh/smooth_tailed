# machine learning packages
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import os
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import json

# Preprocessing the data using sklearn
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader

# plotting packages
import matplotlib.pyplot as plt

# miscellaneous packages
import json
import datetime
from tqdm import tqdm, trange
from types import SimpleNamespace