import json, argparse, os
import numpy as np
import matplotlib.pyplot as plt

from models import *
from experimentations import MonitoredExperimentation
from utils.data import H5Dataset
from optimization.loss import dice
from tqdm import tqdm
import SimpleITK as sitk

if __name__ == "__main__":

    np.random.seed(1234)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--image_indice', type=int, required=True)
    parser.add_argument('--param_file', type=str, required=True)
    args = parser.parse_args()

    with open(args.param_file, 'r') as j:
        parameters = json.load(j)

    # DATA
    dataset = H5Dataset(args.dataset_path)
    if args.image_indice > len(dataset):
        raise ValueError('Image with indice {} does not exist.'.format(args.image_indice))
    
    X, Y = dataset[args.image_indice][0], dataset[args.image_indice][1]

    # MODEL
    model = UnitLinkingPCNN
    expe = MonitoredExperimentation(model, parameters[model.__name__])
    expe.run(X, Y, dice)
    
    plt.imshow(expe.best_maps[0])
    plt.show()