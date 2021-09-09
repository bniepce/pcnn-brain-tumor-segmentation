from models import *
from utils.data import H5Dataset
from optimization import DE
import argparse

if __name__ == "__main__":

    np.random.seed(1234)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    args = parser.parse_args()

    models = [UnitLinkingPCNN, FLSCM, StandardPCNN]
    de = DE(model)
    
    dataset = H5Dataset(args.dataset_path)
    X, Y = dataset[0][0], dataset[0][1]
    for i in models:
        opt_params = de.run((X, Y))
        print(i.__name__)
        print(opt_params)
