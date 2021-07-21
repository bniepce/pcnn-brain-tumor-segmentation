from models import *
from utils.data import H5Dataset
from optimization import DE

if __name__ == "__main__":

    np.random.seed(1234)

    model = UnitLinkingPCNN
    de = DE(model)
    
    dataset = H5Dataset('mri_data.h5')
    X, Y = dataset[0][0], dataset[0][1]
    opt_params = de.run((X, Y))
    print(opt_params)