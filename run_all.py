import json, argparse, os
import numpy as np
from models import *
from utils.data import H5Dataset
from experimentations import MonitoredExperimentation
from optimization.loss import dice
from tqdm import tqdm

if __name__ == "__main__":

    np.random.seed(1234)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--param_file', type=str, required=True)
    args = parser.parse_args()

    with open(args.param_file, 'r') as j:
        parameters = json.load(j)
    with open('results.json', 'r') as f:
        results = json.load(f)

    models = [FLSCM]

    for mod in models:
        sim = MonitoredExperimentation(mod, parameters[mod.__name__])
        dataset = H5Dataset(os.path.join(args.dataset_path))
        with tqdm(total=(len(dataset)), desc='Evaluating slices :') as pbar:
            for idx, (X, Y) in enumerate(dataset):
                sim.run(X, Y, dice)
                pbar.set_description('Average Dice Score : {}'.format(np.mean(sim.best_scores)))
                pbar.update()
            results[mod.__name__][seq] = np.mean(sim.best_scores)

    print(json.dumps(results, sort_keys=True, indent=4))
    with open('results.json', 'w') as f:
        json.dump(results, f)