from .loss import dice
from scipy.optimize import differential_evolution
from models import UnitLinkingPCNN


# TODO: Adapt to any model and any parameter dict
class DE(object):

    def __init__(self, model):
        self.model = model
    
    def fitness_func(self, x, *args):
        image, target = args
        params = x
        parameters = {
            "v_t": 20.0,
            "a_t": params[0],
            "beta": params[1],
            "k_size": int(params[2])
        }
        model = UnitLinkingPCNN(image, parameters)
        score = 0
        for i in range(20):
            pred = model.do_iteration()
            score = dice(target, pred)
        return 1 - score
        
    def run(self, data, bounds=[(0.01, 0.1), (0, 1.0), (3, 9)]):
        result = differential_evolution(self.fitness_func, bounds, args=data, disp=True)
        return result