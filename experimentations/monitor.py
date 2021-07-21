import numpy as np

class MonitoredExperimentation(object):
    
    def __init__(self, model, parameters):
        self.model = model
        self.parameters = parameters
        self.best_scores = []
        self.best_maps = []
    
    def run(self, x, y, score_function, early_stop=2):
        stop_counts = 0
        best_score = 0
        inter_maps = []
        model = self.model(S=x, parameters=self.parameters)
        while True:
            pred = model.do_iteration()
            current_score = score_function(y, pred.astype('uint8'))
            if current_score < best_score:
                stop_counts += 1
            else:
                best_score = current_score
                inter_maps.append(pred)
            if stop_counts == early_stop:
                break
        self.best_maps.append(inter_maps[-(early_stop-1)])
        self.best_scores.append(best_score)