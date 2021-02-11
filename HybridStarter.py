from MovieLens_preProcessing import MovieLens_preProcessing
from RBMAlgorithm import RBMAlgorithm
from HybridAlgorithm import HybridAlgorithm
from Evaluator import Evaluator
from surprise import SVD
from surprise.model_selection import GridSearchCV

import random
import numpy as np



def LoadMovieLensData():
    ml = MovieLens_preProcessing()
    print("Loading movie ratings...")
    data = ml.loadMovieLensLatestSmall()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

np.random.seed(0)
random.seed(0)


(ml, evaluationData, rankings) = LoadMovieLensData()


evaluator = Evaluator(evaluationData, rankings)


SimpleRBM = RBMAlgorithm(epochs=40)


print("Searching for best parameters...")
param_grid = {'n_epochs': [20, 30], 'lr_all': [0.005, 0.010],
              'n_factors': [50, 100]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)

gs.fit(evaluationData)


print("Best RMSE score attained: ", gs.best_score['rmse'])


print(gs.best_params['rmse'])


evaluator = Evaluator(evaluationData, rankings)

params = gs.best_params['rmse']
SVDtuned = SVD(n_epochs = params['n_epochs'], lr_all = params['lr_all'], n_factors = params['n_factors'])


Hybrid = HybridAlgorithm([SimpleRBM, SVDtuned], [0.5, 0.5])

evaluator.AddAlgorithm(SimpleRBM, "RBM")
evaluator.AddAlgorithm(SVDtuned, "SVD - Tuned")
evaluator.AddAlgorithm(Hybrid, "Hybrid")

evaluator.Evaluate(False)

evaluator.SampleTopNRecs(ml)
