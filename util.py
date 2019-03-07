import pandas as pd
import numpy as np
from sklearn import datasets
data = pd.DataFrame({"toothed":["True","True","True","False","True","True","True","True","True","False"],
                     "hair":["True","True","False","True","True","True","False","False","True","False"],
                     "breathes":["True","True","True","True","True","True","False","True","True","True"],
                     "legs":["True","True","False","True","True","True","False","False","True","True"],
                     "species":["Mammal","Mammal","Reptile","Mammal","Mammal","Mammal","Reptile","Reptile","Mammal","Reptile"]}, 
                    columns=["toothed","hair","breathes","legs","species"])
features = data[["toothed","hair","breathes","legs"]]
target = data["species"]

iris = datasets.load_iris()

# metric
def score(ypred, ytrue):
    assert(ypred.shape==ytrue.shape)
    ypred = np.array(ypred)
    ytrue = np.array(ytrue)
    return (ypred==ytrue).sum() / ypred.shape[0]

