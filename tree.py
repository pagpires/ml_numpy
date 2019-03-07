from util import iris, score
import numpy as np

if __name__ == "__main__":
    X, y, name = iris['data'], iris['target'], iris['target_names']

    # sklearn implementation
    clf.fit(X, y)
    ypred = clf.predict(X)
    acc = score(ypred, y)
    print(f'sklearn implementation: acc={acc}')

    # self implementation
    
    print(f'self implementation: acc={acc}')