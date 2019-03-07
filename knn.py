from util import iris, score
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# implement knn

def get_distance(v1, v2):
    return np.sqrt(np.sum((v1-v2)**2))

def get_knn(data, target, k):
    # data = [(x, y)]
    distance = [(y, get_distance(x, target)) for x,y in data]
    sort_dist = sorted(distance, key=lambda x: x[1])
    nn = sort_dist[:k]
    return nn

def get_label(nn):
    # nn = [(y, dist)]
    counts = {}
    for i, n in enumerate(nn):
        label = n[0]
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1
    m_count = 0
    for label, c in counts.items():
        if c > m_count:
            major_label = label
    return major_label

def knn(train, test):
    k = 5
    labels = []
    X, y = train
    data = list(zip(X, y))
    nns = []
    for i, x in enumerate(test):
        nn = get_knn(data, x, k)
        nns.append(nn)
        l = get_label(nn)
        labels.append(l)

    return np.array(labels), nns

if __name__ == "__main__":
    X, y, name = iris['data'], iris['target'], iris['target_names']

    # sklearn implementation
    clf = KNeighborsClassifier()
    clf.fit(X, y)
    ypred = clf.predict(X)
    acc = score(ypred, y)
    print(f'sklearn implementation: acc={acc}')

    # self implementation
    ypred, nns = knn((X, y), X)
    acc = score(ypred, y)
    print(f'self implementation: acc={acc}')