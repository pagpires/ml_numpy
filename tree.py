from util import iris, score
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# TODO: gini vs entropy
# mode = 'dev'
mode = 'max'
# mode = 'pos'

class Node(object):
    def __init__(self, cut=None, label=None):
        self.cut = cut
        self.leftLabel = None
        self.rightLabel = None
        self.left = None
        self.right = None
        if cut != (-1, -1):
            assert(len(label)==2), "Should contain left and right label"
            self.leftLabel = label['left']
            self.rightLabel = label['right']

def entropy(y):
    r = 0
    for v in np.unique(y):
        p = (y==v).sum() / y.shape[0]
        r += -p * np.log(p)
    return r

def get_prob(y):
    prob_dict = {}
    for c in np.unique(y):
        prob_dict[c] = (y==c).sum() / y.shape[0]
    return prob_dict

def info_gain(X, y, key, val):
    old_entropy = entropy(y)
    # note key is the 2nd dim
    idx = X[:, key] <= val
    left = y[idx]
    right = y[~idx]
    new_entropy = entropy(left) + entropy(right)
    ig = old_entropy - new_entropy
    label = {
        'left' : get_prob(left), 
        'right' : get_prob(right)
    }
    return ig, label

def best_split(X, y):
    best_ig = -np.inf
    best_cut = None
    best_label = None
    # igs = {}
    assert(len(np.unique(y))>1), "no need to split for pure y"
    for i in range(X.shape[1]):
        counts, bins = np.histogram(X[:, i])
        vals = np.array([np.mean([bins[j+1], bins[j]]) for j in range(len(bins)-1)])
        idx = counts > 0
        candidate_cut = vals[idx]
        for c in candidate_cut:
            ig, label = info_gain(X, y, i, c)
            # igs[(i,c)] = ig
            if ig > best_ig:
                best_cut = (i, c)
                best_ig = ig
                best_label = label
    
    if best_ig < 0:
        # test 1. always positive ig; 2. largest ig until pure
        if mode == 'dev':
            assert(-1==0), "stop"
        elif mode == 'pos':
            # print(igs)
            best_cut = (-1, -1)
        elif mode == 'max':
            pass
        else:
            raise ValueError
        # assert(best_cut is not None), "Didn't get a cut."
    return (best_cut, best_label)

def subtree(X, y):
    if (len(np.unique(y))==1):
        new_node = Node((-1, -1))
        return new_node

    elif (len(np.unique(y))>1):
        output = best_split(X, y)
        best_cut, best_label = output
        k, v = best_cut
        new_node = Node(cut=best_cut, label=best_label)
        # what if no ig even y types > 1:
        if k== -1:
            return new_node
        idx = X[:, k] <= v
        left_X = X[idx]
        left_y = y[idx]
        right_X = X[~idx]
        right_y = y[~idx]
        new_node.left = subtree(left_X, left_y)
        new_node.right = subtree(right_X, right_y)
        return new_node
    
    else:
        raise ValueError

def single_pred(tree, datum):
    node = tree
    while(node.cut!=(-1, -1)):
        parent = node
        k, v = node.cut
        if datum[k] <= v:
            node = parent.left
            label = parent.leftLabel
        else:
            node = parent.right
            label = parent.rightLabel
        assert(node is not None), 'Get None node'
    best_prob = -1
    # print(parent.cut, parent.left.cut, parent.right.cut, parent.leftLabel, parent.rightLabel)
    for c, prob in label.items():
        if prob > best_prob:
            best_c = c
            best_prob = prob
    return best_c

def predict(tree, X):
    # input feature batch, output label batch
    ypred = []
    for x in X:
        ypred.append(single_pred(tree, x))
    assert(len(ypred)==len(X)), "output length should be same as input rows"
    return ypred

if __name__ == "__main__":
    X, y, name = iris['data'], iris['target'], iris['target_names']

    # sklearn implementation
    clf = DecisionTreeClassifier(criterion='entropy')
    clf.fit(X, y)
    ypred = clf.predict(X)
    acc = score(ypred, y)
    print(f'sklearn implementation: acc={acc}')

    # self implementation
    
    root = subtree(X, y)
    if mode == 'dev':
        traverse = [root]
        i = 0
        while(i<len(traverse)):
            if (traverse[i] is None):
                i += 1
                continue
            node = traverse[i]
            print(node.cut, '->')
            traverse.append(node.left)
            traverse.append(node.right)
            i += 1
    else:
        ypred = predict(root, X)
        acc = score(ypred, y)
        print(f'self implementation: acc={acc}')

