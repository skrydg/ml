import pickle

def apply_to_train(files, f):
    for file in files:
        train = pickle.load(open(file, 'rb'))
        yield f(train)