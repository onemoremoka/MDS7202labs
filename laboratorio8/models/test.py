import pickle

P = "/home/marcosignacio/desktop/repos/MDS7202labs/laboratorio8/apilab/models/best_model.pkl"
with open(P, "rb") as f:
        model = pickle.load(f)


x = model.predict([[1.0, 100.0, 1000.0, 5.0, 200.0, 500.0, 10.0, 50.0, 1.0]])
print(x)