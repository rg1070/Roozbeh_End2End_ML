from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

def train():
    iris = load_iris()
    model = LogisticRegression(max_iter=200)
    model.fit(iris.data, iris.target)
    print("Accuracy:", model.score(iris.data, iris.target))

if __name__ == "__main__":
    train()