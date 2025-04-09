# 🌟 End-to-End Machine Learning Project Tutorial

Welcome to your all-in-one ML project setup! This guide walks you through building, training, and deploying a simple machine learning model using **Poetry**, **pip+venv**, **Docker**, **CLI**, **FastAPI**, and GitHub integration. Let’s go! 🚀

---

## 📌 1. Introduction

This tutorial guides you through building a simple machine learning project using both **Poetry** and **pip + venv**.  
You’ll learn how to structure your project, install dependencies, train a model, run scripts, deploy using Docker, create a CLI or API, and prepare for GitHub.

---

## 🎯 2. Project Goal

Build and deploy a **Logistic Regression** model trained on the classic Iris dataset using `scikit-learn`.

---

## 🧪 3. Poetry Workflow

### ✅ Create the project

```bash
poetry new iris-classifier
cd iris-classifier
```

### ✅ Add dependencies

```bash
poetry add scikit-learn pandas
```

### ✅ Add `train.py`

Create a file: `iris_classifier/train.py`

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

def train():
    iris = load_iris()
    model = LogisticRegression(max_iter=200)
    model.fit(iris.data, iris.target)
    print("Accuracy:", model.score(iris.data, iris.target))

if __name__ == "__main__":
    train()
```

### ▶️ Run it

```bash
poetry run train.py
```

### 📦 Build and Publish

```bash
poetry build
poetry publish --build  # Requires PyPI token
```

---

## ⚙️ 4. pip + venv Workflow

### ✅ Set up

```bash
mkdir iris-classifier
cd iris-classifier
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### ✅ Files

Create `train.py` and `requirements.txt`

**train.py** (same code as above)

**requirements.txt**

```
scikit-learn
pandas
```

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### ▶️ Run

```bash
python train.py
```

---

## 🐳 5. Dockerfile for Deployment

Create a file: `Dockerfile`

```Dockerfile
FROM python:3.13
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "train.py"]
```

### 🔨 Build and run

```bash
docker build -t iris-classifier .
docker run iris-classifier
```

---

## 🧰 6. Turn Into CLI Tool

Enhance `train.py` with CLI support:

```python
import argparse

def train(max_iter=200):
    # training code

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_iter', type=int, default=200)
    args = parser.parse_args()
    train(args.max_iter)
```

Run it like:

```bash
python train.py --max_iter 300
```

---

## 🌐 7. Turn Into an API (FastAPI)

### ✅ Add FastAPI

```bash
poetry add fastapi uvicorn
```

### ✅ Create `api.py`

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/train")
def train_model():
    # training code
    return {"status": "Model trained"}
```

### ▶️ Run API

```bash
poetry run uvicorn api:app --reload
```

---

## 📁 8. GitHub Project Scaffold

Your final structure:

```
iris-classifier/
├── .gitignore
├── Dockerfile
├── requirements.txt or pyproject.toml
├── train.py
├── api.py (optional)
├── iris_classifier/
│   └── __init__.py
├── tests/
│   └── test_iris_classifier.py
README.md
```

---

## 🏁 9. Final Notes

This guide takes you from zero to deployment.  
You’ve built a project that can be run locally, packaged, deployed in Docker, turned into a CLI or API, and prepped for GitHub.  
Now go share your model with the world! 🌍✨

---
