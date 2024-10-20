from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
import numpy as np
import joblib

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Save t-SNE transformed data and labels
np.save('data/X_tsne.npy', X_tsne)
np.save('data/y.npy', y)

# Save the t-SNE object
joblib.dump(tsne, 'models/tsne_model.joblib')