import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load data
df = pd.read_csv("credit_card_scaled.csv")
X = df.values

mlflow.set_experiment("Credit Card Clustering")
mlflow.sklearn.autolog()

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

labels = kmeans.labels_
silhouette = silhouette_score(X, labels)

mlflow.log_metric("silhouette_score", silhouette)
print("Silhouette Score:", silhouette)
