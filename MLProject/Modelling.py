import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load data preprocessing
df = pd.read_csv('Membangun_model/creditcard_preprocessing/credit_card_scaled.csv')

X = df.values

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Credit Card Clustering")

with mlflow.start_run():
    mlflow.sklearn.autolog()

    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X)

    labels = kmeans.labels_
    silhouette = silhouette_score(X, labels)

    print("Silhouette Score:", silhouette)
