# Import necessary libraries and modules
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.spatial.distance import cdist
import joblib

# Fetch the heart disease dataset from UCI ML repository 
heart_disease = fetch_ucirepo(id=45) 
  
# Extract features (X) and target values (y) from the dataset
X = heart_disease.data.features 
y = heart_disease.data.targets 

# Create a DataFrame from the features 
df = pd.DataFrame(X, columns=heart_disease.data.feature_names)

# Rename columns for better understanding
df = df.rename(columns={
    'cp': 'chest_pain_type', 
    'trestbps': 'resting_blood_pressure',
    'chol': 'cholestrol',
    'fbs': 'fasting_blood_sugar',
    'restecg': 'resting_electrocardiograh',
    'thalach': 'max_heart_rate',
    'exang': 'exercise_induced_angina',
    'oldpeak': 'st_depression_exercise',
    'slope': 'peak_exercise_slope',
    'ca': 'major_vessels',
    'thal': 'thalassemia'
})

# Drop rows with missing values from df, X, and y
df = df.dropna()
X = X.dropna()
y = y.dropna()

# Specify categorical feature columns for One-Hot Encoding
categorical_cols = [
    'chest_pain_type', 
    'resting_electrocardiograh', 
    'exercise_induced_angina', 
    'peak_exercise_slope', 
    'thalassemia', 
    'sex', 
    'fasting_blood_sugar', 
    'major_vessels'
]

# Create a preprocessing pipeline that applies One-Hot Encoding to categorical columns
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(drop='first'), categorical_cols)],
    remainder='passthrough'
)

# Fit and transform the data in df using the preprocessor
X_processed = preprocessor.fit_transform(df)

# Standardize all features for K-means clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_processed)

# Use the Elbow Method to determine the optimal number of clusters
distortions = []
inertias = []
mapping1 = {}  # Mapping for distortion values per k
mapping2 = {}  # Mapping for inertia values per k

# Loop over a range of possible cluster values (1 to 10)
for k in range(1, 11):
    kmeanModel = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
    # Calculate distortion as the average squared distance from each point to its nearest cluster center
    distortions.append(sum(np.min(cdist(X_scaled, kmeanModel.cluster_centers_, 'euclidean'), axis=1)**2) / X_scaled.shape[0])
    
    # Inertia is computed directly by KMeans
    inertias.append(kmeanModel.inertia_)
    
    # Store values for later display
    mapping1[k] = distortions[-1]
    mapping2[k] = inertias[-1]

# Print shapes of the processed feature set and target
print(f"X_scaled shape: {X_scaled.shape}")
print(f"y shape: {y.shape if hasattr(y, 'shape') else len(y)}")

# Print distortion values for each k
print("Distortion values:")
for key, val in mapping1.items():
    print(f'{key} : {val}')

# Plot distortions to visualize the "elbow"
plt.plot(range(1, 11), distortions, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.grid()
plt.show()

# Print inertia values for each k
print("Inertia values:")
for key, val in mapping2.items():
    print(f'{key} : {val}')

# Plot inertias to visualize the "elbow"
plt.plot(range(1, 11), inertias, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.grid()
plt.show()

# Determine the number of unique target values (clusters expected)
unique_targets = np.unique(y)
num_clusters = len(unique_targets)
print(f"Number of unique target values: {num_clusters}")

# Perform K-means clustering using the determined number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Create a copy of the original DataFrame and include the cluster assignments
df_clustered = df.copy()
df_clustered['cluster'] = y_kmeans

# Define feature pairs for visualization of clusters
feature_pairs = [
    ['age', 'max_heart_rate'],
    ['cholestrol', 'resting_blood_pressure']
]

# Plot clusters for each feature pair
for features in feature_pairs:
    plt.figure(figsize=(10, 8))
    # For each cluster, scatter plot the data points
    for cluster in range(num_clusters):
        cluster_data = df_clustered[df_clustered['cluster'] == cluster]
        plt.scatter(
            cluster_data[features[0]], 
            cluster_data[features[1]],
            label=f'Cluster {cluster}',
            alpha=0.7
        )
    plt.title(f'K-means Clustering (k={num_clusters}) based on {features[0]} and {features[1]}')
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# If target data (y) exists, compare clusters to the actual target values
if y is not None:
    # Create a DataFrame for the actual target values
    target_df = pd.DataFrame({'Actual': y.values.ravel() if hasattr(y, 'values') else y})
    # Create a DataFrame for the cluster assignments with matching indices
    cluster_df = pd.DataFrame({'Cluster': y_kmeans}, index=df.index)
    # Merge the cluster assignments with the actual target values
    comparison_df = pd.concat([cluster_df, target_df], axis=1, join='inner')
    
    print(f"Cluster array length: {len(y_kmeans)}")
    print(f"Target array length: {len(target_df)}")
    print(f"Comparison dataframe length: {len(comparison_df)}")
    
    # Plot a heatmap to visualize the relationship between clusters and actual target classes
    plt.figure(figsize=(10, 8))
    confusion = pd.crosstab(comparison_df['Cluster'], comparison_df['Actual'])
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
    plt.title('Comparison of Clusters to Actual Target Values')
    plt.xlabel('Actual Target Value')
    plt.ylabel('Cluster')
    plt.show()
    
# Prepare data for Random Forest Classifier by splitting into training and testing sets

# Set the target as the cluster assignments for classification
y = df_clustered['cluster']

# Split the standardized features and the cluster target into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize a Random Forest Classifier
rf = RandomForestClassifier(random_state=42)

# Set up the grid of hyperparameters to search
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],   # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],   # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],     # Minimum number of samples required at a leaf node
    'max_features': ['sqrt', 'log2', None],  # Number of features to consider at split time
    'bootstrap': [True, False]         # Use bootstrap samples when building trees
}

# Use GridSearchCV for exhaustive search over hyperparameter space
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Retrieve the best estimator and make predictions on the test set
rf_baba_pro = grid_search.best_estimator_
y_pred = rf_baba_pro.predict(X_test)

# Print classification report and accuracy
print("\nClassification Report:\n", classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# Plot heatmap of the confusion matrix with appropriate risk labels for clusters
plt.figure(figsize=(6, 4))
# Define labels based on the number of clusters (up to five risk levels)
risk_labels = ['None', 'Very Low', 'Low', 'Medium', 'High'][:num_clusters]
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt="d", 
           xticklabels=risk_labels, yticklabels=risk_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Heart Disease Risk Confusion Matrix")
plt.show()

# Save the scaler, best Random Forest model, and preprocessor for later use
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(rf_baba_pro, "models/best_rf_model.pkl")
joblib.dump(preprocessor, "models/preprocessor.pkl")
