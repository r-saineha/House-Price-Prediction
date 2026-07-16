import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#Joblib is a popular Python library used to provide lightweight, efficient pipelining and parallel computing. It is highly optimized for handling large numpy arrays and is a core dependency of scikit-lear
# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data (1).csv")

print("Dataset Loaded Successfully")
print(df.head())

# -----------------------------
# Convert Date
# -----------------------------
df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day

df.drop("date", axis=1, inplace=True)

# -----------------------------
# Target Variable
# -----------------------------
y = df["price"]

# Features
X = df.drop("price", axis=1)

# -----------------------------
# Numerical & Categorical Columns
# -----------------------------
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns

categorical_features = X.select_dtypes(include=["object"]).columns

print("\nNumeric Columns")
print(numeric_features)

print("\nCategorical Columns")
print(categorical_features)

# -----------------------------
# Preprocessing
# -----------------------------
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# -----------------------------
# Model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# -----------------------------
# Pipeline
# -----------------------------
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# -----------------------------
# Train
# -----------------------------
print("\nTraining Model...\n")

pipeline.fit(X_train, y_train)

print("Training Complete!")

# -----------------------------
# Prediction
# -----------------------------
pred = pipeline.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("\nModel Performance")
print("-----------------------------")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(pipeline, "house_price_model.pkl")

print("\nModel Saved Successfully!")

# -----------------------------
# Feature Importance
# -----------------------------
feature_names = pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()

importance = pipeline.named_steps[
    "model"
].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 Important Features\n")
print(importance_df.head(20))

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10,8))

plt.barh(
    importance_df["Feature"].head(20),
    importance_df["Importance"].head(20)
)

plt.gca().invert_yaxis()

plt.title("Top 20 Important Features")
plt.xlabel("Importance")

plt.tight_layout()

plt.show()


sns.set_style("whitegrid")

print(df.info())

print(df.describe())

print(df.isnull().sum())

# -----------------------------
# Price Distribution
# -----------------------------
plt.figure(figsize=(8,5))
sns.histplot(df["price"], bins=30, kde=True)
plt.title("House Price Distribution")
plt.show()

# -----------------------------
# Correlation Heatmap
# -----------------------------
plt.figure(figsize=(12,8))

numeric_df = df.select_dtypes(include=["int64","float64"])

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# Bedrooms vs Price
# -----------------------------
plt.figure(figsize=(8,5))

sns.boxplot(
    x="bedrooms",
    y="price",
    data=df
)

plt.title("Bedrooms vs Price")
plt.show()

# -----------------------------
# Bathrooms vs Price
# -----------------------------
plt.figure(figsize=(8,5))

sns.scatterplot(
    x="bathrooms",
    y="price",
    data=df
)

plt.title("Bathrooms vs Price")
plt.show()

# -----------------------------
# Living Area vs Price
# -----------------------------
plt.figure(figsize=(8,5))

sns.scatterplot(
    x="sqft_living",
    y="price",
    data=df
)

plt.title("Living Area vs Price")
plt.show()

# -----------------------------
# House Condition
# -----------------------------
plt.figure(figsize=(8,5))

sns.countplot(
    x="condition",
    data=df
)

plt.title("House Condition")
plt.show()

# -----------------------------
# Waterfront
# -----------------------------
plt.figure(figsize=(5,5))

df["waterfront"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.ylabel("")
plt.title("Waterfront Houses")
plt.show()

# -----------------------------
# Top 10 Cities
# -----------------------------
plt.figure(figsize=(10,5))

df["city"].value_counts().head(10).plot(
    kind="bar"
)

plt.title("Top 10 Cities")
plt.xlabel("City")
plt.ylabel("Number of Houses")
plt.show()
