import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
import joblib

# Load the data
train_df = pd.read_csv('restaurant_train(1).csv')
valid_df = pd.read_csv('restaurant_valid(1).csv')

print("Training data columns:", train_df.columns.tolist())
print("Validation data columns:", valid_df.columns.tolist())

# Define target and feature columns
target_column = 'Wait'

# Split features and target
X_train = train_df.drop(target_column, axis=1)
y_train = train_df[target_column]
X_valid = valid_df.drop(target_column, axis=1)
y_valid = valid_df[target_column]

# Define feature columns
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Define preprocessing steps
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define models
models = [
    ('logreg', LogisticRegression(max_iter=1000, random_state=42)),
    ('tree', DecisionTreeClassifier(random_state=42)),
    ('forest', RandomForestClassifier(random_state=42)),
    ('gbc', GradientBoostingClassifier(random_state=42))
]

# Train and evaluate models
for name, model in models:
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    print(f"\nModel: {name}")
    print(f"Accuracy: {accuracy_score(y_valid, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_valid, y_pred, average='weighted'):.4f}")
    print("Classification Report:")
    print(classification_report(y_valid, y_pred))

# Hyperparameter tuning for Random Forest
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

best_clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))])
grid_search = GridSearchCV(best_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\nBest Parameters for Random Forest:")
print(grid_search.best_params_)
print("Best Score for Random Forest:")
print(f"{grid_search.best_score_:.4f}")

# Evaluate best model on validation set
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_valid)
print("\nBest Model Performance on Validation Set:")
print(f"Accuracy: {accuracy_score(y_valid, y_pred_best):.4f}")
print(f"F1 Score: {f1_score(y_valid, y_pred_best, average='weighted'):.4f}")
print("Classification Report:")
print(classification_report(y_valid, y_pred_best))

# Save the best model
joblib.dump(best_model, 'best_model_pipeline.pkl')
print("\nBest model saved as 'best_model_pipeline.pkl'")