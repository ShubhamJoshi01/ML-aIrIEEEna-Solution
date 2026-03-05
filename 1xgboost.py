import pandas as pd

# load training dataset
df = pd.read_csv(r"alrIEEEna_26_dataset\ML Challenge Dataset\TRAIN.csv")

print(df)

df.info()

df.head()


#importing neccesary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# separate features and target column
x=df.drop("Class", axis=1)
y=df["Class"]

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# perform PCA for visualization
pca = PCA(n_components=4)
X_pca = pca.fit_transform(x)

# plot PCA projection
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, alpha=0.5)
plt.show()

# split data into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)

# initialize random forest model
rf=RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42)

# train random forest model
rf.fit(X_train, y_train)

# predict on test split
y_pred=rf.predict(X_test)

# print random forest accuracy
print("Accuracy:",accuracy_score(y_test,y_pred))

# print classification metrics
print(classification_report(y_test,y_pred))

# compute feature importance
importance = pd.Series(rf.feature_importances_, index=x.columns)
print(importance.sort_values(ascending=False))

import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import joblib
from sklearn.model_selection import GridSearchCV

# visualize feature distributions
plt.figure(figsize=(12,6))
sns.boxplot(data=x)
plt.xticks(rotation=90)
plt.show()

# check class distribution across feature bins
for col in x.columns:
    print(col)
    print(pd.crosstab(pd.cut(x[col], bins=5), y))


# split data again for XGBoost training
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# initialize XGBoost classifier (base model)
xgb = XGBClassifier(
    random_state=42,
    eval_metric='logloss'
)
parameter_grid={
    "n_estimators":[200,400,600],
    "max_depth":[4,6,8],
    "learning_rate":[0.01,0.05,0.1],
    "subsample": [0.7,0.8,0.9],
    "colsample_bytree":[0.7,0.8,0.9]
}

#performing cress validation gridsearch method
grid_search=GridSearchCV(
    estimator=xgb,
    param_grid=parameter_grid,
    scoring="f1",
    cv=3,
    verbose=1,
    n_jobs=-1
)

#train model with different parameters
grid_search.fit(X_train,y_train)

best_model=grid_search.best_estimator_

# train XGBoost model
best_model.fit(X_train, y_train)

# predict validation data
y_pred = best_model.predict(X_test)

# print XGBoost accuracy
print("Acuuracy:", accuracy_score(y_test,y_pred))

# print classification report
print(classification_report(y_test, y_pred))


# save trained model for reuse
joblib.dump(best_model, "trained_model.pkl")


# load test dataset for prediction
test_df = pd.read_csv(r"alrIEEEna_26_dataset\ML Challenge Dataset\TEST.csv")

# store ID column separately
test_ids = test_df["ID"]

# remove ID column from features
X_test_final = test_df.drop("ID", axis=1)


# generate predictions for test dataset
test_predictions = best_model.predict(X_test_final)


# create submission dataframe
submission = pd.DataFrame({
    "ID": test_ids,
    "CLASS": test_predictions
})

# save final predictions file
submission.to_csv("FINAL.csv", index=False)