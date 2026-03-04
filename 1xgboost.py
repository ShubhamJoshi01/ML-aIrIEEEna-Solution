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

# initialize XGBoost classifier
xgb = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

# train XGBoost model
xgb.fit(X_train, y_train)

# predict validation data
y_pred = xgb.predict(X_test)

# print XGBoost accuracy
print("Acuuracy:", accuracy_score(y_test,y_pred))

# print classification report
print(classification_report(y_test, y_pred))


# save trained model for reuse
joblib.dump(xgb, "trained_model.pkl")


# load test dataset for prediction
test_df = pd.read_csv(r"alrIEEEna_26_dataset\ML Challenge Dataset\TEST.csv")

# store ID column separately
test_ids = test_df["ID"]

# remove ID column from features
X_test_final = test_df.drop("ID", axis=1)


# generate predictions for test dataset
test_predictions = xgb.predict(X_test_final)


# create submission dataframe
submission = pd.DataFrame({
    "ID": test_ids,
    "CLASS": test_predictions
})

# save final predictions file
submission.to_csv("FINAL.csv", index=False)