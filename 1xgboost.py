import pandas as pd

df = pd.read_csv(r"alrIEEEna_26_dataset\ML Challenge Dataset\TRAIN.csv")

print(df)

df.info()

df.head()



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

x=df.drop("Class", axis=1)
y=df["Class"]

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=4)
X_pca = pca.fit_transform(x)

plt.scatter(X_pca[:,0], X_pca[:,1], c=y, alpha=0.5)
plt.show()

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)

rf=RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42)

rf.fit(X_train, y_train)

y_pred=rf.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))

# import pandas as pd
importance = pd.Series(rf.feature_importances_, index=x.columns)
print(importance.sort_values(ascending=False))

import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

plt.figure(figsize=(12,6))
sns.boxplot(data=x)
plt.xticks(rotation=90)
plt.show()

for col in x.columns:
    print(col)
    print(pd.crosstab(pd.cut(x[col], bins=5), y))


X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

xgb = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)

print("Acuuracy:", accuracy_score(y_test,y_pred))

print(classification_report(y_test, y_pred))



test_df = pd.read_csv(r"alrIEEEna_26_dataset\ML Challenge Dataset\TEST.csv")

test_ids = test_df["ID"]

X_test_final = test_df.drop("ID", axis=1)


test_predictions = xgb.predict(X_test_final)


submission = pd.DataFrame({
    "ID": test_ids,
    "CLASS": test_predictions
})

submission.to_csv("FINAL.csv", index=False)
