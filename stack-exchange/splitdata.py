import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('date_update.csv')
X = data[['text', 'intent']]
y = data[['intentId']]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20)
train_df = X_train.join(y_train)
test_df = X_test.join(y_test)

train_df.to_csv("train-stackexchange.csv")
test_df.to_csv("test-stackexchange.csv")
