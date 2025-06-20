
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv("data/bitcoin_price.csv")
X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(mean_squared_error(y_test, predictions))
print(r2_score(y_test, predictions))

plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:100])
plt.plot(predictions[:100])
plt.title('Bitcoin Price Prediction using SVR')
plt.xlabel('Index')
plt.ylabel('Price')
plt.legend(['Actual', 'Predicted'])
plt.show()
