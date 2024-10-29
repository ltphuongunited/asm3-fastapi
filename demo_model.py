import pickle
from sklearn.linear_model import LinearRegression

# Sample training data
X_train = [[1], [2], [3]]
y_train = [2, 4, 6]

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
with open("model/regression_model.pkl", "wb") as f:
    pickle.dump(model, f)