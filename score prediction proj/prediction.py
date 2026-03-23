import pandas as pd 
from sklearn.linear_model import LinerRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

data = pd.read_csv('data.csv')

x = data[['Hours']]
y = data['Scores']

model = LinearRegression()
model.fit(x, y)

predicted_score = model.predict(x)

mae = mean_absolute_error(y, predicted_score)
mse = mean_squared_error(y, predicted_score)
rmse = np.sqrt(mse)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')

new_prediction = model.predict([[9]])
print(f'Predicted score for 9 hours of study: {new_prediction[0]}')