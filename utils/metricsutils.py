import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def calculate_metrics(model, test_generator, year, scaler_o3):
    rmse = 0
    mae = 0

    for x, y in test_generator:
        predictions = model.predict(x)
        predictions = scaler_o3.inverse_transform(predictions.squeeze())
        y = scaler_o3.inverse_transform(y.squeeze())

        rmse += np.sqrt(np.mean((predictions - y) ** 2))
        mae += mean_absolute_error(y, predictions)
    rmse = rmse / len(test_generator)
    mae = mae / len(test_generator)
    results = pd.DataFrame([[year, rmse, mae]], columns=['YEAR', 'RMSE', 'MAE'])
    results.set_index('YEAR')
    return results
