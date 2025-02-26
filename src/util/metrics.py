from sklearn import metrics


def calculate_metrics(y_test, y_pred):
    return {
        "r2": metrics.r2_score(y_test, y_pred),
        "mae": metrics.mean_absolute_error(y_test, y_pred),
        "mse": metrics.mean_squared_error(y_test, y_pred),
    }
