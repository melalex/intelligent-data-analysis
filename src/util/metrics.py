from sklearn import metrics


def calculate_regression_metrics(y_test, y_pred):
    return {
        "r2": metrics.r2_score(y_test, y_pred),
        "mae": metrics.mean_absolute_error(y_test, y_pred),
        "mse": metrics.mean_squared_error(y_test, y_pred),
    }


def calculate_classification_metrics(y_test, y_pred):
    return {
        "f1": metrics.f1_score(y_test, y_pred, average="weighted"),
        "accuracy": metrics.accuracy_score(y_test, y_pred),
        "recall": metrics.recall_score(y_test, y_pred, average="weighted"),
    }
