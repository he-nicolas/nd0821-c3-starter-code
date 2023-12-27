# Script to train machine learning model.

# Add the necessary imports for the starter code.
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

# Add code to load in the data.
data = pd.read_csv("starter/data/census.csv")

train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Save the encoder and lb
pickle.dump(encoder, open("starter/model/encoder.pickle", "wb"))
pickle.dump(lb, open("starter/model/lb.pickle", "wb"))

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train and save a model.
model = train_model(X_train, y_train)
pickle.dump(model, open("starter/model/model.pickle", "wb"))


def compute_model_metrics_on_slices(
    model,
    data,
    cat_features,
    encoder,
    lb,
    fixed_feature,
    fixed_value,
    print_result=False,
):
    relevant_data = data[data[fixed_feature] == value]
    relevant_X_test, relevant_y_test, _, _ = process_data(
        relevant_data,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    relevant_preds = inference(model, relevant_X_test)

    precision, recall, fbeta = compute_model_metrics(
        relevant_y_test,
        relevant_preds)

    if print_result:
        print("Performance for", fixed_feature, "==", fixed_value, ":")
        print("Precision:", precision)
        print("Recall:", recall)
        print("Fbeta:", fbeta)

    return precision, recall, fbeta


# Print performance on model slices (education fixed)
for value in test["education"].unique():
    performance = compute_model_metrics_on_slices(
        model, test, cat_features, encoder, lb, "education", value, True
    )
