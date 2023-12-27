# Model Card

## Model Details
Nicolas Herrmann created the model. It is random forest classifier using the default hyperparameters in scikit-learn 1.3.2.

## Intended Use
This model should be used to predict the salary (>50K or <=50K) based on a handful of attributes.

## Training Data
The training data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/20/census+income). The target class "salary" includes 2 categories (>50K or <=50K). The training set uses 80% of all available datapoints to train the machine learning model.

## Evaluation Data
The test data was also obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/20/census+income). The test set uses 20% of all available datapoints to evaluate the performance of the machine learning model.

## Metrics
The trained machine learning model was evaluated on three metrics (precision, recall and fbeta) and achieved the following performance:
- precision = 0.729
- recall = 0.620
- fbeta = 0.670

## Ethical Considerations
Providing this prediction model is intended to help people to better understand the impact of features (e.g., education) on the salary. The model is no replacement for human rationality or individual decision making (e.g., salary negotiations).

## Caveats and Recommendations
Before using the model for real-world applications, it would make sense to further investigate a possible bias of the model. Packages like Aequitas can be used to check for unfairness in the underlying data or in the model.
