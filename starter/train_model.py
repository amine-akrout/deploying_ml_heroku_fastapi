# Script to train machine learning model.
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.model import *
from ml.data import process_data
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness
from aequitas.plotting import Plot

# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv("../data/census.csv", sep=", ")
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder_t, lb_t = process_data(
    test, categorical_features=cat_features,
    label="salary", training=False, encoder=encoder, lb=lb)
# Train and save a model.
trained_model = train_model(X_train, y_train)

with open("../model/trained_model.pkl", "wb",) as file:
    pickle.dump(trained_model, file)
with open("../model/lb.pkl", "wb",) as file:
    pickle.dump(lb, file)
with open("../model/encoder.pkl", "wb",) as file:
    pickle.dump(encoder, file)

# caculate model performance on test data
predictions = inference(trained_model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, predictions)
print("precision : ", precision)
print("recall : ", recall)
print("fbeta : ", fbeta)

# performance of the model on slices of the data


def create_slices_performance():
    cat_features.append('salary')
    slices_df = test[cat_features]
    slices_df.reset_index(inplace=True, drop=True)
    slices_df['label_value'] = lb.transform(slices_df['salary']).ravel()
    slices_df = slices_df.drop('salary', axis=1)
    preds = pd.DataFrame(predictions, columns=['score'])
    slices_df = pd.concat([slices_df, preds], axis=1)
    return slices_df


slices_df = create_slices_performance()


def slices_performance():
    g = Group()
    xtab, _ = g.get_crosstabs(slices_df)
    absolute_metrics = g.list_absolute_metrics(xtab)
    check_df = xtab[['attribute_name', 'attribute_value'] +
                    absolute_metrics].round(2)
    check_df.to_csv(r'slice_model_output.txt', sep='\t\t')


slices_performance()
