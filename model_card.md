# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
A Random Forest Classifier using the default parameters 
## Intended Use
The model could be used to predict whether the salary of a person will be more or less than 50k based on the Census Income Data Set
## Training Data
80% of Census Income Data Set by UCI
## Evaluation Data
The other 20%
## Metrics
- precision : 0.74
- recall : 0.62
- fbeta : 0.67

## Ethical Considerations
Model Fairness should be examined further, scince we have race and gender in dataset which could conduct to potential gender,racial discrimination.  

## Caveats and Recommendations
We could perform Hyper-parameter optimization and more feature engineering and feature selection for  better results.