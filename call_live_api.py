import requests


data = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capitalloss": 0,
    "hours-perweek": 40,
    "native-country": "United-States"
}

r = requests.post(
    'https://udacity-census-ml.herokuapp.com/prediction', json=data)

assert r.status_code == 200

print("Response code: %s" % r.status_code)
print("Response body: %s" % r.json())
