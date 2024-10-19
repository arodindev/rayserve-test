import requests

# Test the Ray Serve deployment
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"inputs": [[5.1, 3.5, 1.4, 0.2]]}
)
print(response.json())  # Example for Iris dataset input

"""
sample_request_input = {
    "sepal length": 1.2,
    "sepal width": 1.0,
    "petal length": 1.1,
    "petal width": 0.9,
}
response = requests.get("http://localhost:8000/predict", json=sample_request_input)
print(response.text)
"""