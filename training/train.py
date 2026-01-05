import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

# load settings:file path
with open("settings.json", "r") as f:
    settings = json.load(f)

file_source_path = settings["csv_path"]
file_model_path = settings["model_path"]

# Load dataset
df = pd.read_csv(
    file_source_path,
    sep=';',
    encoding="latin1",
    header=None,             # no header row in file
    names=['v1', 'v2']       # manually name them
)
df.columns = ["label", "text"]

df["label"] = df["label"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2
)

# Pipeline = vectorization + model
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

# Convert to ONNX
initial_type = [("input", StringTensorType([None, 1]))]
onnx_model = convert_sklearn(pipeline, initial_types=initial_type)

with open(file_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())
