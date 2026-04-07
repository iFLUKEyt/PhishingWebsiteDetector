import pandas as pd
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# URL cleaning function
def clean_url(url):
    url = url.lower().replace("http://", "").replace("https://", "").strip("/")
    return url

# Load dataset
df = pd.read_csv("phishing_site_urls.csv")
df['Label'] = df['Label'].map({'bad': 1, 'good': 0})
df['URL'] = df['URL'].apply(clean_url)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['URL'])
y = df['Label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression model
start_time = time.time()
log_model = LogisticRegression(max_iter=500)
log_model.fit(X_train, y_train)
print("Logistic Regression Results:")
print(classification_report(y_test, log_model.predict(X_test)))
print(f"Training time: {time.time() - start_time:.2f} seconds\n")


start_time = time.time()
tree_model = DecisionTreeClassifier(
    max_depth=10,            
    min_samples_split=5,     
    min_samples_leaf=2,       
    random_state=42
)
tree_model.fit(X_train, y_train)
tree_time = time.time() - start_time
print("Decision Tree Results:")
print(classification_report(y_test, tree_model.predict(X_test)))
print(f"Training time: {tree_time:.2f} seconds\n")

# Save models and vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("log_model.pkl", "wb") as f:
    pickle.dump(log_model, f)

print("Models and vectorizer saved successfully.")
