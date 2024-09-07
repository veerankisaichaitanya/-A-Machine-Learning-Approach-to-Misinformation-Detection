import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

# Assuming you have trained your model and vectorizer
model = LogisticRegression()
vectorizer = CountVectorizer()

# Example: Fit the vectorizer and model on some training data
# X_train = vectorizer.fit_transform(training_data)
# model.fit(X_train, y_train)

# Save the trained logistic regression model
with open("fakenews.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Save the count vectorizer
with open("countvectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
