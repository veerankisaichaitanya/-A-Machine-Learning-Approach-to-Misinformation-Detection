#Main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import traceback

app = FastAPI()

# Load the model and vectorizer
try:
    with open("fakenews.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("countvectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except Exception as e:
    raise RuntimeError(f"Error loading model or vectorizer: {str(e)}")

# Define a model for the request body
class RequestBody(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fake News Detection API!"}

@app.post("/predict/")
def predict_fake_news(request_body: RequestBody):
    try:
        text = request_body.text
        transformed_text = vectorizer.transform([text])
        prediction = model.predict(transformed_text)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        # Log the exception for debugging purposes
        print("Error during prediction:", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal Server Error")

#run_api.py
import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

#train_and_save_model.py
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
