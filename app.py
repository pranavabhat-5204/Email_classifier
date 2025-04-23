#Importing the necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch

import torch.nn as nn
import numpy as np
from flask import Flask, render_template, request
import spacy
from sklearn.preprocessing import LabelEncoder

#Loading nl_core_news_sm using spacy for NER masking 
nlp = spacy.load("nl_core_news_sm")
import torch
import torch.nn as nn

app=Flask(__name__, template_folder=r"C:Users/SDP/Desktop")

#Now we need to create a few variables and also provide the vocabulary for decoding the output of the model
data=pd.read_csv("E:/combined_emails_with_natural_pii.csv")
X_train, X_test, y_train, y_test = train_test_split(data['email'], data['type'], test_size=0.2, random_state=42)
label_encoder = LabelEncoder()
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train = vectorizer.fit_transform(X_train)
y_train = label_encoder.fit_transform(y_train)

#Loading the model
model_path = 'E:/your_model_name1.pth'
model = nn.Sequential(
    nn.Linear(28113, 64),   # First layer: fully connected
    nn.ReLU(),            # Activation layer
    nn.Linear(64, 10)     # Output layer
)
model.load_state_dict(torch.load(model_path))

model.eval()
@app.route('/')
def email():
  return render_template('email_pred.html')
@app.route('/predict',methods=['POST','GET'])
def result():
  if request.method=='POST':
    result=request.form['Data']

    #Masking the Text
    doc=nlp(str(result))
    masked_text=str(result)
    for ent in doc.ents:
        if ent.label_ == "EMAIL":
            masked_text = masked_text.replace(ent.text, "[EMAIl]")
        elif ent.label_ == "PERSON":
            masked_text = masked_text.replace(ent.text, "[FULL NAME]")

    # Vectorize the input text before prediction
    result_vectorized = vectorizer.transform([str(result)])
    
    result_prediction = model(torch.tensor(result_vectorized.toarray(), dtype=torch.float32)) 
    predicted_class_index = torch.argmax(result_prediction, dim=1).item()
    class_names = label_encoder.classes_ # Access the correct attribute 'classes_'
    predicted_class_name = class_names[predicted_class_index]
    return render_template('email_result.html',result=predicted_class_name)
if __name__ == "__main__":
    app.run()
