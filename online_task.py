import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import numpy as np




data=pd.read_csv('/content/combined_emails_with_natural_pii.csv')
X_train, X_test, y_train, y_test = train_test_split(data['email'], data['type'], test_size=0.2, random_state=42)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
X_train = X_train.toarray()
X_test = X_test.toarray()
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
y_train = torch.tensor(y_train) # y_train is a NumPy array
y_test = torch.tensor(y_test)   # Assuming y_test is a pandas Series
loss_fn = nn.CrossEntropyLoss()  # Example for multi-class classification
 # Example optimizer

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_true)) * 100
    return acc
model_0 = nn.Sequential(
    nn.Linear(28113, 64),   # First layer: fully connected
    nn.ReLU(),            # Activation layer
    nn.Linear(64, 10)     # Output layer
)
optimizer = torch.optim.Adam(model_0.parameters(), lr=0.001)
epochs=10
for epoch in range(epochs):
    ### Training
    model_0.train()

    # 1. Forward pass (model outputs raw logits)
    y_logits = model_0(X_train) # squeeze to remove extra `1` dimensions, this won't work unless model and data are on same device
    y_pred = torch.argmax(y_logits, dim=1) # turn logits -> pred probs -> pred labls

    #
    loss = loss_fn(y_logits,y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_0.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_0(X_test)
        test_pred = torch.argmax(test_logits, dim=1) # Get predicted class labels
        # 2. Caculate loss/accuracy
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

    # Print out what's happening every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
torch.save(model.state_dict(), 'your_model_name1.pth')