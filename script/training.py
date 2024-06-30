import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import save_pickle, load_pickle

class SentimentNet(nn.Module):
    def __init__(self, input_dim):
        super(SentimentNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x

def train_neural_network(X_train, y_train, X_test, y_test, num_epochs=10, batch_size=32):
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.todense(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.todense(), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    # Create DataLoader for training and testing
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    input_dim = X_train.shape[1]
    model = SentimentNet(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    # Evaluation
    model.eval()
    y_pred = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = model(inputs).squeeze()
            predicted = (outputs >= 0.5).float().tolist()
            y_pred.extend(predicted)
    
    y_pred_binary = [1 if prob >= 0.5 else 0 for prob in y_pred]
    accuracy = accuracy_score(y_test, y_pred_binary)
    print(f'Accuracy: {accuracy}')
    print(classification_report(y_test, y_pred_binary))

    # Save the trained model if accuracy is above 70%
    if accuracy >= 0.70:
        save_pickle(model, '../model/neural_network_model.pkl')
        print("Model saved with accuracy:", accuracy)
    else:
        print("Model not saved due to low accuracy.")

if __name__ == "__main__":
    # Load preprocessed data
    X_train_tfidf, y_train = load_pickle('../data/proses/train_tfidf.pkl')
    X_test_tfidf, y_test = load_pickle('../data/proses/test_tfidf.pkl')

    # Train and evaluate neural network model
    train_neural_network(X_train_tfidf, y_train, X_test_tfidf, y_test)
