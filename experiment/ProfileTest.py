import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import requests
import os

class DummyModel(nn.Module):
    def __init__(self, input_dim=10, output_dim=2):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

def fetch_dummy_data():
    # Simulate fetching dummy data
    print("Fetching data from dummy API...")
    response = requests.get("https://httpbin.org/get")
    return response.json()

def preprocess():
    # Dummy transform pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    print("Dummy transform initialized.")
    return transform

def train_dummy_model():
    model = DummyModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    dummy_input = torch.randn(5, 10)
    dummy_target = torch.randn(5, 2)

    for epoch in range(3):
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")

def main():
    print("== ProfileTest.py ==")
    data = fetch_dummy_data()
    print("Data fetched:", data["url"])
    preprocess()
    train_dummy_model()
    print("Dummy training completed.")

if __name__ == "__main__":
    main()