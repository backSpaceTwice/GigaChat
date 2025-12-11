"""
Neural Network Model for GigaChat
A simple feedforward neural network for intent classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChatbotModel(nn.Module):
    """
    3-layer feedforward neural network for intent classification
    
    Architecture:
        Input Layer -> 128 neurons (ReLU, Dropout)
        Hidden Layer -> 64 neurons (ReLU, Dropout)
        Output Layer -> num_intents neurons
    
    Args:
        input_size (int): Size of input feature vector (vocabulary size)
        output_size (int): Number of intent classes
        hidden_size_1 (int): Size of first hidden layer (default: 128)
        hidden_size_2 (int): Size of second hidden layer (default: 64)
        dropout_rate (float): Dropout probability (default: 0.5)
    """

    def __init__(
        self, 
        input_size, 
        output_size, 
        hidden_size_1=128, 
        hidden_size_2=64, 
        dropout_rate=0.5
    ):
        super(ChatbotModel, self).__init__()

        # Network layers
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, output_size)
        
        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Store architecture info
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = [hidden_size_1, hidden_size_2]

    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, output_size)
        """
        # First hidden layer
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Second hidden layer
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output layer (no activation, used with CrossEntropyLoss)
        x = self.fc3(x)

        return x

    def predict_proba(self, x):
        """
        Get probability distribution over classes
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Probability distribution
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities

    def get_architecture_summary(self):
        """
        Get a summary of the model architecture
        
        Returns:
            dict: Architecture details
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'hidden_layers': self.hidden_sizes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'dropout_rate': self.dropout.p
        }


class ModelConfig:
    """Configuration class for model hyperparameters"""
    
    def __init__(
        self,
        hidden_size_1=128,
        hidden_size_2=64,
        dropout_rate=0.5,
        learning_rate=0.001,
        batch_size=8,
        epochs=100
    ):
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            'hidden_size_1': self.hidden_size_1,
            'hidden_size_2': self.hidden_size_2,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        return cls(**config_dict)