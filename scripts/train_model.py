"""
Training script for GigaChat model
Trains the neural network on intent classification
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

from model import ChatbotModel, ModelConfig
from utils import (
    ensure_nltk_data, tokenize_and_lemmatize, create_bag_of_words,
    load_json, save_json, print_training_stats, calculate_accuracy,
    ColorText, format_time, get_data_path, get_model_path, ensure_directories
)


def prepare_training_data(intents_path):
    """
    Prepare training data from intents file
    
    Args:
        intents_path (Path): Path to intents.json
        
    Returns:
        tuple: (X, y, vocabulary, intents, intents_responses)
    """
    print("📖 Loading and processing intents...")
    
    intents_data = load_json(intents_path)
    if not intents_data:
        raise ValueError("Could not load intents file")
    
    documents = []
    vocabulary = []
    intents = []
    intents_responses = {}
    
    # Parse intents
    for intent in intents_data['intents']:
        tag = intent['tag']
        
        if tag not in intents:
            intents.append(tag)
            intents_responses[tag] = intent['responses']
        
        for pattern in intent['patterns']:
            words = tokenize_and_lemmatize(pattern)
            vocabulary.extend(words)
            documents.append((words, tag))
    
    # Create unique sorted vocabulary
    vocabulary = sorted(set(vocabulary))
    
    print(f"✅ Processed {len(documents)} training examples")
    print(f"✅ Vocabulary size: {len(vocabulary)} unique words")
    print(f"✅ Number of intents: {len(intents)}")
    
    # Create training data
    X = []
    y = []
    
    for words, tag in documents:
        bag = create_bag_of_words(words, vocabulary)
        intent_index = intents.index(tag)
        
        X.append(bag)
        y.append(intent_index)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    return X, y, vocabulary, intents, intents_responses


def train_model(X, y, config, validation_split=0.2):
    """
    Train the chatbot model
    
    Args:
        X (np.ndarray): Input features
        y (np.ndarray): Target labels
        config (ModelConfig): Model configuration
        validation_split (float): Fraction of data for validation
        
    Returns:
        tuple: (trained_model, train_losses, val_losses, val_accuracies)
    """
    print("\n🏋️ Starting training...")
    print(f"Training config: {config.to_dict()}")
    
    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Create dataset
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Split into train/validation
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"📊 Train size: {train_size}, Validation size: {val_size}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize model
    input_size = X.shape[1]
    output_size = len(np.unique(y))
    
    model = ChatbotModel(
        input_size, 
        output_size,
        hidden_size_1=config.hidden_size_1,
        hidden_size_2=config.hidden_size_2,
        dropout_rate=config.dropout_rate
    )
    
    print(f"\n🤖 Model architecture:")
    for key, value in model.get_architecture_summary().items():
        print(f"   {key}: {value}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training history
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    start_time = time.time()
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.numpy())
                all_targets.extend(batch_y.numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = calculate_accuracy(np.array(all_preds), np.array(all_targets))
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # Print progress every 10 epochs or at the end
        if epoch % 10 == 0 or epoch == config.epochs:
            print_training_stats(epoch, config.epochs, avg_train_loss, val_acc)
    
    training_time = time.time() - start_time
    
    print(f"\n✅ Training completed in {format_time(training_time)}")
    print(f"🎯 Best validation accuracy: {best_val_acc:.2f}%")
    
    return model, train_losses, val_losses, val_accuracies


def save_model_and_artifacts(model, vocabulary, intents, output_dir):
    """
    Save trained model and associated artifacts
    
    Args:
        model (ChatbotModel): Trained model
        vocabulary (list): Vocabulary list
        intents (list): Intent labels
        output_dir (Path): Output directory
    """
    print("\n💾 Saving model and artifacts...")
    
    # Save model weights
    model_path = output_dir / 'chatbot_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"✅ Saved model to {model_path}")
    
    # Save dimensions
    dimensions_path = output_dir / 'dimensions.json'
    dimensions = {
        'input_size': model.input_size,
        'output_size': model.output_size
    }
    save_json(dimensions, dimensions_path)
    
    # Save vocabulary
    vocab_path = output_dir / 'vocabulary.json'
    save_json({'vocabulary': vocabulary}, vocab_path)
    
    # Save intents list
    intents_list_path = output_dir / 'intents_list.json'
    save_json({'intents': intents}, intents_list_path)
    
    print(ColorText.success("✅ All artifacts saved successfully!"))


def main():
    """Main training function"""
    print(ColorText.colored("\n" + "="*50, ColorText.CYAN))
    print(ColorText.colored("      GIGACHAT MODEL TRAINING", ColorText.CYAN))
    print(ColorText.colored("="*50 + "\n", ColorText.CYAN))
    
    # Ensure required directories exist
    ensure_directories()
    
    # Ensure NLTK data
    ensure_nltk_data()
    
    # Paths
    intents_path = get_data_path('intents.json')
    output_dir = Path('models')
    
    try:
        # Prepare data
        X, y, vocabulary, intents, intents_responses = prepare_training_data(intents_path)
        
        # Create config
        config = ModelConfig(
            hidden_size_1=128,
            hidden_size_2=64,
            dropout_rate=0.5,
            learning_rate=0.001,
            batch_size=8,
            epochs=100
        )
        
        # Train model
        model, train_losses, val_losses, val_accuracies = train_model(X, y, config)
        
        # Save everything
        save_model_and_artifacts(model, vocabulary, intents, output_dir)
        
        print("\n" + ColorText.colored("="*50, ColorText.GREEN))
        print(ColorText.success("      TRAINING COMPLETE! 🎉"))
        print(ColorText.colored("="*50, ColorText.GREEN))
        print(f"\nYou can now run the chatbot with:")
        print(ColorText.colored("  python src/main.py", ColorText.CYAN))
        print()
        
    except Exception as e:
        print(ColorText.error(f"\n❌ Training failed: {e}"))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()