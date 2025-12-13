"""
Utility functions for GigaChat
Handles text processing, data loading, and helper functions
"""

import os
import json
import nltk
import numpy as np
from pathlib import Path


def ensure_nltk_data():
    """
    Download required NLTK data if not present
    """
    required_packages = ['punkt', 'wordnet', 'omw-1.4', 'punkt_tab']
    
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            try:
                nltk.download(package, quiet=True)
                print(f"✅ Downloaded NLTK package: {package}")
            except:
                print(f"⚠️ Could not download NLTK package: {package}")


def tokenize_and_lemmatize(text):
    """
    Tokenize and lemmatize text
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of lemmatized tokens
    """
    lemmatizer = nltk.WordNetLemmatizer()
    
    # Tokenize
    words = nltk.word_tokenize(text.lower())
    
    # Lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word.isalnum()]
    
    return words


def create_bag_of_words(words, vocabulary):
    """
    Create bag of words vector
    
    Args:
        words (list): List of words
        vocabulary (list): Complete vocabulary
        
    Returns:
        list: Binary bag of words vector
    """
    return [1 if word in words else 0 for word in vocabulary]


def load_json(filepath):
    """
    Load JSON file
    
    Args:
        filepath (str): Path to JSON file
        
    Returns:
        dict: Parsed JSON data
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON in {filepath}: {e}")
        return None


def save_json(data, filepath):
    """
    Save data to JSON file
    
    Args:
        data (dict): Data to save
        filepath (str): Path to save file
    """
    try:
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved to {filepath}")
    except Exception as e:
        print(f"❌ Error saving to {filepath}: {e}")


def ensure_directories():
    """
    Ensure all required directories exist
    """
    directories = ['data', 'models', 'logs']
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def print_training_stats(epoch, total_epochs, loss, accuracy=None):
    """
    Print formatted training statistics
    
    Args:
        epoch (int): Current epoch
        total_epochs (int): Total number of epochs
        loss (float): Current loss
        accuracy (float, optional): Current accuracy
    """
    progress = (epoch / total_epochs) * 100
    bar_length = 30
    filled = int(bar_length * epoch // total_epochs)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    output = f"Epoch [{epoch:3d}/{total_epochs}] {bar} {progress:5.1f}% | Loss: {loss:.4f}"
    
    if accuracy is not None:
        output += f" | Acc: {accuracy:.2f}%"
    
    print(output)


def calculate_accuracy(predictions, targets):
    """
    Calculate classification accuracy
    
    Args:
        predictions (np.ndarray): Predicted class indices
        targets (np.ndarray): True class indices
        
    Returns:
        float: Accuracy percentage
    """
    correct = np.sum(predictions == targets)
    total = len(targets)
    return (correct / total) * 100


def get_confusion_matrix(predictions, targets, num_classes):
    """
    Calculate confusion matrix
    
    Args:
        predictions (np.ndarray): Predicted class indices
        targets (np.ndarray): True class indices
        num_classes (int): Number of classes
        
    Returns:
        np.ndarray: Confusion matrix
    """
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    for pred, target in zip(predictions, targets):
        matrix[target][pred] += 1
    
    return matrix


def format_time(seconds):
    """
    Format seconds into human-readable time
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


class ColorText:
    """ANSI color codes for terminal output"""
    
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    
    @classmethod
    def colored(cls, text, color):
        """Return colored text"""
        return f"{color}{text}{cls.END}"
    
    @classmethod
    def success(cls, text):
        """Return green success text"""
        return cls.colored(text, cls.GREEN)
    
    @classmethod
    def error(cls, text):
        """Return red error text"""
        return cls.colored(text, cls.RED)
    
    @classmethod
    def warning(cls, text):
        """Return yellow warning text"""
        return cls.colored(text, cls.YELLOW)
    
    @classmethod
    def info(cls, text):
        """Return cyan info text"""
        return cls.colored(text, cls.CYAN)


def print_banner():
    """Print GigaChat banner"""
    banner = """
    ╔═══════════════════════════════════════╗
    ║                                       ║
    ║   🔥 GIGACHAT - Gen Z Explainer 🔥   ║
    ║                                       ║
    ║   Your guide to slang, memes, and     ║
    ║   modern internet culture             ║
    ║                                       ║
    ╚═══════════════════════════════════════╝
    """
    print(ColorText.colored(banner, ColorText.CYAN))


def validate_intents_file(filepath):
    """
    Validate intents.json file structure
    
    Args:
        filepath (str): Path to intents file
        
    Returns:
        tuple: (is_valid, error_message)
    """
    data = load_json(filepath)
    
    if data is None:
        return False, "Could not load file"
    
    if 'intents' not in data:
        return False, "Missing 'intents' key"
    
    for i, intent in enumerate(data['intents']):
        if 'tag' not in intent:
            return False, f"Intent {i} missing 'tag'"
        if 'patterns' not in intent:
            return False, f"Intent {i} missing 'patterns'"
        if 'responses' not in intent:
            return False, f"Intent {i} missing 'responses'"
        
        if not isinstance(intent['patterns'], list):
            return False, f"Intent '{intent['tag']}' patterns must be a list"
        if not isinstance(intent['responses'], list):
            return False, f"Intent '{intent['tag']}' responses must be a list"
    
    return True, "Valid intents file"


def get_project_root():
    """
    Get the project root directory
    
    Returns:
        Path: Project root path
    """
    return Path(__file__).parent.parent


def get_data_path(filename):
    """
    Get path to file in data directory
    
    Args:
        filename (str): Name of file
        
    Returns:
        Path: Full path to file
    """
    return get_project_root() / 'data' / filename


def get_model_path(filename):
    """
    Get path to file in models directory
    
    Args:
        filename (str): Name of file
        
    Returns:
        Path: Full path to file
    """
    return get_project_root() / 'models' / filename