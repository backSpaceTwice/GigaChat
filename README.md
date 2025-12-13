# GigaChat -- Gen Z Slang & Meme Explainer

A lightweight PyTorch-based chatbot that explains Gen Z slang, internet
memes, and modern online culture. The model uses a simple feedforward
neural network trained on categorized patterns and responses.

## Features

- AI-powered slang and meme explanations
- 30+ slang terms with clear definitions
- Meme breakdowns including origins and examples
- Context-aware responses
- Easily expandable JSON-based dataset
- Fast, lightweight model for quick inference

## Prerequisites

Before you begin, ensure you have the following installed:

*   **Python**: Version 3.8 or higher is recommended.
*   **pip**: The Python package installer, typically bundled with Python.
*   **Git**: For cloning the repository.

## Quick Start

### Setup for Mac and Linux
- Nav to to project directory

```bash

cd GigaChat
```

```bash
python -m venv venv          # Create a virtual environment (venv is usually included with Python 3.3+)
source venv/bin/activate     # Windows: venv\Scripts\activate

pip install -r requirements.txt

# NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet')"
```

### Train the Model

```bash
python scripts/train_model.py
```

### Run the Chatbot

```bash
python src/main.py
```

### Run the WebApp (hosted on http://localhost:5002/)

```bash
python flask_app.py
```

## Example Usage

    You: what does rizz mean?
    GigaChat: Rizz means charisma or strong flirting ability.

    You: tell me about a meme
    GigaChat: “Distracted Boyfriend” is a 2015 stock photo that became a meme
    used to represent switching attention or temptation.

    You: what's cap?
    GigaChat: “Cap” means a lie or something untrue.

## Project Stats

- 30+ intents
- Model accuracy: ~95%
- Inference time: under 10ms
- Model size: ~50 KB

## Tech Stack

- PyTorch
- NLTK (tokenization, lemmatization)
- JSON-based data files
- Feedforward neural network (3 layers)
- Adam optimizer + cross-entropy loss

## Adding New Content

### Add a Slang Term

Edit `data/intents.json`:

```json
{
  "tag": "your_slang",
  "patterns": ["What does your_slang mean?"],
  "responses": ["your_slang means ..."]
}
```

### Add a Meme

Edit `data/memes.json`:

```json
{
  "meme_key": {
    "name": "Meme Name",
    "origin": "Where it came from",
    "meaning": "What it represents",
    "usage": "How people use it",
    "example": "Example sentence"
  }
}
```

## Configuration

odify training hyperparameters in `scripts/train_model.py`:

```python
BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 100
DROPOUT = 0.5
```
