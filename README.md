# 🔥 GigaChat - Gen Z Slang & Meme Explainer

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A neural network-powered chatbot that helps you understand Gen Z slang, internet memes, and modern youth culture. Built with PyTorch and trained on real-world slang patterns.

## ✨ Features

- 🤖 **AI-Powered Understanding**: Neural network trained to recognize intent from natural language
- 💬 **30+ Slang Terms**: Comprehensive explanations of popular Gen Z expressions
- 🎭 **Meme Encyclopedia**: Detailed breakdowns of viral memes with origin stories
- 🎯 **Context-Aware Responses**: Multiple response variants for natural conversation
- 📚 **Easy to Extend**: Simple JSON-based system for adding new content
- ⚡ **Fast & Lightweight**: Efficient bag-of-words model with quick inference

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gigachat.git
cd gigachat

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

### Training the Model

```bash
python scripts/train_model.py
```

### Running GigaChat

```bash
python src/main.py
```

## 💬 Example Conversations

```
You: what does rizz mean?
GigaChat: Rizz means charisma or smooth flirting ability. Someone with rizz
can charm or attract others easily. Example: 'Bro got her number in 2
minutes, he's got W rizz!'

You: tell me about a meme
GigaChat: 🔥 Distracted Boyfriend

📍 Origin: Stock photo from 2015, went viral in 2017

💭 Meaning: Represents getting distracted by something new while ignoring
your current commitment

📱 How it's used: Used to show temptation, disloyalty, or changing preferences

💬 Example: Me (boyfriend) ignoring my responsibilities (girlfriend) to
scroll TikTok (other woman)

You: what's cap?
GigaChat: Cap means a lie or something false. Saying 'that's cap' means
'that's not true.' Example: 'He said he bench presses 300lbs—that's cap!'
```

## 📊 Project Statistics

- **Intents**: 30+ slang terms and meme categories
- **Training Patterns**: 200+ example sentences
- **Response Variants**: 3-4 per intent
- **Meme Database**: 8+ memes (easily expandable)
- **Model Accuracy**: ~95%+ on test set

## 🛠️ Technology Stack

- **Framework**: PyTorch 2.0+
- **NLP**: NLTK (tokenization, lemmatization)
- **Model**: 3-layer feedforward neural network
- **Training**: Adam optimizer, Cross-Entropy loss
- **Data Format**: JSON for easy maintenance

## 📖 Documentation

- [User Guide](docs/USER_GUIDE.md) - How to use GigaChat
- [Developer Guide](docs/DEVELOPER_GUIDE.md) - How to extend and modify
- [Meme Database](docs/MEME_DATABASE.md) - Adding and managing memes

## 🎓 Team

**Project by**: Andy Nguyen & Quan Khong

**Roles**:

- **Andy Nguyen** - Project Manager, Model Architecture, Training & Optimization
- **Quan Khong** - Lead Developer, Dataset Creation, Response Management, Integration

## 📝 Adding New Content

### Adding Slang Terms

Edit `data/intents.json`:

```json
{
  "tag": "your_slang",
  "patterns": ["What does your_slang mean?", "Explain your_slang"],
  "responses": ["Your_slang means...", "It's used when..."]
}
```

### Adding Memes

Edit `data/memes.json`:

```json
{
  "meme_key": {
    "name": "Meme Name",
    "origin": "Where it came from",
    "meaning": "What it represents",
    "usage": "How people use it",
    "example": "Usage example"
  }
}
```

Or use the helper script:

```bash
python scripts/add_memes.py
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_model.py

# Run with coverage
python -m pytest --cov=src tests/
```

## 🔧 Configuration

Modify training parameters in `scripts/train_model.py`:

```python
BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 100
DROPOUT = 0.5
```

## 📈 Performance

| Metric         | Value       |
| -------------- | ----------- |
| Training Time  | ~30 seconds |
| Inference Time | <10ms       |
| Model Size     | ~50KB       |
| Accuracy       | 95%+        |
| Memory Usage   | <100MB      |

## 🐛 Known Issues

- Requires exact NLTK data downloads
- Very short phrases may confuse the model
- New slang requires retraining

## 🔮 Future Enhancements

- [ ] Add confidence scores to predictions
- [ ] Implement conversation history/context
- [ ] Create web interface (Flask/Streamlit)
- [ ] Add pronunciation guides
- [ ] Include video examples from TikTok
- [ ] Multi-language support
- [ ] Real-time slang updates via API
- [ ] Mobile app version

## 📜 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Data sources: Know Your Meme, Urban Dictionary, r/OutOfTheLoop
- Inspired by modern youth culture and internet linguistics
- Built for educational purposes

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/gigachat/issues)
- **Email**: your.email@example.com
- **Discord**: [Join our server](https://discord.gg/yourserver)

## ⭐ Star History

If you find GigaChat helpful, please consider starring the repository!

---

Made with 💜 by Andy & Quan | Stay based, no cap fr fr 🔥
# GigaChat
