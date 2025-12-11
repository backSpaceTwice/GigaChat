# 🚀 GigaChat Complete Setup Guide

## 📋 Quick Start (5 Minutes)

Follow these steps to get GigaChat running:

### Step 1: Create Project Structure

```bash
# Create main directory
mkdir GigaChat
cd GigaChat

# Create subdirectories
mkdir src data models scripts tests docs notebooks
```

### Step 2: Create Files

Create these files with the content from the artifacts:

```
GigaChat/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py           (empty file)
│   ├── main.py
│   ├── model.py
│   ├── response_handler.py
│   └── utils.py
├── data/
│   ├── intents.json
│   └── memes.json
├── scripts/
│   └── train_model.py
├── models/                    (empty for now)
└── docs/
    └── USER_GUIDE.md
```

**Create empty `__init__.py`:**

```bash
touch src/__init__.py
touch tests/__init__.py
```

### Step 3: Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Step 4: Train the Model

```bash
python scripts/train_model.py
```

You should see:

```
====================================================
           GIGACHAT MODEL TRAINING
====================================================

📖 Loading and processing intents...
✅ Processed 200+ training examples
✅ Vocabulary size: 450 unique words
✅ Number of intents: 32

🏋️ Starting training...
...
✅ Training completed!
```

### Step 5: Run GigaChat

```bash
python src/main.py
```

You should see:

```
    ╔═══════════════════════════════════════╗
    ║                                       ║
    ║   🔥 GIGACHAT - Gen Z Explainer 🔥   ║
    ║                                       ║
    ║   Your guide to slang, memes, and     ║
    ║   modern internet culture             ║
    ║                                       ║
    ╚═══════════════════════════════════════╝

📚 Loaded 10 memes
🎯 Trained on 32 intents

Type your questions or '/quit' to exit.

You:
```

## ✅ Verification Checklist

Make sure everything is working:

- [ ] Virtual environment activated
- [ ] All dependencies installed
- [ ] NLTK data downloaded
- [ ] Model trained successfully
- [ ] `models/chatbot_model.pth` exists
- [ ] `models/dimensions.json` exists
- [ ] Chatbot runs without errors
- [ ] Can ask about slang and get responses
- [ ] Can request memes and get explanations

## 🧪 Test Your Setup

Try these commands to verify everything works:

```bash
# In the GigaChat chat interface:
You: what does rizz mean?
# Should get a clear explanation

You: tell me about a meme
# Should get a formatted meme explanation

You: /stats
# Should show statistics

You: /memes
# Should list all memes

You: /quit
# Should exit cleanly
```

## 🐛 Common Issues & Solutions

### Issue: `ModuleNotFoundError: No module named 'src'`

**Solution:** Make sure you're running from the project root:

```bash
cd GigaChat  # Go to project root
python src/main.py
```

### Issue: `FileNotFoundError: intents.json`

**Solution:** Verify file structure:

```bash
ls data/  # Should show intents.json and memes.json
```

### Issue: `LookupError: NLTK data not found`

**Solution:** Download NLTK data:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Issue: Model file not found

**Solution:** Train the model first:

```bash
python scripts/train_model.py
```

### Issue: `torch` not found

**Solution:** Install PyTorch:

```bash
pip install torch
```

## 📦 File Checklist

Ensure you have all these files:

### Core Files (Required)

- [x] `src/main.py` - Main application
- [x] `src/model.py` - Neural network model
- [x] `src/response_handler.py` - Response management
- [x] `src/utils.py` - Utility functions
- [x] `data/intents.json` - Training data
- [x] `data/memes.json` - Meme database
- [x] `scripts/train_model.py` - Training script
- [x] `requirements.txt` - Dependencies

### Documentation (Recommended)

- [x] `README.md` - Project overview
- [x] `docs/USER_GUIDE.md` - User documentation
- [x] `.gitignore` - Git ignore rules

### Generated Files (After Training)

- [ ] `models/chatbot_model.pth` - Trained model
- [ ] `models/dimensions.json` - Model dimensions
- [ ] `models/vocabulary.json` - Saved vocabulary
- [ ] `models/intents_list.json` - Intent labels

## 🎯 Next Steps

Once everything is working:

1. **Try it out**: Chat with GigaChat and explore features
2. **Add content**: Add more memes to `data/memes.json`
3. **Expand slang**: Add more terms to `data/intents.json`
4. **Retrain**: Run `python scripts/train_model.py` after changes
5. **Customize**: Modify responses to match your style

## 📚 For Development

If you want to develop/modify GigaChat:

```bash
# Install development dependencies
pip install pytest pytest-cov black flake8

# Run tests (when you write them)
pytest tests/

# Format code
black src/

# Lint code
flake8 src/
```

## 🌟 You're All Set!

GigaChat is now ready to use. Start chatting and exploring Gen Z slang and memes!

Need help? Check:

- `docs/USER_GUIDE.md` - How to use GigaChat
- `README.md` - Project overview
- GitHub Issues - Report problems

---

**Project by**: Andy Nguyen & Quan Khong

**Stay based, no cap** 🔥
