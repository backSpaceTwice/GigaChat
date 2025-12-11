"""
GigaChat Flask Web Application
API backend for web interface
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import sys
from pathlib import Path
import secrets

# Add src to path
sys.path.insert(0, str(Path('src').absolute()))

from src.main import GigaChat

# Initialize Flask app
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
CORS(app)

# Load chatbot once at startup
print("🤖 Loading GigaChat...")
chatbot = GigaChat()
chatbot.load_intents()
chatbot.load_model()
print("✅ GigaChat ready!")


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get response from chatbot
        response = chatbot.chat(user_message)
        
        # Store in session history
        if 'chat_history' not in session:
            session['chat_history'] = []
        
        session['chat_history'].append({
            'user': user_message,
            'bot': response
        })
        
        return jsonify({
            'response': response,
            'success': True
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/api/random-meme', methods=['GET'])
def random_meme():
    """Get a random meme"""
    try:
        meme = chatbot.meme_handler.get_random_meme()
        return jsonify({
            'meme': meme,
            'success': True
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/api/stats', methods=['GET'])
def stats():
    """Get chatbot statistics"""
    try:
        return jsonify({
            'intents': len(chatbot.intents),
            'vocabulary': len(chatbot.vocabulary),
            'memes': chatbot.meme_handler.get_meme_count(),
            'meme_list': chatbot.meme_handler.list_all_memes(),
            'success': True
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    """Clear chat history"""
    session['chat_history'] = []
    return jsonify({'success': True})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)