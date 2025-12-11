"""
GigaChat Main Application
A Gen Z slang and meme explainer chatbot
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path if running from project root
if Path('src').exists():
    sys.path.insert(0, str(Path('src').absolute()))

from model import ChatbotModel
from response_handler import MemeResponseHandler, ResponseManager
from utils import (
    ensure_nltk_data, tokenize_and_lemmatize, create_bag_of_words,
    load_json, save_json, print_banner, ColorText, get_data_path, get_model_path
)


class GigaChat:
    """
    Main GigaChat application class
    Handles model loading, prediction, and response generation
    """
    
    def __init__(self, intents_path=None, memes_path=None, model_path=None, dimensions_path=None):
        """
        Initialize GigaChat
        
        Args:
            intents_path (str): Path to intents.json
            memes_path (str): Path to memes.json
            model_path (str): Path to saved model
            dimensions_path (str): Path to dimensions.json
        """
        # Set default paths
        self.intents_path = intents_path or get_data_path('intents.json')
        self.memes_path = memes_path or get_data_path('memes.json')
        self.model_path = model_path or get_model_path('chatbot_model.pth')
        self.dimensions_path = dimensions_path or get_model_path('dimensions.json')
        
        # Initialize components
        self.model = None
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}
        
        # Initialize handlers
        self.meme_handler = MemeResponseHandler(self.memes_path)
        self.response_manager = ResponseManager()
        
        # Ensure NLTK data
        ensure_nltk_data()
    
    def load_intents(self):
        """Load and parse intents file"""
        print(f"📖 Loading intents from {self.intents_path}...")
        
        intents_data = load_json(self.intents_path)
        if not intents_data:
            raise ValueError("Could not load intents file")
        
        documents = []
        
        for intent in intents_data['intents']:
            tag = intent['tag']
            
            if tag not in self.intents:
                self.intents.append(tag)
                self.intents_responses[tag] = intent['responses']
            
            for pattern in intent['patterns']:
                words = tokenize_and_lemmatize(pattern)
                self.vocabulary.extend(words)
                documents.append((words, tag))
        
        # Create unique sorted vocabulary
        self.vocabulary = sorted(set(self.vocabulary))
        
        print(f"✅ Loaded {len(self.intents)} intents")
        print(f"✅ Vocabulary size: {len(self.vocabulary)} words")
        
        return documents
    
    def load_model(self):
        """Load trained model from disk"""
        print(f"🤖 Loading model from {self.model_path}...")
        
        # Load dimensions
        dimensions = load_json(self.dimensions_path)
        if not dimensions:
            raise ValueError("Could not load model dimensions")
        
        # Create model
        self.model = ChatbotModel(
            dimensions['input_size'],
            dimensions['output_size']
        )
        
        # Load weights
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=True)
        )
        self.model.eval()
        
        print(f"✅ Model loaded successfully")
        print(f"   Input size: {dimensions['input_size']}")
        print(f"   Output size: {dimensions['output_size']}")
    
    def predict(self, user_input, confidence_threshold=0.5):
        """
        Predict intent from user input
        
        Args:
            user_input (str): User's message
            confidence_threshold (float): Minimum confidence to return prediction
            
        Returns:
            tuple: (intent, confidence)
        """
        # Tokenize and lemmatize
        words = tokenize_and_lemmatize(user_input)
        
        # Create bag of words
        bag = create_bag_of_words(words, self.vocabulary)
        bag_tensor = torch.tensor([bag], dtype=torch.float32)
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)
            probabilities = torch.softmax(predictions, dim=1)
        
        # Get top prediction
        confidence, predicted_idx = torch.max(probabilities, dim=1)
        confidence = confidence.item()
        predicted_idx = predicted_idx.item()
        
        # Check confidence threshold
        if confidence < confidence_threshold:
            return None, confidence
        
        predicted_intent = self.intents[predicted_idx]
        
        return predicted_intent, confidence
    
    def get_response(self, intent, user_input=""):
        """
        Get response for an intent
        
        Args:
            intent (str): Predicted intent
            user_input (str): Original user input
            
        Returns:
            str: Response text
        """
        # Handle meme requests
        if intent == 'meme_request':
            return self.meme_handler.get_random_meme()
        
        # Get responses for intent
        responses = self.intents_responses.get(intent, [])
        
        if not responses:
            return None
        
        # Select response
        return self.response_manager.select_response(responses, user_input)
    
    def chat(self, user_input):
        """
        Process user input and generate response
        
        Args:
            user_input (str): User's message
            
        Returns:
            str: Chatbot response
        """
        # Predict intent
        intent, confidence = self.predict(user_input)
        
        # If no confident prediction
        if intent is None:
            return "Sorry, I didn't quite get that. Try asking about Gen Z slang or say 'tell me about a meme'!"
        
        # Get response
        response = self.get_response(intent, user_input)
        
        if response is None:
            return "Hmm, I don't have a good response for that yet. Ask me about slang or memes!"
        
        return response
    
    def interactive_mode(self):
        """Run interactive chat session"""
        print_banner()
        print(f"📚 Loaded {self.meme_handler.get_meme_count()} memes")
        print(f"🎯 Trained on {len(self.intents)} intents\n")
        print("Type your questions or commands:")
        print("  • Ask about any Gen Z slang")
        print("  • Say 'tell me about a meme'")
        print("  • Type '/quit' or '/exit' to leave")
        print("  • Type '/help' for more commands\n")
        
        while True:
            try:
                # Get user input
                user_input = input(ColorText.colored("You: ", ColorText.GREEN))
                
                # Handle empty input
                if not user_input.strip():
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if self._handle_command(user_input):
                        break
                    continue
                
                # Get response
                response = self.chat(user_input)
                
                # Print response
                print(ColorText.colored(f"GigaChat: ", ColorText.CYAN) + response + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Peace out! Thanks for chatting!")
                break
            except Exception as e:
                print(ColorText.error(f"\n❌ Error: {e}\n"))
    
    def _handle_command(self, command):
        """
        Handle special commands
        
        Args:
            command (str): Command string
            
        Returns:
            bool: True if should exit, False otherwise
        """
        command = command.lower().strip()
        
        if command in ['/quit', '/exit', '/q']:
            print("\n👋 Peace out! Thanks for chatting!")
            return True
        
        elif command == '/help':
            print("\n📋 Available Commands:")
            print("  /quit, /exit  - Exit the chat")
            print("  /help         - Show this help message")
            print("  /stats        - Show statistics")
            print("  /memes        - List all memes")
            print("  /clear        - Clear screen\n")
        
        elif command == '/stats':
            print(f"\n📊 Statistics:")
            print(f"  Intents: {len(self.intents)}")
            print(f"  Vocabulary: {len(self.vocabulary)} words")
            print(f"  Memes: {self.meme_handler.get_meme_count()}")
            print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}\n")
        
        elif command == '/memes':
            print(f"\n🎭 Available Memes ({self.meme_handler.get_meme_count()}):")
            for meme_name in self.meme_handler.list_all_memes():
                print(f"  • {meme_name}")
            print()
        
        elif command == '/clear':
            os.system('cls' if os.name == 'nt' else 'clear')
        
        else:
            print(ColorText.warning(f"\n⚠️ Unknown command: {command}"))
            print("Type '/help' for available commands\n")
        
        return False


def main():
    """Main entry point"""
    try:
        # Create GigaChat instance
        chatbot = GigaChat()
        
        # Load intents
        chatbot.load_intents()
        
        # Load model
        chatbot.load_model()
        
        # Start interactive mode
        chatbot.interactive_mode()
        
    except FileNotFoundError as e:
        print(ColorText.error(f"\n❌ File not found: {e}"))
        print("Make sure you've trained the model first by running: python scripts/train_model.py")
    except Exception as e:
        print(ColorText.error(f"\n❌ Error: {e}"))
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()