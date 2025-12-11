"""
Response Handler for GigaChat
Manages response selection and meme database
"""

import json
import random
from pathlib import Path


class MemeResponseHandler:
    """
    Handles meme explanations with origin stories and context
    """
    
    def __init__(self, memes_file=None):
        """
        Initialize the meme handler
        
        Args:
            memes_file (str, optional): Path to memes.json file
        """
        self.meme_database = {}
        
        if memes_file and Path(memes_file).exists():
            self.load_memes_from_file(memes_file)
        else:
            self._init_default_memes()
    
    def _init_default_memes(self):
        """Initialize with default meme database"""
        self.meme_database = {
            'distracted_boyfriend': {
                'name': 'Distracted Boyfriend',
                'origin': 'Stock photo from 2015, went viral in 2017',
                'meaning': 'Represents getting distracted by something new while ignoring your current commitment',
                'usage': 'Used to show temptation, disloyalty, or changing preferences',
                'example': 'Me (boyfriend) ignoring my responsibilities (girlfriend) to scroll TikTok (other woman)'
            },
            'drakeposting': {
                'name': 'Drake Posting',
                'origin': 'From Drake\'s "Hotline Bling" music video (2015)',
                'meaning': 'Shows preference by rejecting one thing and approving another',
                'usage': 'Top panel shows Drake disapproving, bottom shows him approving',
                'example': '❌ Studying for finals | ✅ Binge-watching Netflix at 3am'
            },
            'woman_yelling_at_cat': {
                'name': 'Woman Yelling at Cat',
                'origin': 'Combines a 2018 Real Housewives scene with a confused cat photo',
                'meaning': 'Shows two opposing reactions or misunderstandings in a situation',
                'usage': 'Left side: angry/explaining, Right side: confused/unbothered',
                'example': 'Parents explaining why I need to study vs Me not caring about their lecture'
            },
            'this_is_fine': {
                'name': 'This Is Fine',
                'origin': 'From webcomic artist KC Green in 2013',
                'meaning': 'Accepting a disastrous situation while pretending everything is okay',
                'usage': 'Shows someone staying calm during chaos or denial of problems',
                'example': 'Me sitting in my room saying "this is fine" while my life falls apart'
            },
            'galaxy_brain': {
                'name': 'Galaxy Brain / Expanding Brain',
                'origin': 'Started around 2017, shows increasingly glowing brain images',
                'meaning': 'Ironically shows "ascending" levels of intelligence, usually getting dumber',
                'usage': 'Each level shows a "smarter" but actually worse idea',
                'example': 'Small brain: Study | Big brain: Cram night before | Galaxy brain: Just wing it'
            },
            'spiderman_pointing': {
                'name': 'Spider-Man Pointing at Spider-Man',
                'origin': 'From 1967 Spider-Man cartoon episode',
                'meaning': 'When two people/things are exactly the same or accusing each other',
                'usage': 'Shows similarity, hypocrisy, or mutual accusations',
                'example': 'Me and my friend both saying we\'re the responsible one in the friendship'
            },
            'is_this_a_pigeon': {
                'name': 'Is This a Pigeon?',
                'origin': 'From 1990s anime "The Brave Fighter of Sun Fighbird"',
                'meaning': 'Misidentifying something obvious or being completely wrong',
                'usage': 'Shows someone confidently being incorrect about something',
                'example': 'Me looking at any bird: "Is this a pigeon?"'
            },
            'two_buttons': {
                'name': 'Two Buttons',
                'origin': 'From comic artist Jake Clark in 2015',
                'meaning': 'Being unable to choose between two conflicting options',
                'usage': 'Shows difficult decisions or internal conflicts',
                'example': 'Button 1: Sleep early | Button 2: Watch one more episode | Me: *sweating*'
            }
        }
    
    def load_memes_from_file(self, filepath):
        """
        Load memes from a JSON file
        
        Args:
            filepath (str): Path to memes.json
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.meme_database = json.load(f)
            print(f"✅ Loaded {len(self.meme_database)} memes from {filepath}")
        except Exception as e:
            print(f"⚠️ Error loading memes from {filepath}: {e}")
            print("Using default meme database instead.")
            self._init_default_memes()
    
    def save_memes_to_file(self, filepath):
        """
        Save current meme database to JSON file
        
        Args:
            filepath (str): Path to save memes.json
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.meme_database, f, indent=2, ensure_ascii=False)
            print(f"✅ Saved {len(self.meme_database)} memes to {filepath}")
        except Exception as e:
            print(f"❌ Error saving memes to {filepath}: {e}")
    
    def get_random_meme(self):
        """
        Return random meme explanation
        
        Returns:
            str: Formatted meme explanation
        """
        if not self.meme_database:
            return "I don't have any memes in my database yet! Add some using add_meme()."
        
        meme_key = random.choice(list(self.meme_database.keys()))
        return self.format_meme_response(meme_key)
    
    def get_meme_by_key(self, key):
        """
        Get specific meme by key
        
        Args:
            key (str): Meme key/identifier
            
        Returns:
            str: Formatted meme explanation or error message
        """
        if key in self.meme_database:
            return self.format_meme_response(key)
        return f"I don't have info on '{key}' yet. Try asking for a random meme!"
    
    def format_meme_response(self, meme_key):
        """
        Format a complete meme explanation
        
        Args:
            meme_key (str): Key of meme in database
            
        Returns:
            str: Formatted meme explanation
        """
        if meme_key not in self.meme_database:
            return "I don't have info on that meme yet, but I'm always learning!"
        
        meme = self.meme_database[meme_key]
        
        response = f"""🔥 {meme['name']}

📍 Origin: {meme['origin']}

💭 Meaning: {meme['meaning']}

📱 How it's used: {meme['usage']}

💬 Example: {meme['example']}"""
        
        return response
    
    def add_meme(self, key, name, origin, meaning, usage, example):
        """
        Add new meme to database
        
        Args:
            key (str): Unique identifier for the meme
            name (str): Display name
            origin (str): Origin story
            meaning (str): What it represents
            usage (str): How it's used
            example (str): Usage example
        """
        self.meme_database[key] = {
            'name': name,
            'origin': origin,
            'meaning': meaning,
            'usage': usage,
            'example': example
        }
        print(f"✅ Added meme: {name}")
    
    def remove_meme(self, key):
        """
        Remove meme from database
        
        Args:
            key (str): Meme key to remove
        """
        if key in self.meme_database:
            name = self.meme_database[key]['name']
            del self.meme_database[key]
            print(f"🗑️ Removed meme: {name}")
        else:
            print(f"⚠️ Meme '{key}' not found in database")
    
    def get_meme_count(self):
        """
        Return number of memes in database
        
        Returns:
            int: Number of memes
        """
        return len(self.meme_database)
    
    def list_all_memes(self):
        """
        List all meme names in database
        
        Returns:
            list: List of meme names
        """
        return [meme['name'] for meme in self.meme_database.values()]
    
    def search_memes(self, query):
        """
        Search for memes by name or keyword
        
        Args:
            query (str): Search term
            
        Returns:
            list: List of matching meme keys
        """
        query_lower = query.lower()
        matches = []
        
        for key, meme in self.meme_database.items():
            if (query_lower in meme['name'].lower() or
                query_lower in meme['meaning'].lower() or
                query_lower in key.lower()):
                matches.append(key)
        
        return matches


class ResponseManager:
    """
    Manages response selection with context awareness
    """
    
    def __init__(self):
        self.response_history = []
        self.max_history = 10
    
    def select_response(self, responses, user_message="", avoid_recent=True):
        """
        Select appropriate response, optionally avoiding recent responses
        
        Args:
            responses (list): List of possible responses
            user_message (str): User's message for context
            avoid_recent (bool): Whether to avoid recently used responses
            
        Returns:
            str: Selected response
        """
        if not responses:
            return None
        
        # Filter out recently used responses if requested
        if avoid_recent and len(responses) > 1:
            available = [r for r in responses if r not in self.response_history[-3:]]
            if available:
                responses = available
        
        # Select response
        response = random.choice(responses)
        
        # Update history
        self.response_history.append(response)
        if len(self.response_history) > self.max_history:
            self.response_history.pop(0)
        
        return response
    
    def clear_history(self):
        """Clear response history"""
        self.response_history = []