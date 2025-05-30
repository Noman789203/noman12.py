import speech_recognition as sr
from gtts import gTTS
import cv2
import numpy as np
from datetime import datetime, timedelta
import json
import os
from deep_translator import GoogleTranslator
import threading
import queue
from transformers import pipeline
import torch
from PIL import Image
import logging
import sys
import pygame  # Add pygame for better audio handling
from textblob import TextBlob  # For sentiment analysis
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import wave
import pyaudio
import numpy as np
from scipy.io import wavfile
import librosa
import sounddevice as sd
from io import BytesIO

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Initialize pygame mixer for audio
pygame.mixer.init()

class AIChatbotError(Exception):
    """Custom exception for AIChatbot errors"""
    pass 

class AIChatbot:
    def __init__(self, config_file='config.json'):
        try:
            # Load configuration
            self.config = self._load_config(config_file)
            
            # Initialize token tracking
            self.token_tracking = {
                'total_tokens': 0,
                'tokens_per_request': 0,
                'max_tokens_per_request': 4000,
                'daily_token_limit': 100000,
                'tokens_used_today': 0,
                'last_reset_date': datetime.now().date(),
                'token_history': [],
                'api_calls': {
                    'total': 0,
                    'successful': 0,
                    'failed': 0
                }
            }
            
            # Initialize token usage alerts
            self.token_alerts = {
                'warning_threshold': 0.8,  # 80% of limit
                'critical_threshold': 0.9,  # 90% of limit
                'alerts_sent': {
                    'warning': False,
                    'critical': False
                }
            }
            
            # Add token-related responses
            self.token_responses = {
                'warning': {
                    'en': "Warning: You have used {percentage}% of your daily token limit.",
                    'hi': "चेतावनी: आपने अपनी दैनिक टोकन सीमा का {percentage}% उपयोग कर लिया है।",
                    'ur': "انتباہ: آپ نے اپنی روزانہ کی ٹوکن کی حد کا {percentage}% استعمال کر لیا ہے۔"
                },
                'critical': {
                    'en': "Critical: You have used {percentage}% of your daily token limit. Please upgrade your plan.",
                    'hi': "गंभीर: आपने अपनी दैनिक टोकन सीमा का {percentage}% उपयोग कर लिया है। कृपया अपनी योजना अपग्रेड करें।",
                    'ur': "تشویشناک: آپ نے اپنی روزانہ کی ٹوکن کی حد کا {percentage}% استعمال کر لیا ہے۔ براہ کرم اپنی پلان کو اپ گریڈ کریں۔"
                },
                'limit_reached': {
                    'en': "You have reached your daily token limit. Please try again tomorrow.",
                    'hi': "आपने अपनी दैनिक टोकन सीमा तक पहुंच गए हैं। कृपया कल फिर से प्रयास करें।",
                    'ur': "آپ اپنی روزانہ کی ٹوکن کی حد تک پہنچ گئے ہیں۔ براہ کرم کل دوبارہ کوشش کریں۔"
                }
            }
            
            # Initialize speech recognition
            self.recognizer = sr.Recognizer()
            self.mic = sr.Microphone()
            
            # Initialize programming language support
            self.programming_languages = {
                'python': {
                    'extensions': ['.py'],
                    'keywords': ['def', 'class', 'import', 'from', 'if', 'else', 'for', 'while'],
                    'syntax': {
                        'indentation': 'spaces',
                        'comment': '#',
                        'string': ['"', "'"],
                        'block': ':'
                    }
                },
                'javascript': {
                    'extensions': ['.js'],
                    'keywords': ['function', 'const', 'let', 'var', 'if', 'else', 'for', 'while'],
                    'syntax': {
                        'indentation': 'spaces',
                        'comment': '//',
                        'string': ['"', "'", '`'],
                        'block': '{'
                    }
                },
                'java': {
                    'extensions': ['.java'],
                    'keywords': ['public', 'class', 'void', 'static', 'if', 'else', 'for', 'while'],
                    'syntax': {
                        'indentation': 'spaces',
                        'comment': '//',
                        'string': ['"'],
                        'block': '{'
                    }
                },
                'cpp': {
                    'extensions': ['.cpp', '.hpp'],
                    'keywords': ['int', 'void', 'class', 'if', 'else', 'for', 'while'],
                    'syntax': {
                        'indentation': 'spaces',
                        'comment': '//',
                        'string': ['"'],
                        'block': '{'
                    }
                }
            }
            
            # Initialize code analysis tools
            self.code_analysis = {
                'current_language': None,
                'code_context': [],
                'syntax_errors': [],
                'suggestions': []
            }
            
            # Add programming-related responses
            self.programming_responses = {
                'code_detected': {
                    'en': "I detected {language} code. Would you like me to analyze it?",
                    'hi': "मैंने {language} कोड का पता लगाया। क्या आप चाहेंगे कि मैं इसका विश्लेषण करूं?",
                    'ur': "میں نے {language} کوڈ کا پتہ لگایا۔ کیا آپ چاہیں گے کہ میں اس کا تجزیہ کروں؟"
                },
                'syntax_error': {
                    'en': "I found a syntax error: {error}",
                    'hi': "मैंने एक सिंटैक्स त्रुटि पाई: {error}",
                    'ur': "میں نے ایک سنٹیکس غلطی پائی: {error}"
                },
                'code_suggestion': {
                    'en': "Here's a suggestion to improve your code: {suggestion}",
                    'hi': "आपके कोड को बेहतर बनाने के लिए एक सुझाव: {suggestion}",
                    'ur': "آپ کے کوڈ کو بہتر بنانے کے لیے ایک تجویز: {suggestion}"
                }
            }
            
            # Initialize audio settings
            self.audio_queue = queue.Queue()
            self.is_speaking = False
            self.audio_settings = {
                'sample_rate': 44100,
                'channels': 2,
                'chunk_size': 1024
            }
            
            # Initialize audio processing
            self.pyaudio = pyaudio.PyAudio()
            self.audio_stream = None
            
            # Initialize deep learning models with specific versions
            try:
                # Initialize transformer models for better understanding
                self.nlp = pipeline(
                    "text2text-generation",
                    model="t5-small",
                    revision="main"
                )
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    revision="main"
                )
                self.qa_model = pipeline(
                    "question-answering",
                    model="distilbert-base-cased-distilled-squad",
                    revision="main"
                )
                self.summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    revision="main"
                )
                self.classifier = pipeline(
                    "text-classification",
                    model="distilbert-base-uncased",
                    revision="main"
                )
                
                # Set device preference
                device = "cuda" if torch.cuda.is_available() else "cpu"
                for model in [self.nlp, self.sentiment_analyzer, self.qa_model, self.summarizer, self.classifier]:
                    model.device = device
                
                logging.info(f"Deep learning models initialized successfully on {device}")
            except Exception as e:
                logging.error(f"Failed to initialize deep learning models: {str(e)}")
                # Initialize with fallback models
                self._initialize_fallback_models()
                raise AIChatbotError("Model initialization failed")
            
            # Initialize camera
            self.camera = None
            self.camera_queue = queue.Queue()
            
            # Initialize language settings with more languages
            self.languages = {
                'english': 'en',
                'hindi': 'hi',
                'urdu': 'ur',
                'japanese': 'ja',
                'chinese': 'zh-CN',
                'spanish': 'es',
                'french': 'fr',
                'german': 'de',
                'arabic': 'ar',
                'bengali': 'bn',
                'tamil': 'ta',
                'telugu': 'te',
                'marathi': 'mr',
                'gujarati': 'gu',
                'kannada': 'kn',
                'malayalam': 'ml',
                'punjabi': 'pa'
            }
            self.current_language = self.config.get('default_language', 'en')
            
            # Initialize user information
            self.user_info = {
                'name': '',
                'title': '',  # Mr., Mrs., Miss, etc.
                'preferred_language': 'en',
                'last_interaction': None,
                'subject': '',  # Subject being taught
                'grade': '',    # Grade/class level
                'learning_style': ''  # Visual, auditory, etc.
            }
            
            # Initialize conversation tracking
            self.conversation_tracking = {
                'start_time': datetime.now(),
                'total_interactions': 0,
                'language_changes': 0,
                'topics_discussed': set(),
                'questions_asked': 0,
                'responses_given': 0,
                'average_response_time': 0,
                'last_interaction_time': None,
                'session_duration': 0,
                'user_engagement': {
                    'active_time': 0,
                    'idle_time': 0,
                    'last_activity': None
                },
                'learning_metrics': {
                    'accuracy': 0,
                    'response_quality': 0,
                    'user_satisfaction': 0,
                    'improvement_rate': 0
                }
            }
            
            # Initialize teacher-like responses with more variations
            self.teacher_responses = {
                'greeting': {
                    'en': "Hello {title} {name}, how can I help you today?",
                    'hi': "नमस्ते {title} {name}, मैं आपकी कैसे मदद कर सकता/सकती हूं?",
                    'ur': "السلام علیکم {title} {name}، میں آپ کی کیسے مدد کر سکتا/سکتی ہوں؟",
                    'ja': "こんにちは {title} {name}さん、お手伝いできますか？"
                },
                'not_understood': {
                    'en': "I apologize {title} {name}, but I didn't quite understand. Could you please rephrase that?",
                    'hi': "क्षमा करें {title} {name}, मैं समझ नहीं पाया/पाई। क्या आप दोबारा कह सकते हैं?",
                    'ur': "معذرت {title} {name}، میں سمجھ نہیں پایا/پائی۔ کیا آپ دوبارہ کہہ سکتے ہیں؟",
                    'ja': "申し訳ありません {title} {name}さん、理解できませんでした。もう一度言っていただけますか？"
                },
                'thinking': {
                    'en': "Let me think about that, {title} {name}...",
                    'hi': "मैं इस बारे में सोचता/सोचती हूं, {title} {name}...",
                    'ur': "میں اس بارے میں سوچتا/سوچتی ہوں، {title} {name}...",
                    'ja': "考えさせていただきます、{title} {name}さん..."
                },
                'encouragement': {
                    'en': "That's a great question, {title} {name}!",
                    'hi': "बहुत अच्छा प्रश्न है, {title} {name}!",
                    'ur': "بہت اچھا سوال ہے، {title} {name}!",
                    'ja': "素晴らしい質問ですね、{title} {name}さん！"
                },
                'clarification': {
                    'en': "Could you please explain what you mean by that, {title} {name}?",
                    'hi': "क्या आप इसका मतलब समझा सकते हैं, {title} {name}?",
                    'ur': "کیا آپ اس کا مطلب سمجھا سکتے ہیں، {title} {name}?",
                    'ja': "もう少し詳しく説明していただけますか、{title} {name}さん？"
                },
                'praise': {
                    'en': "Excellent thinking, {title} {name}!",
                    'hi': "बहुत बढ़िया सोचा, {title} {name}!",
                    'ur': "بہت عمدہ سوچا، {title} {name}!",
                    'ja': "素晴らしい考えですね、{title} {name}さん！"
                }
            }
            
            # Initialize teaching styles
            self.teaching_styles = {
                'visual': {
                    'en': "Let me show you a visual example...",
                    'hi': "मैं आपको एक दृश्य उदाहरण दिखाता/दिखाती हूं...",
                    'ur': "میں آپ کو ایک بصری مثال دکھاتا/دکھاتی ہوں...",
                    'ja': "視覚的な例をお見せしましょう..."
                },
                'auditory': {
                    'en': "Let me explain this in detail...",
                    'hi': "मैं इसे विस्तार से समझाता/समझाती हूं...",
                    'ur': "میں اسے تفصیل سے سمجھاتا/سمجھاتی ہوں...",
                    'ja': "詳しく説明させていただきます..."
                },
                'interactive': {
                    'en': "Let's work through this together...",
                    'hi': "चलिए इसे एक साथ समझते हैं...",
                    'ur': "آئیے اسے ایک ساتھ سمجھتے ہیں...",
                    'ja': "一緒に考えていきましょう..."
                }
            }
            
            # Initialize conversation memory
            self.conversation_memory = {
                'context': [],
                'user_preferences': {},
                'topics': set(),
                'difficulty_level': 'medium',
                'last_topics': []
            }
            
            # Initialize reminders and settings
            self.reminders = []
            self.user_preferences = self.config.get('user_preferences', {})
            self.conversation_history = []
            
            # Initialize voice conversation settings
            self.voice_settings = {
                'is_voice_mode': False,
                'voice_commands': {
                    'start_talking': ['start talking', 'switch to voice', 'voice mode'],
                    'stop_talking': ['stop talking', 'switch to text', 'text mode'],
                    'pause': ['pause', 'wait', 'stop'],
                    'resume': ['continue', 'resume', 'go on']
                },
                'voice_indicators': {
                    'listening': '🔊',
                    'speaking': '🔈',
                    'thinking': '💭'
                },
                'voice_prompts': {
                    'start': {
                        'en': "Voice mode activated. I'm listening...",
                        'hi': "वॉइस मोड सक्रिय। मैं सुन रहा/रही हूं...",
                        'ur': "وائس موڈ فعال۔ میں سن رہا/رہی ہوں..."
                    },
                    'stop': {
                        'en': "Switching to text mode.",
                        'hi': "टेक्स्ट मोड में स्विच कर रहा/रही हूं।",
                        'ur': "ٹیکسٹ موڈ میں سوئچ کر رہا/رہی ہوں۔"
                    }
                }
            }
            
            # Initialize voice conversation state
            self.voice_state = {
                'is_active': False,
                'is_paused': False,
                'last_voice_time': None,
                'voice_duration': 0,
                'voice_interactions': 0
            }
            
            # Load knowledge base
            self.knowledge_base = self.load_knowledge_base()
            
            # Initialize smart assistant features
            self.smart_features = {
                'reminders': [],
                'alarms': [],
                'calendar_events': [],
                'notes': [],
                'tasks': [],
                'preferences': {},
                'third_party_integrations': {
                    'weather': None,
                    'news': None,
                    'calendar': None,
                    'email': None
                }
            }
            
            logging.info("AIChatbot initialized successfully")
            
        except Exception as e:
            logging.error(f"Initialization error: {str(e)}")
            raise AIChatbotError(f"Failed to initialize AIChatbot: {str(e)}")

    def _load_config(self, config_file):
        """Load configuration from JSON file"""
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Create default configuration
                default_config = {
                    'model': 't5-small',
                    'languages': {
                        'english': 'en',
                        'spanish': 'es',
                        'french': 'fr',
                        'german': 'de',
                        'chinese': 'zh-CN'
                    },
                    'default_language': 'en',
                    'user_preferences': {},
                    'camera_settings': {
                        'resolution': [640, 480],
                        'fps': 30
                    }
                }
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=4)
                return default_config
        except Exception as e:
            logging.error(f"Error loading configuration: {str(e)}")
            raise AIChatbotError(f"Configuration error: {str(e)}")

    def load_knowledge_base(self):
        """Load knowledge base from JSON file"""
        try:
            if os.path.exists('knowledge_base.json'):
                with open('knowledge_base.json', 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Create default knowledge base
                default_kb = {
                    "general": {
                        "greetings": ["hello", "hi", "hey", "greetings"],
                        "farewells": ["goodbye", "bye", "see you", "farewell"],
                        "thanks": ["thank you", "thanks", "appreciate it"]
                    },
                    "programming": {
                        "languages": ["python", "java", "javascript", "c++", "ruby"],
                        "concepts": ["variables", "functions", "classes", "objects", "loops"]
                    },
                    "commands": {
                        "camera": ["take picture", "capture image", "photo"],
                        "reminder": ["set reminder", "remind me", "schedule"],
                        "language": ["switch to", "change language", "set language"]
                    }
                }
                with open('knowledge_base.json', 'w', encoding='utf-8') as f:
                    json.dump(default_kb, f, indent=4)
                return default_kb
        except Exception as e:
            logging.error(f"Error loading knowledge base: {str(e)}")
            raise AIChatbotError(f"Knowledge base error: {str(e)}")

    def listen(self, timeout=5, phrase_time_limit=10):
        """Listen to user's voice input with timeout and phrase time limit"""
        try:
            with self.mic as source:
                logging.info("Listening...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                try:
                    audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                    text = self.recognizer.recognize_google(audio, language=self.current_language)
                    logging.info(f"Recognized text: {text}")
                    return text.lower()
                except sr.WaitTimeoutError:
                    logging.warning("No speech detected within timeout period")
                    return ""
                except sr.UnknownValueError:
                    logging.warning("Could not understand audio")
                    return ""
                except sr.RequestError as e:
                    logging.error(f"Could not request results from speech recognition service: {str(e)}")
                    return ""
        except Exception as e:
            logging.error(f"Error in voice recognition: {str(e)}")
            return ""

    def speak(self, text, slow=False, volume=1.0):
        """Convert text to speech with enhanced audio playback and volume control"""
        try:
            if not text:
                logging.warning("Empty text provided for speech synthesis")
                return
                
            # Create audio in memory using BytesIO
            audio_buffer = BytesIO()
            tts = gTTS(text=text, lang=self.current_language, slow=slow)
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            # Load and play audio directly from memory
            pygame.mixer.music.load(audio_buffer)
            pygame.mixer.music.set_volume(volume)
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            # Clean up
            pygame.mixer.music.unload()
            audio_buffer.close()
            
        except Exception as e:
            logging.error(f"Error in speech synthesis: {str(e)}")
            raise AIChatbotError(f"Speech synthesis failed: {str(e)}")

    def start_camera(self):
        """Initialize and start camera capture with error handling"""
        try:
            if self.camera is None:
                self.camera = cv2.VideoCapture(0)
                if not self.camera.isOpened():
                    logging.error("Could not open camera")
                    raise AIChatbotError("Camera initialization failed")
                
                # Set camera properties from config
                camera_settings = self.config.get('camera_settings', {})
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_settings.get('resolution', [640, 480])[0])
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_settings.get('resolution', [640, 480])[1])
                self.camera.set(cv2.CAP_PROP_FPS, camera_settings.get('fps', 30))
                
                logging.info("Camera initialized successfully")
                return True
            return True
        except Exception as e:
            logging.error(f"Camera initialization error: {str(e)}")
            raise AIChatbotError(f"Camera error: {str(e)}")

    def capture_image(self, save_dir='captures'):
        """Capture a single image from camera with enhanced features"""
        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            if self.start_camera():
                ret, frame = self.camera.read()
                if ret:
                    # Apply basic image processing
                    frame = cv2.flip(frame, 1)  # Mirror the image
                    
                    # Add timestamp overlay
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Save image with metadata
                    filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    filepath = os.path.join(save_dir, filename)
                    cv2.imwrite(filepath, frame)
                    
                    # Save metadata
                    metadata = {
                        'timestamp': timestamp,
                        'resolution': frame.shape[:2],
                        'format': 'jpg'
                    }
                    with open(f"{filepath}.json", 'w') as f:
                        json.dump(metadata, f)
                    
                    logging.info(f"Image captured and saved as {filepath}")
                    return filepath
                else:
                    logging.error("Failed to capture frame from camera")
                    return None
        except Exception as e:
            logging.error(f"Error capturing image: {str(e)}")
            raise AIChatbotError(f"Image capture failed: {str(e)}")

    def stop_camera(self):
        """Safely stop and release camera resources"""
        try:
            if self.camera is not None:
                self.camera.release()
                self.camera = None
                cv2.destroyAllWindows()
                logging.info("Camera resources released")
        except Exception as e:
            logging.error(f"Error releasing camera resources: {str(e)}")
            raise AIChatbotError(f"Camera cleanup failed: {str(e)}")

    def set_reminder(self, task, time_str, repeat=None, priority='medium'):
        """Set a reminder with enhanced features"""
        try:
            reminder_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
            if reminder_time < datetime.now():
                raise ValueError("Reminder time cannot be in the past")
                
            reminder = {
                "task": task,
                "time": time_str,
                "repeat": repeat,  # None, 'daily', 'weekly', 'monthly'
                "priority": priority,  # 'low', 'medium', 'high'
                "completed": False,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "last_modified": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.reminders.append(reminder)
            self._save_reminders()
            
            logging.info(f"Reminder set: {task} at {time_str}")
            return f"Reminder set for {task} at {time_str}"
            
        except ValueError as e:
            logging.error(f"Invalid reminder time format: {str(e)}")
            return "Please provide time in format: YYYY-MM-DD HH:MM"
        except Exception as e:
            logging.error(f"Error setting reminder: {str(e)}")
            raise AIChatbotError(f"Failed to set reminder: {str(e)}")

    def check_reminders(self):
        """Check for due reminders with enhanced functionality"""
        try:
            current_time = datetime.now()
            due_reminders = []
            
            for reminder in self.reminders:
                if not reminder['completed']:
                    reminder_time = datetime.strptime(reminder['time'], "%Y-%m-%d %H:%M")
                    
                    # Check if reminder is due
                    if reminder_time <= current_time:
                        due_reminders.append(reminder)
                        
                        # Handle repeating reminders
                        if reminder['repeat']:
                            self._update_repeating_reminder(reminder)
                        else:
                            reminder['completed'] = True
            
            if due_reminders:
                self._save_reminders()
                
            return due_reminders
            
        except Exception as e:
            logging.error(f"Error checking reminders: {str(e)}")
            raise AIChatbotError(f"Failed to check reminders: {str(e)}")

    def _update_repeating_reminder(self, reminder):
        """Update the time for repeating reminders"""
        try:
            current_time = datetime.strptime(reminder['time'], "%Y-%m-%d %H:%M")
            
            if reminder['repeat'] == 'daily':
                new_time = current_time + timedelta(days=1)
            elif reminder['repeat'] == 'weekly':
                new_time = current_time + timedelta(weeks=1)
            elif reminder['repeat'] == 'monthly':
                # Add one month, handling month boundaries
                if current_time.month == 12:
                    new_time = current_time.replace(year=current_time.year + 1, month=1)
                else:
                    new_time = current_time.replace(month=current_time.month + 1)
            
            reminder['time'] = new_time.strftime("%Y-%m-%d %H:%M")
            reminder['last_modified'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
        except Exception as e:
            logging.error(f"Error updating repeating reminder: {str(e)}")
            raise AIChatbotError(f"Failed to update repeating reminder: {str(e)}")

    def _save_reminders(self):
        """Save reminders to file"""
        try:
            with open('reminders.json', 'w', encoding='utf-8') as f:
                json.dump(self.reminders, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving reminders: {str(e)}")
            raise AIChatbotError(f"Failed to save reminders: {str(e)}")

    def load_reminders(self):
        """Load reminders from file"""
        try:
            if os.path.exists('reminders.json'):
                with open('reminders.json', 'r', encoding='utf-8') as f:
                    self.reminders = json.load(f)
        except Exception as e:
            logging.error(f"Error loading reminders: {str(e)}")
            raise AIChatbotError(f"Failed to load reminders: {str(e)}")

    def analyze_emotion(self, text):
        """Analyze emotion in text"""
        try:
            # Get sentiment
            sentiment = self.sentiment_analyzer(text)[0]
            
            # Get TextBlob sentiment for more detailed analysis
            blob = TextBlob(text)
            
            # Extract keywords
            tokens = word_tokenize(text.lower())
            stop_words = set(stopwords.words('english'))
            keywords = [word for word in tokens if word.isalnum() and word not in stop_words]
            
            # Extract topics
            self.conversation_memory['topics'].update(keywords)
            
            emotion_data = {
                'sentiment': sentiment['label'],
                'confidence': sentiment['score'],
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity,
                'keywords': keywords,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.conversation_memory['emotion_history'].append(emotion_data)
            return emotion_data
            
        except Exception as e:
            logging.error(f"Error analyzing emotion: {str(e)}")
            return None

    def get_emotion_response(self, emotion_data):
        """Generate response based on emotion analysis"""
        try:
            if emotion_data['sentiment'] == 'POSITIVE':
                if emotion_data['polarity'] > 0.5:
                    return "I'm glad you're feeling positive! "
                return "That's good to hear! "
            elif emotion_data['sentiment'] == 'NEGATIVE':
                if emotion_data['polarity'] < -0.5:
                    return "I'm sorry you're feeling down. "
                return "I understand this might be difficult. "
            return ""
        except Exception as e:
            logging.error(f"Error generating emotion response: {str(e)}")
            return ""

    def process_voice_command(self, text):
        """Process voice commands with enhanced recognition"""
        try:
            text = text.lower()
            
            # Check for voice commands
            for command_type, phrases in self.voice_settings['voice_commands'].items():
                if any(phrase in text for phrase in phrases):
                    if command_type == 'take_picture':
                        return self.capture_image()
                    elif command_type == 'set_reminder':
                        return self._process_reminder_command(text)
                    elif command_type == 'change_language':
                        return self._process_language_command(text)
                    elif command_type == 'help':
                        return self._get_help_text()
            
            return None
        except Exception as e:
            logging.error(f"Error processing voice command: {str(e)}")
            return None

    def _process_reminder_command(self, text):
        """Process reminder voice command"""
        try:
            # Extract task and time from voice command
            parts = text.split(" for ")
            if len(parts) > 1:
                task = parts[1].split(" at ")[0]
                time_str = parts[1].split(" at ")[1]
                return self.set_reminder(task, time_str)
            return "Please specify the task and time for the reminder."
        except Exception as e:
            logging.error(f"Error processing reminder command: {str(e)}")
            return "I couldn't understand the reminder details. Please try again."

    def _process_language_command(self, text):
        """Process language change voice command"""
        try:
            for lang_name, lang_code in self.languages.items():
                if f"switch to {lang_name}" in text:
                    self.current_language = lang_code
                    return f"Switched to {lang_name}"
            return "I couldn't understand which language to switch to."
        except Exception as e:
            logging.error(f"Error processing language command: {str(e)}")
            return "I had trouble changing the language. Please try again."

    def process_command(self, text):
        """Process user commands with fallback handling"""
        try:
            start_time = datetime.now()
            
            # Check for voice mode commands
            if self._process_voice_command(text):
                return ""
            
            # Add to conversation history
            self.conversation_history.append({
                "user": text,
                "timestamp": start_time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Update tracking for question
            if '?' in text or any(word in text.lower() for word in ['how', 'what', 'why', 'when', 'where']):
                self.update_conversation_tracking('question')
            
            # Check for name/title in input
            self._update_user_info_from_text(text)
            
            # Add encouragement for good questions
            if '?' in text or any(word in text.lower() for word in ['how', 'what', 'why', 'when', 'where']):
                encouragement = self.get_teacher_response('encouragement')
                if encouragement:
                    self.speak(encouragement, volume=0.8)
            
            # Generate response using NLP with fallback
            try:
                # Add thinking response
                thinking_response = self.get_teacher_response('thinking')
                if thinking_response:
                    self.speak(thinking_response, volume=0.8)
                
                if self.nlp:
                    response = self.nlp(text)[0]['generated_text']
                else:
                    # Fallback to basic response generation
                    response = self._generate_basic_response(text)
                
                if not response or response.strip() == "":
                    response = self.get_teacher_response('not_understood')
                else:
                    # Add teaching style specific response
                    if self.user_info['learning_style']:
                        style_response = self.teaching_styles[self.user_info['learning_style']].get(self.current_language)
                        if style_response:
                            response = f"{style_response}\n{response}"
            except Exception as e:
                logging.error(f"NLP generation error: {str(e)}")
                response = self._generate_basic_response(text)

            # Add response to history and update tracking
            self._add_to_history(response)
            response_time = (datetime.now() - start_time).total_seconds()
            self.update_conversation_tracking('response', {'response_time': response_time})
            
            return response

        except Exception as e:
            logging.error(f"Error processing command: {str(e)}")
            raise AIChatbotError(f"Command processing failed: {str(e)}")

    def _add_to_history(self, response):
        """Add bot response to conversation history"""
        self.conversation_history.append({
            "bot": response,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    def _get_help_text(self):
        """Get help text with available commands"""
        help_text = """
Available commands:
1. Language:
   - "switch to [language]" (english, spanish, french, german, chinese)

2. Camera:
   - "take picture" or "capture image"

3. Reminders:
   - "set reminder for [task] at [YYYY-MM-DD HH:MM]"

4. General:
   - "help" - Show this help message
   - "quit" - Exit the program

You can also speak commands by pressing Enter when prompted.
"""
        return help_text

    def _generate_basic_response(self, text):
        """Generate basic response when NLP models are unavailable"""
        try:
            # Simple response generation based on keywords
            text = text.lower()
            
            if 'hello' in text or 'hi' in text:
                return "Hello! How can I help you today?"
            elif 'how are you' in text:
                return "I'm doing well, thank you for asking! How can I assist you?"
            elif 'thank' in text:
                return "You're welcome! Is there anything else I can help you with?"
            elif 'bye' in text or 'goodbye' in text:
                return "Goodbye! Have a great day!"
            elif '?' in text:
                return "That's an interesting question. Could you please provide more details?"
            else:
                return "I understand. Please tell me more about that."
                
        except Exception as e:
            logging.error(f"Error generating basic response: {str(e)}")
            return "I'm having trouble understanding. Could you please rephrase that?"

    def get_conversation_summary(self):
        """Generate a summary of the conversation"""
        try:
            if not self.conversation_memory['emotion_history']:
                return "No conversation history available."
                
            # Calculate average sentiment
            sentiments = [e['sentiment'] for e in self.conversation_memory['emotion_history']]
            avg_sentiment = max(set(sentiments), key=sentiments.count)
            
            # Get most common topics
            topics = list(self.conversation_memory['topics'])[:5]
            
            summary = f"""
Conversation Summary:
- Overall sentiment: {avg_sentiment}
- Main topics discussed: {', '.join(topics)}
- Number of exchanges: {len(self.conversation_history)}
- Current language: {self.current_language}
"""
            return summary
        except Exception as e:
            logging.error(f"Error generating conversation summary: {str(e)}")
            return "Error generating conversation summary."

    def run(self):
        """Main loop for running the chatbot with voice mode support"""
        try:
            print("AI Chatbot initialized. Type 'quit' to exit.")
            print("Type 'help' for available commands.")
            print("Say 'start talking' to begin voice conversation.")
            
            # Load saved reminders
            self.load_reminders()
            
            while True:
                try:
                    # Check for due reminders
                    due_reminders = self.check_reminders()
                    for reminder in due_reminders:
                        reminder_msg = f"\nREMINDER: {reminder['task']}"
                        print(reminder_msg)
                        self.speak(reminder_msg)

                    # Get user input
                    if not self.voice_state['is_active']:
                        user_input = input("\nYou (type or press Enter to speak): ")
                        
                        if user_input.lower() == 'quit':
                            print("Goodbye!")
                            break
                        
                        # If no text input, try voice
                        if not user_input:
                            user_input = self.listen()
                            if user_input:
                                print(f"You said: {user_input}")

                    if user_input:
                        # Process input and get response
                        response = self.process_command(user_input)
                        
                        # Translate if not in English
                        if self.current_language != 'en':
                            response = self.translate(response, self.current_language)
                        
                        # Output response
                        if not self.voice_state['is_active']:
                            print(f"Bot: {response}")
                        self.speak(response)

                except KeyboardInterrupt:
                    print("\nOperation cancelled by user")
                    break
                except Exception as e:
                    logging.error(f"Error in main loop: {str(e)}")
                    print(f"An error occurred: {str(e)}")
                    continue

        except Exception as e:
            logging.error(f"Fatal error in main loop: {str(e)}")
            raise AIChatbotError(f"Chatbot execution failed: {str(e)}")
        finally:
            # Cleanup
            self.stop_voice_conversation()
            self.stop_camera()
            self.save_knowledge_base()
            self._save_reminders()
            logging.info("Chatbot shutdown complete")

    def translate(self, text, target_lang):
        """Translate text to target language with enhanced error handling"""
        try:
            if not text:
                logging.warning("Empty text provided for translation")
                return text
                
            if target_lang == 'en':
                return text
                
            try:
                translator = GoogleTranslator(source='auto', target=target_lang)
                translated_text = translator.translate(text)
                logging.info(f"Text translated to {target_lang}")
                return translated_text
            except Exception as e:
                logging.error(f"Translation error: {str(e)}")
                return text
                
        except Exception as e:
            logging.error(f"Translation service error: {str(e)}")
            raise AIChatbotError(f"Translation failed: {str(e)}")

    def save_knowledge_base(self):
        """Save knowledge base to JSON file"""
        try:
            with open('knowledge_base.json', 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, indent=4)
            logging.info("Knowledge base saved successfully")
        except Exception as e:
            logging.error(f"Error saving knowledge base: {str(e)}")
            raise AIChatbotError(f"Failed to save knowledge base: {str(e)}")

    def update_knowledge_base(self, category, key, value):
        """Update knowledge base with new information"""
        try:
            if category not in self.knowledge_base:
                self.knowledge_base[category] = {}
            
            if isinstance(value, list):
                if key not in self.knowledge_base[category]:
                    self.knowledge_base[category][key] = []
                self.knowledge_base[category][key].extend(value)
            else:
                self.knowledge_base[category][key] = value
                
            self.save_knowledge_base()
            logging.info(f"Knowledge base updated: {category}.{key}")
        except Exception as e:
            logging.error(f"Error updating knowledge base: {str(e)}")
            raise AIChatbotError(f"Failed to update knowledge base: {str(e)}")

    def search_knowledge_base(self, query):
        """Search knowledge base for information"""
        try:
            results = []
            query = query.lower()
            
            for category, data in self.knowledge_base.items():
                for key, value in data.items():
                    if isinstance(value, list):
                        matches = [item for item in value if query in item.lower()]
                        if matches:
                            results.append({
                                'category': category,
                                'key': key,
                                'matches': matches
                            })
                    elif isinstance(value, str) and query in value.lower():
                        results.append({
                            'category': category,
                            'key': key,
                            'matches': [value]
                        })
            
            return results
        except Exception as e:
            logging.error(f"Error searching knowledge base: {str(e)}")
            raise AIChatbotError(f"Knowledge base search failed: {str(e)}")

    def set_user_info(self, name, title=''):
        """Set user information with proper title"""
        try:
            self.user_info['name'] = name
            self.user_info['title'] = title
            self.user_info['last_interaction'] = datetime.now()
            logging.info(f"User info set: {title} {name}")
        except Exception as e:
            logging.error(f"Error setting user info: {str(e)}")

    def get_teacher_response(self, response_type, language=None):
        """Get a teacher-like response in the specified language"""
        try:
            if language is None:
                language = self.current_language
                
            # Get response template
            response = self.teacher_responses.get(response_type, {}).get(language)
            if not response:
                # Fallback to English if translation not available
                response = self.teacher_responses.get(response_type, {}).get('en')
            
            # Format response with user info
            return response.format(
                title=self.user_info['title'],
                name=self.user_info['name']
            )
        except Exception as e:
            logging.error(f"Error getting teacher response: {str(e)}")
            return ""

    def _update_user_info_from_text(self, text):
        """Update user information from text input"""
        try:
            # Check for name/title patterns
            name_patterns = [
                r'(?:my name is|i am|i\'m) (\w+)',
                r'(?:call me|please call me) (\w+)',
                r'(?:this is) (\w+)'
            ]
            
            title_patterns = [
                r'(?:i am|i\'m) (mr\.|mrs\.|miss|ms\.|sir|madam)',
                r'(?:call me) (mr\.|mrs\.|miss|ms\.|sir|madam)'
            ]
            
            # Check for name
            for pattern in name_patterns:
                match = re.search(pattern, text.lower())
                if match:
                    self.user_info['name'] = match.group(1)
                    break
            
            # Check for title
            for pattern in title_patterns:
                match = re.search(pattern, text.lower())
                if match:
                    self.user_info['title'] = match.group(1)
                    break
                    
        except Exception as e:
            logging.error(f"Error updating user info: {str(e)}")

    def set_teaching_style(self, style):
        """Set the teaching style (visual, auditory, interactive)"""
        try:
            if style in self.teaching_styles:
                self.user_info['learning_style'] = style
                response = self.teaching_styles[style].get(self.current_language)
                if response:
                    return response
            return f"Teaching style set to {style}"
        except Exception as e:
            logging.error(f"Error setting teaching style: {str(e)}")
            return ""

    def set_difficulty_level(self, level):
        """Set the difficulty level of explanations"""
        try:
            if level in ['easy', 'medium', 'hard']:
                self.conversation_memory['difficulty_level'] = level
                return f"Difficulty level set to {level}"
            return "Please specify a valid difficulty level (easy, medium, hard)"
        except Exception as e:
            logging.error(f"Error setting difficulty level: {str(e)}")
            return ""

    def update_conversation_tracking(self, interaction_type, data=None):
        """Update conversation tracking metrics"""
        try:
            current_time = datetime.now()
            
            # Update basic metrics
            self.conversation_tracking['total_interactions'] += 1
            self.conversation_tracking['last_interaction_time'] = current_time
            
            # Calculate session duration
            self.conversation_tracking['session_duration'] = (current_time - self.conversation_tracking['start_time']).total_seconds()
            
            # Update specific metrics based on interaction type
            if interaction_type == 'language_change':
                self.conversation_tracking['language_changes'] += 1
            elif interaction_type == 'question':
                self.conversation_tracking['questions_asked'] += 1
            elif interaction_type == 'response':
                self.conversation_tracking['responses_given'] += 1
                if data and 'response_time' in data:
                    # Update average response time
                    current_avg = self.conversation_tracking['average_response_time']
                    new_response_time = data['response_time']
                    self.conversation_tracking['average_response_time'] = (
                        (current_avg * (self.conversation_tracking['responses_given'] - 1) + new_response_time) /
                        self.conversation_tracking['responses_given']
                    )
            
            # Update user engagement
            if self.conversation_tracking['user_engagement']['last_activity']:
                time_since_last = (current_time - self.conversation_tracking['user_engagement']['last_activity']).total_seconds()
                if time_since_last < 300:  # 5 minutes threshold for active time
                    self.conversation_tracking['user_engagement']['active_time'] += time_since_last
                else:
                    self.conversation_tracking['user_engagement']['idle_time'] += time_since_last
            
            self.conversation_tracking['user_engagement']['last_activity'] = current_time
            
        except Exception as e:
            logging.error(f"Error updating conversation tracking: {str(e)}")

    def get_conversation_stats(self):
        """Get current conversation statistics"""
        try:
            stats = {
                'session_duration': f"{self.conversation_tracking['session_duration'] / 60:.2f} minutes",
                'total_interactions': self.conversation_tracking['total_interactions'],
                'questions_asked': self.conversation_tracking['questions_asked'],
                'responses_given': self.conversation_tracking['responses_given'],
                'average_response_time': f"{self.conversation_tracking['average_response_time']:.2f} seconds",
                'language_changes': self.conversation_tracking['language_changes'],
                'user_engagement': {
                    'active_time': f"{self.conversation_tracking['user_engagement']['active_time'] / 60:.2f} minutes",
                    'idle_time': f"{self.conversation_tracking['user_engagement']['idle_time'] / 60:.2f} minutes",
                    'engagement_rate': f"{(self.conversation_tracking['user_engagement']['active_time'] / self.conversation_tracking['session_duration'] * 100):.2f}%"
                }
            }
            return stats
        except Exception as e:
            logging.error(f"Error getting conversation stats: {str(e)}")
            return {}

    def process_smart_command(self, text):
        """Process smart assistant commands"""
        try:
            text = text.lower()
            
            # Check for reminder commands
            if any(phrase in text for phrase in ['remind me', 'set reminder', 'schedule']):
                return self._handle_reminder_command(text)
            
            # Check for alarm commands
            elif any(phrase in text for phrase in ['set alarm', 'wake me up', 'alarm']):
                return self._handle_alarm_command(text)
            
            # Check for calendar commands
            elif any(phrase in text for phrase in ['add to calendar', 'schedule meeting', 'calendar']):
                return self._handle_calendar_command(text)
            
            # Check for note commands
            elif any(phrase in text for phrase in ['take note', 'remember', 'note down']):
                return self._handle_note_command(text)
            
            # Check for task commands
            elif any(phrase in text for phrase in ['add task', 'to-do', 'task']):
                return self._handle_task_command(text)
            
            # Check for third-party integrations
            elif any(phrase in text for phrase in ['weather', 'forecast']):
                return self._handle_weather_command(text)
            elif any(phrase in text for phrase in ['news', 'headlines']):
                return self._handle_news_command(text)
            elif any(phrase in text for phrase in ['email', 'check mail']):
                return self._handle_email_command(text)
            
            return None
            
        except Exception as e:
            logging.error(f"Error processing smart command: {str(e)}")
            return None

    def _handle_reminder_command(self, text):
        """Handle reminder-related commands"""
        try:
            # Extract time and task from text
            time_pattern = r'at (\d{1,2}(?::\d{2})? ?(?:am|pm)?)'
            task_pattern = r'to (.*?)(?: at|$)'
            
            time_match = re.search(time_pattern, text)
            task_match = re.search(task_pattern, text)
            
            if time_match and task_match:
                time_str = time_match.group(1)
                task = task_match.group(1)
                
                # Add to reminders
                reminder = {
                    'task': task,
                    'time': time_str,
                    'created_at': datetime.now(),
                    'status': 'pending'
                }
                self.smart_features['reminders'].append(reminder)
                
                return f"I'll remind you to {task} at {time_str}"
            return "I couldn't understand the reminder details. Please try again."
            
        except Exception as e:
            logging.error(f"Error handling reminder command: {str(e)}")
            return "Sorry, I had trouble setting the reminder."

    def _handle_alarm_command(self, text):
        """Handle alarm-related commands"""
        try:
            # Extract time from text
            time_pattern = r'at (\d{1,2}(?::\d{2})? ?(?:am|pm)?)'
            time_match = re.search(time_pattern, text)
            
            if time_match:
                time_str = time_match.group(1)
                
                # Add to alarms
                alarm = {
                    'time': time_str,
                    'created_at': datetime.now(),
                    'status': 'pending'
                }
                self.smart_features['alarms'].append(alarm)
                
                return f"Alarm set for {time_str}"
            return "I couldn't understand the alarm time. Please try again."
            
        except Exception as e:
            logging.error(f"Error handling alarm command: {str(e)}")
            return "Sorry, I had trouble setting the alarm."

    def _handle_calendar_command(self, text):
        """Handle calendar-related commands"""
        try:
            # Extract event details from text
            event_pattern = r'add (.*?) to calendar'
            time_pattern = r'at (\d{1,2}(?::\d{2})? ?(?:am|pm)?)'
            
            event_match = re.search(event_pattern, text)
            time_match = re.search(time_pattern, text)
            
            if event_match and time_match:
                event = event_match.group(1)
                time_str = time_match.group(1)
                
                # Add to calendar events
                calendar_event = {
                    'event': event,
                    'time': time_str,
                    'created_at': datetime.now(),
                    'status': 'scheduled'
                }
                self.smart_features['calendar_events'].append(calendar_event)
                
                return f"Added {event} to your calendar for {time_str}"
            return "I couldn't understand the calendar event details. Please try again."
            
        except Exception as e:
            logging.error(f"Error handling calendar command: {str(e)}")
            return "Sorry, I had trouble adding the calendar event."

    def _handle_note_command(self, text):
        """Handle note-taking commands"""
        try:
            # Extract note content
            note_pattern = r'note (.*?)(?: at|$)'
            note_match = re.search(note_pattern, text)
            
            if note_match:
                note_content = note_match.group(1)
                
                # Add to notes
                note = {
                    'content': note_content,
                    'created_at': datetime.now(),
                    'tags': []
                }
                self.smart_features['notes'].append(note)
                
                return f"I've noted down: {note_content}"
            return "I couldn't understand what to note down. Please try again."
            
        except Exception as e:
            logging.error(f"Error handling note command: {str(e)}")
            return "Sorry, I had trouble taking the note."

    def _handle_task_command(self, text):
        """Handle task-related commands"""
        try:
            # Extract task details
            task_pattern = r'add task (.*?)(?: at|$)'
            task_match = re.search(task_pattern, text)
            
            if task_match:
                task_content = task_match.group(1)
                
                # Add to tasks
                task = {
                    'content': task_content,
                    'created_at': datetime.now(),
                    'status': 'pending',
                    'priority': 'medium'
                }
                self.smart_features['tasks'].append(task)
                
                return f"Added task: {task_content}"
            return "I couldn't understand the task details. Please try again."
            
        except Exception as e:
            logging.error(f"Error handling task command: {str(e)}")
            return "Sorry, I had trouble adding the task."

    def _handle_weather_command(self, text):
        """Handle weather-related commands"""
        try:
            # Placeholder for weather API integration
            return "Weather feature coming soon!"
        except Exception as e:
            logging.error(f"Error handling weather command: {str(e)}")
            return "Sorry, I couldn't get the weather information."

    def _handle_news_command(self, text):
        """Handle news-related commands"""
        try:
            # Placeholder for news API integration
            return "News feature coming soon!"
        except Exception as e:
            logging.error(f"Error handling news command: {str(e)}")
            return "Sorry, I couldn't get the news."

    def _handle_email_command(self, text):
        """Handle email-related commands"""
        try:
            # Placeholder for email integration
            return "Email feature coming soon!"
        except Exception as e:
            logging.error(f"Error handling email command: {str(e)}")
            return "Sorry, I couldn't access your email."

    def update_learning_metrics(self, response_quality, user_satisfaction):
        """Update deep learning metrics"""
        try:
            metrics = self.conversation_tracking['learning_metrics']
            
            # Update accuracy based on response quality
            metrics['accuracy'] = (metrics['accuracy'] * 0.9) + (response_quality * 0.1)
            
            # Update response quality
            metrics['response_quality'] = (metrics['response_quality'] * 0.9) + (response_quality * 0.1)
            
            # Update user satisfaction
            metrics['user_satisfaction'] = (metrics['user_satisfaction'] * 0.9) + (user_satisfaction * 0.1)
            
            # Calculate improvement rate
            metrics['improvement_rate'] = (
                (metrics['accuracy'] + metrics['response_quality'] + metrics['user_satisfaction']) / 3
            )
            
        except Exception as e:
            logging.error(f"Error updating learning metrics: {str(e)}")

    def detect_programming_language(self, text):
        """Detect programming language from text"""
        try:
            # Check for language-specific keywords
            for lang, info in self.programming_languages.items():
                keyword_count = sum(1 for keyword in info['keywords'] if keyword in text.lower())
                if keyword_count >= 2:  # If at least 2 keywords are found
                    self.code_analysis['current_language'] = lang
                    return lang
            return None
        except Exception as e:
            logging.error(f"Error detecting programming language: {str(e)}")
            return None

    def analyze_code(self, code, language):
        """Analyze code for syntax and provide suggestions"""
        try:
            if language not in self.programming_languages:
                return "Unsupported programming language"
            
            # Get language syntax rules
            syntax = self.programming_languages[language]['syntax']
            
            # Basic syntax checking
            lines = code.split('\n')
            errors = []
            suggestions = []
            
            for i, line in enumerate(lines, 1):
                # Check indentation
                if line.strip() and not line.startswith(' ' * 4) and syntax['indentation'] == 'spaces':
                    errors.append(f"Line {i}: Incorrect indentation")
                
                # Check for unclosed strings
                for quote in syntax['string']:
                    if line.count(quote) % 2 != 0:
                        errors.append(f"Line {i}: Unclosed string")
                
                # Check for common mistakes
                if 'if' in line and not any(block in line for block in [':', '{']):
                    errors.append(f"Line {i}: Missing block delimiter")
            
            # Add suggestions for improvement
            if len(lines) > 10:
                suggestions.append("Consider breaking down the code into smaller functions")
            if any(len(line) > 80 for line in lines):
                suggestions.append("Some lines are too long, consider breaking them into multiple lines")
            
            self.code_analysis['syntax_errors'] = errors
            self.code_analysis['suggestions'] = suggestions
            
            return {
                'errors': errors,
                'suggestions': suggestions
            }
            
        except Exception as e:
            logging.error(f"Error analyzing code: {str(e)}")
            return None

    def start_voice_conversation(self):
        """Start voice conversation mode"""
        try:
            self.voice_state['is_active'] = True
            self.voice_state['is_paused'] = False
            self.voice_state['last_voice_time'] = datetime.now()
            
            # Announce voice mode activation
            prompt = self.voice_settings['voice_prompts']['start'][self.current_language]
            print(f"{self.voice_settings['voice_indicators']['listening']} {prompt}")
            self.speak(prompt)
            
            # Start continuous listening
            while self.voice_state['is_active'] and not self.voice_state['is_paused']:
                try:
                    # Listen for user input
                    user_input = self.listen()
                    
                    if user_input:
                        # Process voice commands first
                        if self._process_voice_command(user_input):
                            continue
                        
                        # Process normal conversation
                        response = self.process_command(user_input)
                        
                        # Speak the response
                        print(f"{self.voice_settings['voice_indicators']['speaking']} Bot: {response}")
                        self.speak(response)
                        
                        # Update voice state
                        self.voice_state['voice_interactions'] += 1
                        self.voice_state['voice_duration'] = (datetime.now() - self.voice_state['last_voice_time']).total_seconds()
                        
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    logging.error(f"Error in voice conversation: {str(e)}")
                    continue
                    
        except Exception as e:
            logging.error(f"Error starting voice conversation: {str(e)}")
            self.stop_voice_conversation()

    def stop_voice_conversation(self):
        """Stop voice conversation mode"""
        try:
            if self.voice_state['is_active']:
                self.voice_state['is_active'] = False
                self.voice_state['is_paused'] = False
                
                # Announce mode switch
                prompt = self.voice_settings['voice_prompts']['stop'][self.current_language]
                print(f"{self.voice_settings['voice_indicators']['speaking']} {prompt}")
                self.speak(prompt)
                
        except Exception as e:
            logging.error(f"Error stopping voice conversation: {str(e)}")

    def pause_voice_conversation(self):
        """Pause voice conversation"""
        try:
            if self.voice_state['is_active'] and not self.voice_state['is_paused']:
                self.voice_state['is_paused'] = True
                self.speak("Conversation paused. Say 'continue' to resume.")
        except Exception as e:
            logging.error(f"Error pausing voice conversation: {str(e)}")

    def resume_voice_conversation(self):
        """Resume voice conversation"""
        try:
            if self.voice_state['is_active'] and self.voice_state['is_paused']:
                self.voice_state['is_paused'] = False
                self.speak("Resuming conversation...")
        except Exception as e:
            logging.error(f"Error resuming voice conversation: {str(e)}")

    def _process_voice_command(self, text):
        """Process voice-specific commands"""
        try:
            text = text.lower()
            
            # Check for voice mode commands
            if any(cmd in text for cmd in self.voice_settings['voice_commands']['start_talking']):
                if not self.voice_state['is_active']:
                    self.start_voice_conversation()
                return True
                
            elif any(cmd in text for cmd in self.voice_settings['voice_commands']['stop_talking']):
                if self.voice_state['is_active']:
                    self.stop_voice_conversation()
                return True
                
            elif any(cmd in text for cmd in self.voice_settings['voice_commands']['pause']):
                if self.voice_state['is_active']:
                    self.pause_voice_conversation()
                return True
                
            elif any(cmd in text for cmd in self.voice_settings['voice_commands']['resume']):
                if self.voice_state['is_active']:
                    self.resume_voice_conversation()
                return True
                
            return False
            
        except Exception as e:
            logging.error(f"Error processing voice command: {str(e)}")
            return False

    def _initialize_fallback_models(self):
        """Initialize fallback models when primary models fail"""
        try:
            logging.info("Initializing fallback models...")
            
            # Initialize basic text processing
            self.nlp = None
            self.sentiment_analyzer = None
            self.qa_model = None
            self.summarizer = None
            self.classifier = None
            
            # Use TextBlob for basic sentiment analysis
            self.textblob = TextBlob
            
            logging.info("Fallback models initialized")
        except Exception as e:
            logging.error(f"Failed to initialize fallback models: {str(e)}")
            raise AIChatbotError("Fallback model initialization failed")

def create_chatbot(config_file='config.json'):
    """Factory function to create and initialize a chatbot instance"""
    try:
        return AIChatbot(config_file)
    except Exception as e:
        logging.error(f"Error creating chatbot: {str(e)}")
        raise

def main():
    """Main entry point for running the chatbot directly"""
    try:
        # Create a simple command-line interface
        print("Initializing AI Chatbot...")
        chatbot = create_chatbot()
        
        print("\nAI Chatbot is ready!")
        print("Type 'help' for available commands")
        print("Press Enter to use voice input")
        print("Type 'quit' to exit\n")
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                
                # If no text input, try voice
                if not user_input:
                    print("Listening... (speak now)")
                    user_input = chatbot.listen()
                    if user_input:
                        print(f"You said: {user_input}")

                if user_input:
                    # Process the input
                    response = chatbot.process_command(user_input)
                    
                    # Translate if not in English
                    if chatbot.current_language != 'en':
                        response = chatbot.translate(response, chatbot.current_language)
                    
                    # Output response
                    print(f"Bot: {response}")
                    chatbot.speak(response)

            except KeyboardInterrupt:
                print("\nOperation cancelled by user")
                break
            except Exception as e:
                logging.error(f"Error in main loop: {str(e)}")
                print(f"An error occurred: {str(e)}")
                continue
                
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        print(f"Fatal error: {str(e)}")
        sys.exit(1)
    finally:
        # Cleanup
        if 'chatbot' in locals():
            chatbot.stop_voice_conversation()
            chatbot.stop_camera()
            chatbot.save_knowledge_base()
            chatbot._save_reminders()
        logging.info("Chatbot shutdown complete")

if __name__ == "__main__":
    main() 
