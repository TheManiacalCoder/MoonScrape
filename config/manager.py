import json
from pathlib import Path

class ConfigManager:
    def __init__(self):
        self.config_path = Path("config.json")
        # Initialize default_ai_model before loading config
        self.default_ai_model = "amazon/nova-micro-v1"
        self._load_config()
        
    def _load_config(self):
        try:
            with open(self.config_path) as f:
                config = json.load(f)
            self.email = config['email']
            self.api_key = config['api_key']
            self.openrouter_api_key = config.get('openrouter', {}).get('api_key', '')
            # Add search API configuration
            self.search_api_key = config.get('search', {}).get('api_key', '')
            # Use default_ai_model if not specified in config
            self.ai_model = config.get('openrouter', {}).get('ai_model', self.default_ai_model)
        except Exception as e:
            print(f"Error loading config: {e}")
            # Set fallback values if config loading fails
            self.email = ""
            self.api_key = ""
            self.openrouter_api_key = ""
            self.search_api_key = ""
            self.ai_model = self.default_ai_model 