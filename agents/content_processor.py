import re
import aiohttp
import asyncio
from typing import List, Dict
from colorama import Fore, Style
from config.manager import ConfigManager

class ContentProcessor:
    def __init__(self, db):
        self.db = db
        self.config = ConfigManager()
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
            "Content-Type": "application/json"
        }

    async def extract_key_points(self, content: str) -> str:
        prompt = f"""
        Content Analysis Requirements:
        1. Extract factual SEO elements
        2. Identify key points
        3. Maintain temporal consistency
        
        {content}
        """
        
        payload = {
            "model": self.config.ai_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 10000
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=self.headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                else:
                    error = await response.text()
                    print(f"{Fore.RED}AI extraction error: {error}{Style.RESET_ALL}")
                    return None

    async def process_urls(self, urls: List[str]) -> Dict[str, str]:
        processed_data = {}
        for url in urls:
            try:
                with self.db.conn:
                    cursor = self.db.conn.cursor()
                    cursor.execute('''SELECT content FROM seo_content 
                                   JOIN urls ON seo_content.url_id = urls.id 
                                   WHERE urls.url = ?''', (url,))
                    result = cursor.fetchone()
                    if result:
                        processed_data[url] = result[0]
                        print(f"{Fore.GREEN}RAW CONTENT SAVED: {url}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}ERROR IGNORED: {str(e)}{Style.RESET_ALL}")
        return processed_data 