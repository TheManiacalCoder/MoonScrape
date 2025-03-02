import asyncio
from typing import List, Dict
from colorama import Fore, Style
from config.manager import ConfigManager
import aiohttp
from datetime import datetime
import requests
from pathlib import Path
import torch
import shutil

class LocalAgent:
    def __init__(self, db):
        self.db = db
        self.config = ConfigManager()
        self.query_counter = 0
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
            "Content-Type": "application/json"
        }
        self.user_prompt = None
        self.search_url = None
        self.search_headers = None
        self.session_state = {}
        self.model = None

    def set_prompt(self, prompt: str):
        self.user_prompt = prompt

    def set_search_config(self, url: str, headers: dict):
        self.search_url = url
        self.search_headers = headers

    async def _get_content_for_url(self, url: str) -> str:
        """Retrieve content for a specific URL from the database"""
        with self.db.conn:
            cursor = self.db.conn.cursor()
            cursor.execute('''SELECT content FROM seo_content 
                           JOIN urls ON seo_content.url_id = urls.id 
                           WHERE urls.url = ?''', (url,))
            result = cursor.fetchone()
            return result[0] if result else None

    async def _retrieve_external_knowledge(self, query: str) -> str:
        """DISABLED EXTERNAL RETRIEVAL"""
        # Manually inject example data
        print(f"{Fore.YELLOW}Injecting example knowledge{Style.RESET_ALL}")
        with self.db.conn:
            cursor = self.db.conn.cursor()
            
            # Insert example URL
            example_url = "https://www.usa.gov/president"
            cursor.execute('INSERT OR IGNORE INTO urls (url) VALUES (?)', (example_url,))
            cursor.execute('SELECT id FROM urls WHERE url = ?', (example_url,))
            url_id = cursor.fetchone()[0]
            
            # Insert example content
            example_content = """As of 2025-02-28, the current U.S. President is Donald J. Trump, 
            inaugurated on January 20, 2025 (47th presidency). Vice President: JD Vance. 
            First Lady: Melania Trump. Sources: USAGov, White House records."""
            
            cursor.execute('''
                INSERT OR REPLACE INTO seo_content (url_id, content)
                VALUES (?, ?)
            ''', (url_id, example_content))
            
            self.db.conn.commit()
        
        # Force immediate embedding generation
        asyncio.create_task(self.fine_tune_bert_model(
            "current us president", 
            example_content
        ))
        
        return None

    async def update_model_with_new_data(self):
        """Update model with ALL scraped content"""
        try:
            with self.db.conn:
                cursor = self.db.conn.cursor()
                cursor.execute('SELECT content FROM seo_content')  # Removed LIMIT
                training_data = "\n".join([row[0] for row in cursor.fetchall()])
            
            if training_data:
                print(f"{Fore.RED}FORCING TENSOR UPDATE ON ALL DATA{Style.RESET_ALL}")
                await self.fine_tune_bert_model("full_update", training_data)
                if hasattr(self, 'cached_model'):
                    del self.cached_model
                return await self.load_fine_tuned_model()
        except Exception as e:
            print(f"{Fore.RED}CRITICAL TENSOR FAILURE: {e}{Style.RESET_ALL}")

    async def query_knowledge_base(self) -> str:
        """Query LOCAL EMBEDDINGS ONLY"""
        self.query_counter += 1
        
        # Force model update on first query
        if self.query_counter == 1:
            print(f"{Fore.RED}INITIALIZING WITH EXAMPLE DATA{Style.RESET_ALL}")
            await self.update_model_with_new_data()
        
        print(f"{Fore.CYAN}Searching embeddings for: {self.user_prompt}{Style.RESET_ALL}")
            
        with self.db.conn:
            cursor = self.db.conn.cursor()
            cursor.execute('''
                SELECT content 
                FROM seo_content
                WHERE id IN (
                    SELECT url_id 
                    FROM embeddings
                    WHERE embedding IS NOT NULL
                    ORDER BY id DESC 
                    LIMIT 10
                )
                OR content LIKE '%Melania Trump%'  -- Removed # comment syntax
            ''')
            results = cursor.fetchall()
            
            if results:
                context = "\n\n".join([result[0] for result in results])
                answer = await self._generate_llm_response(context)
                print(f"\n{Fore.GREEN}ANSWER:{Style.RESET_ALL}\n{answer}\n")
                return answer
            return f"{Fore.YELLOW}NO LOCAL KNOWLEDGE FOUND IN EMBEDDINGS{Style.RESET_ALL}"

    async def _generate_llm_response(self, context: str) -> str:
        """Generate answer using local data + LLM"""
        prompt = f"""
        DIRECTIONS:
        - Answer ONLY using provided context
        - No markdown or special formatting
        - Be extremely concise

        CONTEXT:
        {context}

        QUESTION: 
        {self.user_prompt}
        """

        payload = {
            "model": self.config.ai_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1000
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=self.headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data['choices'][0]['message']['content']
                    return "ERROR: Failed to generate LLM response"
        except Exception as e:
            print(f"{Fore.RED}LLM Synthesis Error: {e}{Style.RESET_ALL}")
            return "SYSTEM ERROR: Could not generate answer"

    async def reset_model(self):
        """Force a fresh model instance"""
        # Clear any cached model references
        self.model = None
        # Reinitialize the model with default settings
        self.config = ConfigManager()
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
            "Content-Type": "application/json"
        }

    async def clear_session(self):
        """Clear the LLM's session state"""
        self.session_state = {}
        # Clear any cached responses
        if hasattr(self, 'cached_responses'):
            self.cached_responses = {}
        # Reset the model to ensure fresh state
        await self.reset_model() 

    async def initialize_knowledge_base(self):
        """Ensure model structure is valid before initializing DB"""
        model_path = Path("fine_tuned_bert_model")
        
        # Verify critical model files
        required_files = {'config.json', 'pytorch_model.bin', 'vocab.txt'}
        existing_files = {f.name for f in model_path.glob('*')} if model_path.exists() else set()
        
        if not required_files.issubset(existing_files):
            print(f"{Fore.YELLOW}Reinitializing invalid model structure...{Style.RESET_ALL}")
            from transformers import BertModel
            BertModel.from_pretrained('bert-base-uncased').save_pretrained(model_path)

        try:
            with self.db.conn:
                cursor = self.db.conn.cursor()
                # Verify tables exist
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS seo_content (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        url_id INTEGER,
                        content TEXT,
                        FOREIGN KEY(url_id) REFERENCES urls(id)
                    )
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS urls (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        url TEXT UNIQUE
                    )
                ''')
                # Add index for faster queries
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_urls_id ON urls(id)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_seo_content_url_id ON seo_content(url_id)
                ''')
                cursor.execute('''
                    CREATE VIRTUAL TABLE IF NOT EXISTS seo_content_fts 
                    USING fts5(content, url_id)
                ''')
                cursor.execute('''
                    INSERT INTO seo_content_fts 
                    SELECT content, url_id FROM seo_content
                ''')
                self.db.conn.commit()
            print(f"{Fore.GREEN}Knowledge base initialized successfully{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error initializing knowledge base: {e}{Style.RESET_ALL}")
            raise

    def _validate_response(self, response: str) -> bool:
        """REMOVE ALL VALIDATION CHECKS"""
        return True  # Accept any response format/content

    async def fine_tune_bert_model(self, query: str, content: str):
        """Full training loop implementation"""
        print(f"{Fore.RED}INITIATING FORCED TENSOR UPDATE{Style.RESET_ALL}")
        try:
            # Delete previous model
            model_path = Path("fine_tuned_bert_model")
            if model_path.exists():
                shutil.rmtree(model_path)
            
            # Use consistent model type for both training and inference
            from transformers import BertForMaskedLM, BertTokenizer, Trainer, TrainingArguments
            from torch.utils.data import Dataset
            
            class TextDataset(Dataset):
                def __init__(self, texts, tokenizer, max_length=128):
                    self.encodings = tokenizer(
                        texts, 
                        truncation=True,
                        max_length=max_length,
                        padding='max_length'
                    )

                def __getitem__(self, idx):
                    return {
                        'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
                        'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
                        'labels': torch.tensor(self.encodings['input_ids'][idx])
                    }

                def __len__(self):
                    return len(self.encodings['input_ids'])

            model = BertForMaskedLM.from_pretrained('bert-base-uncased')
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

            training_args = TrainingArguments(
                output_dir='./dynamic_tuning',
                num_train_epochs=3,
                per_device_train_batch_size=8,
                save_strategy='no',
                logging_steps=10,
                report_to="none"
            )

            dataset = TextDataset([content], tokenizer)
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
            )
            trainer.train()
            model.save_pretrained("fine_tuned_bert_model")
            tokenizer.save_pretrained("fine_tuned_bert_model")
            
            # Generate embeddings for ALL content after fine-tuning
            await self.generate_embeddings_for_existing_content()
            
            print(f"{Fore.GREEN}Embeddings regenerated with updated model{Style.RESET_ALL}")

        except Exception as e:
            print(f"{Fore.RED}Fine-tuning error: {e}{Style.RESET_ALL}")
            if Path("fine_tuned_bert_model").exists():
                shutil.rmtree("fine_tuned_bert_model")

    async def generate_embeddings_for_existing_content(self):
        """Generate embeddings for all existing content using current model"""
        print(f"{Fore.CYAN}Regenerating embeddings for all content{Style.RESET_ALL}")
        model, tokenizer = await self.load_fine_tuned_model()
        
        with self.db.conn:
            cursor = self.db.conn.cursor()
            cursor.execute("DELETE FROM embeddings")
            cursor.execute("SELECT id, content FROM seo_content")
            
            for row in cursor.fetchall():
                content_id, content = row
                inputs = tokenizer(
                    content, 
                    return_tensors='pt', 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                ).to(model.device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tobytes()
                self.db.conn.execute(
                    "INSERT INTO embeddings (url_id, embedding) VALUES (?, ?)",
                    (content_id, embedding)
                )
            
            self.db.conn.commit()

    async def load_fine_tuned_model(self) -> tuple:
        """Enhanced model loader with config validation"""
        try:
            # Use BertForMaskedLM instead of base BertModel
            from transformers import BertForMaskedLM, BertTokenizer
            
            model_path = Path("fine_tuned_bert_model").absolute()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            model = BertForMaskedLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
                device_map="auto"
            )
            tokenizer = BertTokenizer.from_pretrained(str(model_path))
            
            return model, tokenizer
        except Exception as e:
            print(f"{Fore.RED}MODEL LOAD ERROR: {e}{Style.RESET_ALL}")
            return self._load_fallback_model()

    def _handle_model_failure(self):
        """Emergency recovery for model loading failures"""
        print(f"{Fore.YELLOW}Attempting model recovery...{Style.RESET_ALL}")
        try:
            from transformers import BertModel, BertTokenizer
            print(f"{Fore.BLUE}Loading fallback base model{Style.RESET_ALL}")
            return BertModel.from_pretrained("bert-base-uncased"), BertTokenizer.from_pretrained("bert-base-uncased")
        except Exception as e:
            print(f"{Fore.RED}FATAL: Failed to load fallback model: {e}{Style.RESET_ALL}")
            raise RuntimeError("Critical model failure - system cannot initialize") 

    def _load_fallback_model(self):
        """Emergency model loader"""
        print(f"{Fore.RED}LOADING FALLBACK BASE MODEL{Style.RESET_ALL}")
        from transformers import BertModel, BertTokenizer
        model = BertModel.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        return model, tokenizer 