import aiohttp
import asyncio
from typing import List, Dict
from colorama import Fore, Style
from config.manager import ConfigManager
from datetime import datetime
from nltk.tokenize import word_tokenize

try:
    import numpy as np
except ImportError as e:
    print(f"{Fore.RED}Error importing NumPy: {e}{Style.RESET_ALL}")
    print("Attempting to install NumPy...")
    import subprocess
    subprocess.run(["pip", "install", "numpy"])
    import numpy as np

from pathlib import Path
import torch

try:
    from transformers import BertTokenizer, BertModel, AutoModel
except ImportError as e:
    print(f"{Fore.RED}Error importing transformers or peft: {e}{Style.RESET_ALL}")
    print("Attempting to install correct versions...")
    import subprocess
    subprocess.run(["pip", "install", "transformers==4.35.0", "torch==2.0.1", "huggingface_hub==0.16.0", "peft"])
    from transformers import BertTokenizer, BertModel, AutoModel

try:
    from torch import Tensor
except ImportError as e:
    print(f"PyTorch import error: {e}")
    print("Attempting to install correct version...")
    import subprocess
    subprocess.run([
        "pip", "install", 
        "torch==2.0.1", 
        "torchvision==0.15.2", 
        "torchaudio==2.0.2", 
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ])
    from torch import Tensor

from peft import LoraConfig, get_peft_model

class IntentAgent:
    def __init__(self, db):
        self.db = db
        self.config = ConfigManager()
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
            "Content-Type": "application/json"
        }
        self.user_prompt = None
        
        # Initialize model check attributes FIRST
        self.last_model_check = 0
        self.model_refresh_interval = 60  # Seconds

        # Load the fine-tuned BERT model
        self.bert_model, self.tokenizer = self._load_fine_tuned_model()
        if not self.bert_model:
            print(f"{Fore.RED}Failed to load fine-tuned BERT model, using default{Style.RESET_ALL}")
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
        
        self.bert_model.eval()  # Set to evaluation mode

    def _load_fine_tuned_model(self):
        """Modified with freshness check"""
        try:
            model_path = Path("fine_tuned_bert_model")
            current_mtime = model_path.stat().st_mtime
            
            if current_mtime > self.last_model_check:
                print(f"{Fore.CYAN}Detected model updates, reloading...{Style.RESET_ALL}")
                self.last_model_check = current_mtime
                return self._force_model_reload()
                
            # Return cached model if recent
            return self.bert_model, self.tokenizer
            
        except Exception as e:
            print(f"{Fore.RED}Model reload error: {e}{Style.RESET_ALL}")
            return None, None

    def _force_model_reload(self):
        """Force fresh model load"""
        from transformers import BertModel, BertTokenizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = BertModel.from_pretrained(
            "fine_tuned_bert_model",
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
            device_map="auto"
        ).to(device)
        
        tokenizer = BertTokenizer.from_pretrained("fine_tuned_bert_model")
        return model, tokenizer

    def set_prompt(self, prompt: str):
        self.user_prompt = prompt

    def _get_sentence_vector(self, sentence: str):
        # Ensure we're using fine-tuned model
        if not self.bert_model:
            raise RuntimeError("Fine-tuned model not loaded")
        
        # Get model device
        device = self.bert_model.device
        
        inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to model's device
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        
        # Move result to CPU before converting to numpy
        return torch.mean(outputs.last_hidden_state, dim=1).squeeze().cpu().numpy()

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        vec1 = self._get_sentence_vector(text1)
        vec2 = self._get_sentence_vector(text2)
        
        if vec1 is None or vec2 is None:
            return 0.0
            
        # Ensure we're using numpy arrays on CPU    
        cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return max(0.0, min(1.0, cosine_sim))

    async def filter_relevant_content(self, content: str) -> str:
        if not self.user_prompt:
            return content
            
        print(f"\n{Fore.CYAN}Analyzing content for intent: {self.user_prompt}{Style.RESET_ALL}")
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_year = datetime.now().year
        
        intent_analysis = f"""
        Analyze this content for relevance to: {self.user_prompt}
        
        Requirements:
        - Present specific facts and data
        - Use only the most relevant URL
        - Maintain factual accuracy
        - Focus on evidence-based presentation
        """
        
        prompt = f"""
        {intent_analysis}
        
        Now analyze this content:
        {content}
        
        Apply temporal filtering based on the query analysis.
        Exclude outdated content unless explicitly requested.
        Ensure all information is current as of {current_date}.
        """
        
        payload = {
            "model": self.config.ai_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 5000
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=self.headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                else:
                    error = await response.text()
                    print(f"{Fore.RED}Error filtering content: {error}{Style.RESET_ALL}")
                    return None

    async def _spell_check_query(self, query: str) -> str:
        prompt = f"""
        Review this search query for spelling/grammar issues:
        "{query}"
        
        Return ONLY: 
        - The corrected query if errors found
        - The original query if no errors
        """
        
        payload = {
            "model": self.config.ai_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 100
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=self.headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content'].strip('"')
                return query

    async def process_urls(self, urls: List[str], benchmark_data: dict = None) -> Dict[str, str]:
        processed_data = {}
        total_urls = len(urls)
        current_date = datetime.now().strftime("%Y-%m-%d")

        async def fetch_url_content(url):
            try:
                with self.db.conn:
                    cursor = self.db.conn.cursor()
                    cursor.execute('''SELECT content FROM seo_content 
                                   JOIN urls ON seo_content.url_id = urls.id 
                                   WHERE urls.url = ?''', (url,))
                    result = cursor.fetchone()
                    
                    if result and result[0]:
                        content = result[0]
                        
                        # Modify content processing
                        content = f"""
                        {content}
                        
                        TEMPORAL VALIDATION:
                        - New claims must include verifiable sources
                        - Official records take precedence
                        - Contradictions require multi-source confirmation
                        """
                        
                        # Add timestamp context if relevant
                        if "current" in self.user_prompt.lower() or "today" in self.user_prompt.lower():
                            content = f"""
                            Current Date: {current_date}
                            
                            {content}
                            
                            Important:
                            - Verify all information is current as of {current_date}
                            - Reject outdated information
                            - Prioritize recent sources
                            """
                        return url, content
            except Exception as e:
                print(f"Processing failed for {url}: {str(e)}")
            return url, None

        # Process all URLs concurrently
        print(f"\nStarting concurrent URL content collection for {total_urls} URLs...")
        tasks = [fetch_url_content(url) for url in urls]
        results = await asyncio.gather(*tasks)
        
        # Collect successful results
        for url, content in results:
            if content:
                processed_data[url] = content

        print(f"\nURL content collection complete! Processed {total_urls} URLs, found relevant content in {len(processed_data)} URLs")
        
        if processed_data:
            print(f"\nStarting final summary with intent-focused analysis...")
            combined_content = "\n\n".join(processed_data.values())
            
            # Improved prompt for better formatting
            analysis_prompt = f"""
            Analyze this content and present factual information relevant to: {self.user_prompt}
            
            Content Sources:
            {combined_content}
            
            Format the response as follows:
            
            ### Key Findings
            - [Main point 1]
            - [Main point 2]
            - [Main point 3]
            
            ### Supporting Evidence
            - [Quote or fact 1 with source]
            - [Quote or fact 2 with source]
            
            ### Conclusion
            [Summary of findings]
            
            Requirements:
            - Use clear section headers
            - Include bullet points for key information
            - Maintain proper formatting
            - Be concise and factual
            """
            
            # Initial quality check based on intent matching
            semantic_score = self._calculate_semantic_similarity(combined_content, self.user_prompt)
            
            # If we have a strong match, return immediately
            if semantic_score >= 0.9:  # High confidence match
                print(f"Strong intent match found (score: {semantic_score:.2f}), returning initial content")
                return combined_content
            
            # Single focused analysis using scraped content
            print(f"\nPerforming focused analysis (initial intent score: {semantic_score:.2f})")
            
            payload = {
                "model": self.config.ai_model,
                "messages": [{"role": "user", "content": analysis_prompt}],
                "temperature": 0.3,
                "max_tokens": 2000
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=self.headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        analysis = data['choices'][0]['message']['content']
                        
                        # Update benchmark data
                        if benchmark_data:
                            benchmark_data['total_requests'] += 1
                            tokens = len(analysis.split())
                            benchmark_data['total_tokens'] += tokens
                        
                        # Verify final intent match
                        final_semantic_score = self._calculate_semantic_similarity(analysis, self.user_prompt)
                        print(f"Final intent match score: {final_semantic_score:.2f}")
                        
                        if final_semantic_score >= 0.8:  # Good enough match
                            return analysis
                        else:
                            print("Insufficient intent match, returning raw content")
                            return combined_content
                    else:
                        error = await response.text()
                        print(f"{Fore.RED}Analysis failed: {error}{Style.RESET_ALL}")
                        return combined_content
        else:
            print("No relevant content found for summary")
            return None

    def _evaluate_analysis_quality(self, analysis: str, epoch: int) -> float:
        score = 0.0
        
        if analysis:
            score += 0.2
            
        current_year = datetime.now().year
        if str(current_year) in analysis:
            score += 0.1 + (0.02 * epoch)
            
        if "as of" in analysis.lower() or "current" in analysis.lower():
            score += 0.1
            
        structure_components = [
            "### Executive Summary",
            "### Key Findings",
            "### Detailed Analysis",
            "### Recommendations",
            "### Sources"
        ]
        for i, component in enumerate(structure_components):
            if component in analysis:
                score += 0.1 + (0.02 * epoch)
                
        if epoch == 1 and "facts" in analysis.lower():
            score += 0.1
        if epoch == 2 and "evidence" in analysis.lower():
            score += 0.1
        if epoch == 3 and "patterns" in analysis.lower():
            score += 0.1
        if epoch == 4 and "insights" in analysis.lower():
            score += 0.1
        if epoch == 5 and "recommendations" in analysis.lower():
            score += 0.1
            
        score += min(len(analysis) / (2000 + (epoch * 200)), 0.2)
        
        if self.bert_model:
            semantic_score = self._calculate_semantic_similarity(analysis, self.user_prompt)
            score += semantic_score * (0.1 + (0.02 * epoch))
        
        if "clearly" in analysis.lower() or "concisely" in analysis.lower():
            score += 0.05 * epoch
            
        depth_indicators = ["detailed", "in-depth", "comprehensive", "thorough"]
        for indicator in depth_indicators:
            if indicator in analysis.lower():
                score += 0.05 * epoch
                
        if "specific" in analysis.lower() or "precise" in analysis.lower():
            score += 0.05 * epoch
            
        evidence_indicators = ["data", "statistics", "research", "study", "source"]
        for indicator in evidence_indicators:
            if indicator in analysis.lower():
                score += 0.05 * epoch
                
        if "actionable" in analysis.lower() or "recommendation" in analysis.lower():
            score += 0.05 * epoch
            
        return min(score, 1.0)