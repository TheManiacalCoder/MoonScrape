import requests
import base64
import json
from storage.database_manager import DatabaseManager
from config.manager import ConfigManager
from bs4 import BeautifulSoup
import sqlite3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import random
import re
from colorama import init, Fore, Style
from agents.analyzer import OpenRouterAnalyzer
import asyncio
import chardet
from agents.content_processor import ContentProcessor
from agents.intent_agent import IntentAgent
import os
import aiohttp
from datetime import datetime
from typing import List
from pathlib import Path
from agents.local_agent import LocalAgent
from agents.scheduled_search import ScheduledSearch
from shared import db, config
import subprocess
import sys
from packaging import version
import shutil
from transformers import BertModel, BertTokenizer

# Add these lines to make url and headers available for import
__all__ = ['main', 'url', 'headers']

init()  # Initialize Colorama to convert ANSI sequences to Windows equivalents

def show_title_screen():
    print("\n" + "=" * 50)
    print(" " * 20 + "MoonScrape")
    print("=" * 50)
    print(" " * 15 + "A product of OdinWeb3Labs")
    print("=" * 50)
    print("\nInitializing web scraper...")
    print("=" * 50 + "\n")

# Get credentials from config
cred = base64.b64encode(
    f"{config.email}:{config.api_key}".encode()
).decode()

# Set the API endpoint
url = "https://api.dataforseo.com/v3/serp/google/organic/live/advanced"

# Set headers
headers = {
   "Authorization": f"Basic {cred}",
   "Content-Type": "application/json"
}

# Now import ScheduledSearch

# Move the keyword input and main process into a loop
async def main_loop():
    scheduled_search = ScheduledSearch()
    
    while True:
        print("\nMain Menu:")
        print("1. Perform new web search")
        print("2. Query local knowledge base")
        print("3. Schedule search")
        print("4. Bulk search from keywords.txt")
        print("5. Reset database (WARNING: Clears ALL data)")
        print("6. Exit")
        
        choice = input("Enter menu selection (1-6): ").strip()
        
        if choice == '1':
            global keyword, payload, response
            keyword = input("\nEnter search keyword: ").strip()
            payload = [
               {
                   "language_code": "en",
                   "location_code": 2840,
                   "keyword": keyword
               }
            ]
            response = requests.post(url, headers=headers, json=payload)
            await main(keyword)
            
        elif choice == '2':
            question = input("\nEnter your question: ").strip()
            await query_local_knowledge(question)
            
        elif choice == '3':
            await scheduled_search.schedule_search_menu()
            
        elif choice == '4':
            await scheduled_search.process_bulk_search()
            
        elif choice == '5':
            confirm = input(f"{Fore.RED}WARNING: This will reset ALL database content. Continue? (y/n): {Style.RESET_ALL}").strip().lower()
            if confirm == 'y':
                await reset_database()
            else:
                print(f"{Fore.YELLOW}Database reset cancelled{Style.RESET_ALL}")
            
        elif choice == '6' or choice.lower() == 'exit':
            print("Exiting MoonScrape. Goodbye!")
            break
            
        else:
            print(f"{Fore.RED}Invalid choice. Please enter 1-6.{Style.RESET_ALL}")
        
        print("\n" + "=" * 50)
        print("Ready for new operation")
        print("=" * 50)

# Add list of domains to exclude
BLACKLISTED_DOMAINS = {
    'reddit.com',
    'youtube.com',
    'vimeo.com',
    'tiktok.com',
    'twitter.com',
    'facebook.com',
    'instagram.com',
    'quora.com',
    'pinterest.com'
}

def is_valid_url(url):
    if not url:
        return False
    return not any(domain in url.lower() for domain in BLACKLISTED_DOMAINS)

# Configure stealth headers
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0'
]

def create_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def scrape_seo_content(url):
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        # Detect encoding
        if 'charset=' in response.headers.get('content-type', ''):
            encoding = response.headers['content-type'].split('charset=')[-1]
        else:
            detected = chardet.detect(response.content)
            encoding = detected['encoding'] if detected['confidence'] > 0.9 else 'utf-8'
            
        try:
            content = response.content.decode(encoding, errors='strict')
        except UnicodeDecodeError:
            content = response.content.decode(encoding, errors='replace')
            
        # Check for error pages
        error_phrases = [
            'technical difficulties',
            'please try again',
            'forbidden',
            'access denied',
            'unavailable',
            'error occurred',
            'we\'re sorry',
            'temporarily unavailable'
        ]
        
        if any(phrase.lower() in content.lower() for phrase in error_phrases):
            return None
            
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove unwanted elements
        for tag in ['script', 'style', 'nav', 'footer', 'header', 'aside', 
                   'form', 'iframe', 'button', 'noscript', 'meta', 'link']:
            for element in soup(tag):
                element.decompose()

        # Find main content
        main_content = []
        for tag in ['article', 'main', 'body']:
            element = soup.find(tag)
            if element:
                main_content.append(element.get_text(' ', strip=True))
                break

        if not main_content:
            return None

        # Clean and format the content
        clean_content = '\n\n'.join(main_content)
        clean_content = re.sub(r'\n{3,}', '\n\n', clean_content)
        return clean_content.strip()
        
    except Exception as e:
        print(f"{Fore.RED}Error scraping {url}: {str(e)}{Style.RESET_ALL}")
        return None

def process_results(items):
    return [item['url'] for item in items if 'url' in item and is_valid_url(item['url'])]

async def run_analysis(collected_urls):
    processor = ContentProcessor(db)
    processed_data = await processor.process_urls(collected_urls)
    
    if processed_data:
        analyzer = OpenRouterAnalyzer(db)
        report = await analyzer.analyze_urls(list(processed_data.keys()))
        if report:
            await analyzer.save_report(report)
    else:
        print(f"{Fore.RED}No valid content to analyze{Style.RESET_ALL}")

def show_progress(step, total_steps, message):
    progress = (step / total_steps) * 100
    print(f"{Fore.CYAN}[{progress:.0f}%] {message}{Style.RESET_ALL}")

async def scrape_urls_concurrently(urls: List[str], db) -> List[str]:
    async def process_single_url(url):
        try:
            if not url:
                return None
                
            # Check if URL already exists in database
            with db.conn:
                cursor = db.conn.cursor()
                cursor.execute('SELECT id FROM urls WHERE url = ?', (url,))
                existing = cursor.fetchone()
                
                if existing:
                    url_id = existing[0]
                else:
                    # Insert new URL and get its ID
                    cursor.execute('INSERT INTO urls (url) VALUES (?)', (url,))
                    url_id = cursor.lastrowid
                
            # Scrape content
            content = scrape_seo_content(url)
            if content:
                # Save content in batch
                with db.conn:
                    db.conn.execute('INSERT INTO seo_content (url_id, content) VALUES (?, ?)', 
                                  (url_id, content))
                return url
            return None
        except Exception as e:
            print(f"{Fore.RED}Error processing {url}: {str(e)}{Style.RESET_ALL}")
            return None

    # Process all URLs concurrently with improved error handling
    print(f"\nStarting concurrent URL collection for {len(urls)} URLs...")
    
    # Use a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent requests
    
    async def limited_task(url):
        async with semaphore:
            return await process_single_url(url)
    
    # Batch process URLs
    results = await asyncio.gather(*(limited_task(url) for url in urls))
    
    # Filter out None values and return successful URLs
    successful_urls = [url for url in results if url is not None]
    print(f"Successfully collected {len(successful_urls)} URLs")
    return successful_urls

async def main(keyword: str = None):
    try:
        # If keyword is not provided, use the global keyword
        if keyword is None:
            keyword = globals().get('keyword')
            
        # Set up the payload
        payload = [{
            "language_code": "en",
            "location_code": 2840,
            "keyword": keyword
        }]
        
        # Make the API request
        response = requests.post(url, headers=headers, json=payload)
        
        # Reset benchmark file at the start
        benchmark_file = Path("benchmark_results.txt")
        if benchmark_file.exists():
            benchmark_file.unlink()
            print(f"{Fore.YELLOW}Deleted existing benchmark file{Style.RESET_ALL}")
        
        # Initialize fresh benchmark data
        benchmark_data = {
            'total_requests': 0,
            'total_tokens': 0,
            'search_keyword': keyword
        }
        
        show_progress(0, 4, "Starting search...")
        
        # Start timing only when the actual search begins
        search_start = datetime.now()
        
        # Make the initial search request
        response_data = response.json()
        if not response_data or 'tasks' not in response_data:
            raise ValueError("Invalid API response format")
        
        show_progress(1, 4, "Processing results...")
        
        results = response_data['tasks'][0]['result'][0]
        if not results or 'items' not in results:
            raise ValueError("No search results found")
        
        valid_urls = process_results(results['items'])
        
        show_progress(2, 4, "Checking for additional pages...")
        
        page = 1
        while len(valid_urls) < 10 and page < results.get('metrics', {}).get('pagination', {}).get('total', 1):
            page += 1
            next_payload = [{
                "language_code": "en",
                "location_code": 2840,
                "keyword": keyword,
                "page": page
            }]
            next_response = requests.post(url, headers=headers, json=next_payload)
            next_data = next_response.json()
            if next_data and 'tasks' in next_data:
                valid_urls += process_results(next_data['tasks'][0]['result'][0]['items'])
        
        show_progress(3, 4, "Processing URLs concurrently...")
        
        # Process all URLs concurrently
        collected_urls = await scrape_urls_concurrently(valid_urls[:10], db)
        
        # REMOVED FORCED UPDATE HERE - NOW HANDLED BY QUERY COUNTER
        print(f"{Fore.CYAN}New data ready for tensor integration{Style.RESET_ALL}")
        
        show_progress(4, 4, "Search complete!")
        
        print(f"\n{Fore.CYAN}Starting intent-based filtering...{Style.RESET_ALL}")
        intent_agent = IntentAgent(db)
        intent_agent.set_prompt(keyword)
        
        # Pass benchmark data to process_urls
        final_summary = await intent_agent.process_urls(collected_urls, benchmark_data)
        
        if final_summary:
            print(f"\n{Fore.MAGENTA}Preparing final summary for analysis...{Style.RESET_ALL}")
            print(f"{Fore.WHITE}Summary length: {len(final_summary)} characters{Style.RESET_ALL}")
            
            print("\nStarting comprehensive SEO analysis...")
            analyzer = OpenRouterAnalyzer(db)
            analyzer.set_prompt(keyword)
            
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            summary_data = {"final_summary": final_summary}
            report = await analyzer.analyze_urls(summary_data)
            
            if report:
                await analyzer.save_report(report)
                print(f"{Fore.GREEN}Comprehensive SEO analysis complete!{Style.RESET_ALL}")
                print(f"\n{Fore.CYAN}Final Report:{Style.RESET_ALL}")
                print(report)
                
                # Calculate total duration
                total_duration = (datetime.now() - search_start).total_seconds()
                
                # Write single benchmark report
                with open("benchmark_results.txt", "w", encoding="utf-8") as f:
                    f.write(f"""
                    Benchmark Report for Search: {keyword}
                    =============================
                    
                    Total Duration: {total_duration:.2f} seconds
                    
                    Resource Usage:
                    - Total Requests: {benchmark_data['total_requests']}
                    - Total Tokens Processed: {benchmark_data['total_tokens']}
                    - URLs Processed: {len(collected_urls)}
                    - Summary Length: {len(final_summary)} characters
                    """)
                print(f"{Fore.GREEN}Benchmark report saved{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Failed to generate analysis report{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error in main workflow: {e}{Style.RESET_ALL}")

async def query_local_knowledge(question: str):
    print(f"\n{Fore.CYAN}Querying knowledge base (KAG+RAG)...{Style.RESET_ALL}")
    
    # Initialize the LocalAgent
    local_agent = LocalAgent(db)
    await local_agent.initialize_knowledge_base()
    
    # Set the search prompt
    local_agent.set_prompt(question)
    
    # Process the query through EMBEDDINGS ONLY
    result = await local_agent.query_knowledge_base()
    print(f"\n{Fore.GREEN}KNOWLEDGE BASE RESPONSE:{Style.RESET_ALL}\n{result}")
    return result

async def reset_database():
    """Completely reset the database and model"""
    try:
        print(f"{Fore.YELLOW}Resetting to blank state...{Style.RESET_ALL}")
        
        # Reset all tables
        with db.conn:
            # Drop all tables if they exist
            db.conn.execute("DROP TABLE IF EXISTS embeddings")
            db.conn.execute("DROP TABLE IF EXISTS seo_content")
            db.conn.execute("DROP TABLE IF EXISTS urls")
            db.conn.execute("DROP TABLE IF EXISTS llm_model")
            db.conn.execute("DROP TABLE IF EXISTS llm_only")
            db.conn.execute("DROP TABLE IF EXISTS disable_embeddings")
            print(f"{Fore.GREEN}All database tables have been dropped{Style.RESET_ALL}")
            
            # Recreate empty tables with default structure
            db._init_db()
            print(f"{Fore.GREEN}Database reinitialized to blank state{Style.RESET_ALL}")
            
        # Delete all cached files in the analysis directory
        analysis_dir = Path("analysis")
        if analysis_dir.exists():
            # Delete all files in the analysis directory
            for file_path in analysis_dir.glob("*"):
                try:
                    if file_path.is_file():
                        file_path.unlink()
                        print(f"{Fore.GREEN}Deleted cached file: {file_path}{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}Error deleting {file_path}: {e}{Style.RESET_ALL}")
            
            print(f"{Fore.GREEN}All cached files in analysis directory deleted{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}Analysis directory not found{Style.RESET_ALL}")
            
        # Delete benchmark file if it exists
        benchmark_file = Path("benchmark_results.txt")
        if benchmark_file.exists():
            benchmark_file.unlink()
            print(f"{Fore.GREEN}Deleted benchmark file{Style.RESET_ALL}")
            
        # Force LLM-only mode by setting a flag in the database
        with db.conn:
            # Recreate tables to ensure clean state
            db.conn.execute("CREATE TABLE IF NOT EXISTS llm_only (flag INTEGER)")
            db.conn.execute("INSERT OR REPLACE INTO llm_only (flag) VALUES (1)")
            # Disable embedding storage
            db.conn.execute("CREATE TABLE IF NOT EXISTS disable_embeddings (flag INTEGER)")
            db.conn.execute("INSERT OR REPLACE INTO disable_embeddings (flag) VALUES (1)")
            # Disable external knowledge retrieval
            db.conn.execute("CREATE TABLE IF NOT EXISTS disable_external_knowledge (flag INTEGER)")
            db.conn.execute("INSERT OR REPLACE INTO disable_external_knowledge (flag) VALUES (1)")
            # Reset LLM model to default
            db.conn.execute("CREATE TABLE IF NOT EXISTS llm_model (model TEXT)")
            db.conn.execute("INSERT OR REPLACE INTO llm_model (model) VALUES (?)", 
                          (config.default_ai_model,))
            print(f"{Fore.GREEN}Forced LLM-only mode, disabled embedding storage, and reset LLM to default{Style.RESET_ALL}")
            
        # Clear the LLM's session state and force a fresh model instance
        local_agent = LocalAgent(db)
        await local_agent.clear_session()
        print(f"{Fore.GREEN}LLM session state cleared{Style.RESET_ALL}")
        
        # Force a fresh model instance
        await local_agent.reset_model()
        print(f"{Fore.GREEN}LLM model reset to original, untrained state{Style.RESET_ALL}")
            
        print(f"{Fore.GREEN}System reset to blank state complete!{Style.RESET_ALL}")
            
        # Add nuclear file system cleanup
        for temp_dir in [Path("C:/tmp"), Path("~/AppData/Local/Temp").expanduser()]:
            for temp_file in temp_dir.glob("moonscrape_*"):
                try:
                    if temp_file.is_file():
                        temp_file.unlink()
                        print(f"Purged temp file: {temp_file}")
                except Exception as e:
                    print(f"Error purging {temp_file}: {e}")

        # Replace problematic model re-download with clean install
        subprocess.run([
            "pip", "install", 
            "-r", "requirements.txt",
            "--force-reinstall",
            "--no-deps"
        ], check=True)
        
        print(f"{Fore.GREEN}Model cache forcibly refreshed{Style.RESET_ALL}")
        
        # Add model reinitialization
        model_path = Path("fine_tuned_bert_model")
        if model_path.exists():
            shutil.rmtree(model_path)
            
        BertModel.from_pretrained('bert-base-uncased').save_pretrained(model_path)
        BertTokenizer.from_pretrained('bert-base-uncased').save_pretrained(model_path)
        
        print(f"{Fore.GREEN}Model structure reinitialized{Style.RESET_ALL}")
            
    except Exception as e:
        print(f"{Fore.RED}Error resetting to blank state: {e}{Style.RESET_ALL}")

def check_versions():
    required_versions = {
        'transformers': '4.35.0',
        'torch': '2.0.1',
        'huggingface_hub': '0.16.0'
    }
    
    for package, req_version in required_versions.items():
        try:
            mod = __import__(package)
            if version.parse(mod.__version__) < version.parse(req_version):
                print(f"{Fore.RED}Error: {package} version {mod.__version__} is installed, but version {req_version} or higher is required{Style.RESET_ALL}")
                return False
        except ImportError:
            print(f"{Fore.RED}Error: {package} is not installed{Style.RESET_ALL}")
            return False
    return True

if not check_versions():
    sys.exit(1)

def initialize_model_structure():
    """Ensure valid model structure exists before any operations"""
    from transformers import BertModel, BertTokenizer
    model_path = Path("fine_tuned_bert_model")
    
    required_files = {'config.json', 'pytorch_model.bin', 'vocab.txt'}
    existing_files = {f.name for f in model_path.glob('*')} if model_path.exists() else set()
    
    if not required_files.issubset(existing_files):
        print(f"{Fore.YELLOW}Initializing fresh BERT model structure...{Style.RESET_ALL}")
        BertModel.from_pretrained('bert-base-uncased').save_pretrained(model_path)
        BertTokenizer.from_pretrained('bert-base-uncased').save_pretrained(model_path)
        print(f"{Fore.GREEN}Model structure validated{Style.RESET_ALL}")

# Modify the main execution block at the bottom of the file
if __name__ == "__main__":
    # Show title screen only once at startup
    show_title_screen()
    
    # Add model initialization before any operations
    initialize_model_structure()
    
    # Start the main loop
    asyncio.run(main_loop())
