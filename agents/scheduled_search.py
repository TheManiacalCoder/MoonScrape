import asyncio
from datetime import datetime, timedelta
from typing import Optional
import schedule
import time
from pathlib import Path
from shared import db, config
from colorama import Fore, Style
import requests

class ScheduledSearch:
    def __init__(self):
        self.scheduled_jobs = []
        self.running = False

    async def run_scheduled_search(self, keyword: str, search_time: str):
        """Run a search at the specified time"""
        try:
            # Lazy import to avoid circular dependency
            from SERP_Scraper import main, url, headers
            
            # Parse the search time
            search_datetime = datetime.strptime(search_time, "%Y-%m-%d %H:%M")
            now = datetime.now()
            
            if search_datetime < now:
                print(f"{Fore.RED}Error: Scheduled time must be in the future{Style.RESET_ALL}")
                return

            # Calculate delay in seconds
            delay = (search_datetime - now).total_seconds()
            
            print(f"{Fore.GREEN}Scheduled search for '{keyword}' at {search_time}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Waiting for {delay:.1f} seconds...{Style.RESET_ALL}")
            
            await asyncio.sleep(delay)
            
            print(f"\n{Fore.GREEN}Starting scheduled search for '{keyword}'{Style.RESET_ALL}")
            await main(keyword)
            
            # Add timestamp to the final output
            self.timestamp_output(keyword)
            
        except Exception as e:
            print(f"{Fore.RED}Error in scheduled search: {e}{Style.RESET_ALL}")

    def timestamp_output(self, keyword: str):
        """Add timestamp to the final output files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Rename benchmark file
        benchmark_file = Path("benchmark_results.txt")
        if benchmark_file.exists():
            new_name = f"benchmark_{keyword}_{timestamp}.txt"
            benchmark_file.rename(new_name)
            print(f"{Fore.GREEN}Saved benchmark results to {new_name}{Style.RESET_ALL}")
        
        # Rename analysis file
        analysis_file = Path("analysis/aggregated_analysis.txt")
        if analysis_file.exists():
            new_name = f"analysis/analysis_{keyword}_{timestamp}.txt"
            analysis_file.rename(new_name)
            print(f"{Fore.GREEN}Saved analysis results to {new_name}{Style.RESET_ALL}")

    async def schedule_search_menu(self):
        """Menu for scheduling searches"""
        while True:
            print("\n" + "=" * 50)
            print(f"{Fore.CYAN}SCHEDULED SEARCH MENU{Style.RESET_ALL}")
            print("=" * 50)
            print(f"{Fore.YELLOW}1. Schedule new search{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}2. View scheduled searches{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}3. Cancel scheduled search{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}4. Bulk search from keywords.txt{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}5. Return to main menu{Style.RESET_ALL}")
            print("=" * 50)
            
            choice = input(f"{Fore.GREEN}Enter your choice (1-5): {Style.RESET_ALL}").strip()
            
            if choice == '1':
                print("\n" + "=" * 50)
                print(f"{Fore.CYAN}SCHEDULE NEW SEARCH{Style.RESET_ALL}")
                print("=" * 50)
                
                while True:
                    keyword = input(f"{Fore.GREEN}Enter search keyword (or 'back' to return): {Style.RESET_ALL}").strip()
                    if keyword.lower() == 'back':
                        break
                        
                    if not keyword:
                        print(f"{Fore.RED}Error: Keyword cannot be empty{Style.RESET_ALL}")
                        continue
                        
                    while True:
                        search_time = input(f"{Fore.GREEN}Enter search time (YYYY-MM-DD HH:MM) or 'now' for immediate search: {Style.RESET_ALL}").strip()
                        
                        if search_time.lower() == 'now':
                            search_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                            print(f"{Fore.YELLOW}Starting search immediately...{Style.RESET_ALL}")
                            break
                            
                        try:
                            # Validate time format
                            search_datetime = datetime.strptime(search_time, "%Y-%m-%d %H:%M")
                            now = datetime.now()
                            
                            if search_datetime < now:
                                print(f"{Fore.RED}Error: Time must be in the future{Style.RESET_ALL}")
                                continue
                                
                            # Calculate and show time until search
                            time_until = search_datetime - now
                            hours, remainder = divmod(time_until.total_seconds(), 3600)
                            minutes = remainder // 60
                            print(f"{Fore.CYAN}Search will run in {int(hours)} hours and {int(minutes)} minutes{Style.RESET_ALL}")
                            break
                            
                        except ValueError:
                            print(f"{Fore.RED}Invalid time format. Please use YYYY-MM-DD HH:MM{Style.RESET_ALL}")
                            continue
                            
                    # Confirm with user
                    confirm = input(f"{Fore.GREEN}Schedule search for '{keyword}' at {search_time}? (y/n): {Style.RESET_ALL}").strip().lower()
                    if confirm == 'y':
                        try:
                            task = asyncio.create_task(self.run_scheduled_search(keyword, search_time))
                            self.scheduled_jobs.append((keyword, search_time, task))
                            print(f"{Fore.GREEN}Search scheduled successfully!{Style.RESET_ALL}")
                            break
                        except Exception as e:
                            print(f"{Fore.RED}Error scheduling search: {e}{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.YELLOW}Search not scheduled{Style.RESET_ALL}")
                        break
                    
            elif choice == '2':
                print("\n" + "=" * 50)
                print(f"{Fore.CYAN}CURRENTLY SCHEDULED SEARCHES{Style.RESET_ALL}")
                print("=" * 50)
                
                if not self.scheduled_jobs:
                    print(f"{Fore.YELLOW}No scheduled searches{Style.RESET_ALL}")
                else:
                    for i, (keyword, search_time, _) in enumerate(self.scheduled_jobs, 1):
                        search_datetime = datetime.strptime(search_time, "%Y-%m-%d %H:%M")
                        now = datetime.now()
                        time_until = search_datetime - now
                        hours, remainder = divmod(time_until.total_seconds(), 3600)
                        minutes = remainder // 60
                        print(f"{Fore.YELLOW}{i}. {keyword} at {search_time} ({int(hours)}h {int(minutes)}m remaining){Style.RESET_ALL}")
                        
            elif choice == '3':
                print("\n" + "=" * 50)
                print(f"{Fore.CYAN}CANCEL SCHEDULED SEARCH{Style.RESET_ALL}")
                print("=" * 50)
                
                if not self.scheduled_jobs:
                    print(f"{Fore.YELLOW}No scheduled searches to cancel{Style.RESET_ALL}")
                else:
                    while True:
                        try:
                            index = input(f"{Fore.GREEN}Enter the number of the search to cancel (or 'back' to return): {Style.RESET_ALL}").strip()
                            if index.lower() == 'back':
                                break
                                
                            index = int(index) - 1
                            if 0 <= index < len(self.scheduled_jobs):
                                keyword, search_time, task = self.scheduled_jobs.pop(index)
                                task.cancel()
                                print(f"{Fore.GREEN}Cancelled search for '{keyword}' at {search_time}{Style.RESET_ALL}")
                                break
                            else:
                                print(f"{Fore.RED}Invalid selection{Style.RESET_ALL}")
                        except ValueError:
                            print(f"{Fore.RED}Please enter a valid number{Style.RESET_ALL}")
                            
            elif choice == '4':
                await self.process_bulk_search()
                
            elif choice == '5':
                print(f"{Fore.YELLOW}Returning to main menu...{Style.RESET_ALL}")
                break
                
            else:
                print(f"{Fore.RED}Invalid choice. Please enter 1-5{Style.RESET_ALL}") 

    async def process_bulk_search(self):
        """Process keywords from keywords.txt file"""
        keywords_file = Path("keywords.txt")
        
        if not keywords_file.exists():
            print(f"{Fore.RED}Error: keywords.txt file not found{Style.RESET_ALL}")
            return
            
        try:
            with open(keywords_file, 'r') as f:
                keywords = [line.strip() for line in f if line.strip()]
                
            if not keywords:
                print(f"{Fore.YELLOW}No keywords found in keywords.txt{Style.RESET_ALL}")
                return
                
            print(f"\n{Fore.CYAN}Found {len(keywords)} keywords to process:{Style.RESET_ALL}")
            for i, keyword in enumerate(keywords, 1):
                print(f"{i}. {keyword}")
                
            confirm = input(f"\n{Fore.GREEN}Start bulk search now? (y/n): {Style.RESET_ALL}").strip().lower()
            if confirm != 'y':
                print(f"{Fore.YELLOW}Bulk search cancelled{Style.RESET_ALL}")
                return
                
            print(f"\n{Fore.CYAN}Starting bulk search...{Style.RESET_ALL}")
            for keyword in keywords:
                print(f"\n{Fore.GREEN}Processing keyword: {keyword}{Style.RESET_ALL}")
                try:
                    # Lazy import to avoid circular dependency
                    from SERP_Scraper import main
                    await main(keyword)
                    self.timestamp_output(keyword)
                except Exception as e:
                    print(f"{Fore.RED}Error processing {keyword}: {e}{Style.RESET_ALL}")
                    
            print(f"\n{Fore.GREEN}Bulk search completed!{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}Error reading keywords.txt: {e}{Style.RESET_ALL}") 