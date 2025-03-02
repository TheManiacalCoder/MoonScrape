import asyncio
from datetime import datetime, timedelta
from typing import Optional
import schedule
import time
from pathlib import Path
from SERP_Scraper import main, show_title_screen, db, config
from colorama import Fore, Style

class ScheduledSearch:
    def __init__(self):
        self.scheduled_jobs = []
        self.running = False

    async def run_scheduled_search(self, keyword: str, search_time: str):
        """Run a search at the specified time"""
        try:
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
            print("\nScheduled Search Menu:")
            print("1. Schedule new search")
            print("2. View scheduled searches")
            print("3. Cancel scheduled search")
            print("4. Return to main menu")
            
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == '1':
                keyword = input("Enter search keyword: ").strip()
                search_time = input("Enter search time (YYYY-MM-DD HH:MM): ").strip()
                
                try:
                    # Validate time format
                    datetime.strptime(search_time, "%Y-%m-%d %H:%M")
                    task = asyncio.create_task(self.run_scheduled_search(keyword, search_time))
                    self.scheduled_jobs.append((keyword, search_time, task))
                    print(f"{Fore.GREEN}Search scheduled successfully!{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Invalid time format. Please use YYYY-MM-DD HH:MM{Style.RESET_ALL}")
                    
            elif choice == '2':
                if not self.scheduled_jobs:
                    print(f"{Fore.YELLOW}No scheduled searches{Style.RESET_ALL}")
                else:
                    print("\nScheduled Searches:")
                    for i, (keyword, search_time, _) in enumerate(self.scheduled_jobs, 1):
                        print(f"{i}. {keyword} at {search_time}")
                        
            elif choice == '3':
                if not self.scheduled_jobs:
                    print(f"{Fore.YELLOW}No scheduled searches to cancel{Style.RESET_ALL}")
                else:
                    try:
                        index = int(input("Enter the number of the search to cancel: ")) - 1
                        if 0 <= index < len(self.scheduled_jobs):
                            _, _, task = self.scheduled_jobs.pop(index)
                            task.cancel()
                            print(f"{Fore.GREEN}Search cancelled successfully{Style.RESET_ALL}")
                        else:
                            print(f"{Fore.RED}Invalid selection{Style.RESET_ALL}")
                    except ValueError:
                        print(f"{Fore.RED}Please enter a valid number{Style.RESET_ALL}")
                        
            elif choice == '4':
                break
                
            else:
                print(f"{Fore.RED}Invalid choice. Please enter 1-4{Style.RESET_ALL}")

# Add to SERP_Scraper.py menu
async def main_loop():
    scheduled_search = ScheduledSearch()
    
    while True:
        print("\nMain Menu:")
        print("1. Perform new web search")
        print("2. Query local knowledge base")
        print("3. Schedule search")
        print("4. Exit")
        
        choice = input("Enter menu selection (1-4): ").strip()
        
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
            await main()
            
        elif choice == '2':
            question = input("\nEnter your question: ").strip()
            await query_local_knowledge(question)
            
        elif choice == '3':
            await scheduled_search.schedule_search_menu()
            
        elif choice == '4' or choice.lower() == 'exit':
            print("Exiting MoonScrape. Goodbye!")
            break
            
        else:
            print(f"{Fore.RED}Invalid choice. Please enter 1-4.{Style.RESET_ALL}")
        
        print("\n" + "=" * 50)
        print("Ready for new operation")
        print("=" * 50) 