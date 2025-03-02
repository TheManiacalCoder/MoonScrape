import sqlite3
from colorama import Fore, Style

# After getting SERP results
urls = [result['url'] for result in parsed_data['results']]

try:
    # Use batch insert instead of individual inserts
    with db.conn:  # Use the existing database connection
        db.conn.executemany('''INSERT OR IGNORE INTO urls (url) VALUES (?)''',
                          [(url,) for url in urls])
        print(f"{Fore.GREEN}Saved {len(urls)} URLs successfully")
except sqlite3.Error as e:
    print(f"{Fore.RED}Database error: {e}{Style.RESET_ALL}")

### Summary
[Provide a concise summary of the key information relevant to the query]

### Key Points
- [Point 1 with source reference]
- [Point 2 with source reference]
- [Point 3 with source reference]

### Conclusion
[Final assessment and confidence level based on the available knowledge] 
