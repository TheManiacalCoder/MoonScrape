from storage.database_manager import DatabaseManager
from config.manager import ConfigManager

# Initialize database and config
db = DatabaseManager(reset_db=False)
config = ConfigManager() 