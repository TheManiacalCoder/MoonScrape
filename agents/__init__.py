# This file makes the agents directory a Python package
from .analyzer import OpenRouterAnalyzer
from .content_processor import ContentProcessor
from .intent_agent import IntentAgent
from .local_agent import LocalAgent

def __getattr__(name):
    if name == 'ScheduledSearch':
        from .scheduled_search import ScheduledSearch
        return ScheduledSearch
    raise AttributeError(f"module 'agents' has no attribute '{name}'") 