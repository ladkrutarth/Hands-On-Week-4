from typing import List, Dict, Optional
from collections import deque

class ConversationMemory:
    """Manages conversation state for multiple sessions."""
    
    def __init__(self, max_history: int = 5):
        self.sessions: Dict[str, deque] = {}
        self.max_history = max_history

    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = deque(maxlen=self.max_history * 2)
        self.sessions[session_id].append({"role": role, "content": content})

    def get_history(self, session_id: str) -> str:
        if session_id not in self.sessions:
            return ""
        
        history = []
        for msg in self.sessions[session_id]:
            history.append(f"{msg['role'].upper()}: {msg['content']}")
        return "\n".join(history)

    def clear(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

# Singleton instance for the system
_global_memory = ConversationMemory()

def get_memory():
    return _global_memory
