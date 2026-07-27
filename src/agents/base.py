from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

class AgentAction(BaseModel):
    step: int
    tool: str
    args: Dict[str, Any]
    result: str

class AgentResult(BaseModel):
    answer: str
    actions: List[AgentAction]
    status: str = "success"
    session_id: Optional[str] = None
    trace: Optional[List[str]] = None

class BaseAgent(ABC):
    @abstractmethod
    def run(self, query: str, session_id: Optional[str] = None) -> AgentResult:
        pass
