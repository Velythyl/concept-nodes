"""Chat history management for the 3D scene visualization agent.

This module provides a ChatHistory class that maintains conversation context
for the Langchain agent, separate from the UI display concerns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Role = Literal["user", "assistant"]


@dataclass
class ChatMessage:
    """A single chat message with role and content."""
    role: Role
    content: str


@dataclass
class ChatHistory:
    """Manages chat history for agent context.
    
    This class maintains the conversation history that gets passed to the
    Langchain agent for context. It's separate from UI display concerns.
    """
    messages: list[ChatMessage] = field(default_factory=list)
    max_messages: int = 50  # Limit history to prevent context overflow
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to the history."""
        self.messages.append(ChatMessage(role="user", content=content))
        self._trim_if_needed()
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the history."""
        self.messages.append(ChatMessage(role="assistant", content=content))
        self._trim_if_needed()
    
    def _trim_if_needed(self) -> None:
        """Trim oldest messages if we exceed max_messages."""
        if len(self.messages) > self.max_messages:
            # Keep the most recent messages
            self.messages = self.messages[-self.max_messages:]
    
    def get_langchain_messages(self) -> list[dict]:
        """Get messages in Langchain format for the agent.
        
        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.messages
        ]
    
    def clear(self) -> None:
        """Clear all chat history."""
        self.messages = []
    
    def __len__(self) -> int:
        """Return the number of messages in history."""
        return len(self.messages)
    
    def is_empty(self) -> bool:
        """Check if history is empty."""
        return len(self.messages) == 0


# Callback type for UI updates
from typing import Callable, Protocol


class ChatDisplayCallback(Protocol):
    """Protocol for chat display callbacks."""
    
    def __call__(self, sender: str, message: str) -> None:
        """Called when a message should be displayed in the UI."""
        ...


@dataclass
class ChatManager:
    """Manages both chat history (for agent context) and UI display.
    
    This is the main interface used by the agent and visualizer.
    It keeps the history management separate from display concerns.
    """
    history: ChatHistory = field(default_factory=ChatHistory)
    display_callback: ChatDisplayCallback | None = None
    
    def set_display_callback(self, callback: ChatDisplayCallback) -> None:
        """Set the callback for UI display updates."""
        self.display_callback = callback
    
    def add_user_message(self, content: str) -> None:
        """Add a user message and notify display."""
        content = content.strip()
        if not content:
            return
        
        self.history.add_user_message(content)
        
        if self.display_callback:
            self.display_callback("User", content)
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message and notify display."""
        content = content.strip()
        if not content:
            return
        
        self.history.add_assistant_message(content)
        
        if self.display_callback:
            self.display_callback("Agent", content)
    
    def get_context_messages(self) -> list[dict]:
        """Get all messages for agent context."""
        return self.history.get_langchain_messages()
    
    def clear(self) -> None:
        """Clear chat history."""
        self.history.clear()
