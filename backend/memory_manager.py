from typing import List, Dict

class MemoryManager:
    """
    Memory manager that maintains complete conversation pairs.
    No LangChain dependency - pure Python, lightweight and efficient.
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize memory manager with pair-based windowing.
        
        Args:
            window_size (int): Number of recent MESSAGE PAIRS to retain.
                             5 pairs = 10 messages (5 user + 5 assistant)
                             This ensures context is never cut mid-conversation.
        """
        self.window_size = window_size

    def get_recent_history(self, full_history: List[Dict]) -> List[Dict]:
        """
        Get last N complete conversation pairs (user + assistant).
        
        Args:
            full_history: List of all messages [{"role": "user/assistant", "content": "..."}]
            
        Returns:
            List[Dict]: Last N pairs = (window_size * 2) messages
        """
        if not full_history:
            return []
        
        # Return last (window_size * 2) messages to get complete pairs
        # Example: window_size=5 means last 10 messages (5 Q&A pairs)
        max_messages = self.window_size * 2
        return full_history[-max_messages:]
    
    def get_last_pair(self, full_history: List[Dict]) -> List[Dict]:
        """
        Get only the most recent user-assistant pair.
        Useful for follow-up questions where only immediate context matters.
        
        Returns:
            Last 2 messages (1 Q&A pair) or empty if less than 2 messages
        """
        if len(full_history) < 2:
            return []
        return full_history[-2:]
    
    def estimate_tokens(self, messages: List[Dict]) -> int:
        """
        Estimate token count for messages (no API call).
        Rule of thumb: 1 token ≈ 4 characters for English text.
        
        Args:
            messages: List of message dicts
            
        Returns:
            Estimated token count
        """
        total_chars = sum(len(msg.get('content', '')) for msg in messages)
        # More accurate estimation: ~4 chars per token for English
        return total_chars // 4
    
    def format_for_display(self, history: List[Dict]) -> str:
        """
        Format memory for debugging/logging.
        
        Args:
            history: List of message dicts
            
        Returns:
            Formatted string with role and content
        """
        if not history:
            return "No conversation history"
        
        formatted_lines = []
        for i, msg in enumerate(history, 1):
            role = msg.get('role', 'unknown').capitalize()
            content = msg.get('content', '')
            # Truncate long messages for display
            preview = content[:100] + "..." if len(content) > 100 else content
            formatted_lines.append(f"{i}. [{role}]: {preview}")
        
        return "\n".join(formatted_lines)
    
    def get_stats(self, full_history: List[Dict]) -> Dict:
        """
        Get memory statistics for monitoring (no API calls).
        
        Args:
            full_history: Complete conversation history
            
        Returns:
            Dict with memory stats including token estimates
        """
        recent = self.get_recent_history(full_history)
        
        return {
            "total_messages_in_session": len(full_history),
            "window_size_pairs": self.window_size,
            "messages_in_memory": len(recent),
            "estimated_memory_tokens": self.estimate_tokens(recent),
            "memory_pairs": len(recent) // 2
        }
    
    def validate_history(self, history: List[Dict]) -> bool:
        """
        Validate that history contains properly formatted messages.
        
        Args:
            history: List of message dicts to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not history:
            return True
        
        for msg in history:
            if not isinstance(msg, dict):
                return False
            if 'role' not in msg or 'content' not in msg:
                return False
            if msg['role'] not in ['user', 'assistant']:
                return False
        
        return True