class ConversationManager:
    def __init__(self):
        self.conversation_history = []
        self.max_history = 6  # Keep last 3 exchanges
        
    def add_exchange(self, user_input, assistant_response):
        """Add conversation exchange to history"""
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        self.conversation_history.append({
            "role": "assistant", 
            "content": assistant_response
        })
        
        # Keep only recent history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_conversation_context(self):
        """Get formatted conversation history for AI context"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []