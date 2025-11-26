class ConversationManager:
    def __init__(self):
        self.conversation_history = []
        self.max_history = 6

    def add_exchange(self, user_input, assistant_response):
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": assistant_response})
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

    def get_conversation_context(self):
        return self.conversation_history

    def clear_history(self):
        self.conversation_history = []
