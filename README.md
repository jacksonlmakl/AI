### Explanation

1. **Imports**:
   - Import `ChatMessageHistory` from `langchain.memory` to handle chat history.

2. **Initialization**:
   - Use `ChatMessageHistory` for managing memory. This class provides methods for adding and retrieving chat messages.

3. **Memory Handling**:
   - In the `invoke` method of `CustomChain`, retrieve messages from `ChatMessageHistory` to maintain the context.
   - Add user and AI messages to `ChatMessageHistory` after generating a response.

4. **Cache Management**:
   - Save and load memory from `memory_cache.json` to persist chat history across sessions.

### Key Concepts

- **Reading and Writing Memory**: Memory is read before executing the core logic and written after generating the response to maintain continuity in conversations.
- **Memory Structure**: Messages can be stored as a list of `ChatMessages` and queried as needed.

### References
- LangChain Memory Management: [LangChain Documentation](https://python.langchain.com/v0.1/docs/modules/memory/chat_messages/)
- Using ChatMessageHistory: [LangChain Chat Messages](https://python.langchain.com/v0.1/docs/modules/memory/chat_messages/)

By following this approach, you can effectively manage memory in your LangChain application, enabling your conversational AI to maintain context and provide more coherent responses over multiple interactions.