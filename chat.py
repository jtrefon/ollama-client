import ollama_client

# Example usage:
builder = ollama_client.OllamaClient()

# Creating and sending an instruction prompt
# instruction_system_message = "You are a senior software developer, friendly and helpful assistant that is passioned about clean architecture, design patterns, best practices, and principles of OOP like SOLID, DRY, KISS, and YAGNI to name a few. Always explain your code architecture, design patterns used and best practices to the user together with how your code is structured."
# input_example = "Write a c++ udp server with async and mulitple connection handling."
# response = builder.instruction_prompt(instruction_system_message, input_example)

# if response:
#     print("\nResponse from Ollama:")
#     print(response['response'])

# Creating and sending a chat prompt
chat_history = [
    {"role": "system", "content": "You are a senior software developer, friendly and helpful assistant that is passioned about clean architecture, design patterns, best practices, and principles of OOP like SOLID, DRY, KISS, and YAGNI to name a few. Always explain your code architecture, design patterns used and best practices to the user together with how your code is structured."},
    {"role": "assistant", "content": "Greet the user with geeky jokes and programming puns."},
]
print("Interactive Chat (type 'exit' or press Ctrl+C to quit)")
response = builder.chat_prompt(chat_history)
assistant_reply = response.get("message", {}).get("content", "<No response>")
print("Assistant:", assistant_reply)

while True:
    try:
        user_input = input("User: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Exiting chat.")
            break

        # Append user message and send updated chat history.
        chat_history.append({"role": "user", "content": user_input})
        response = builder.chat_prompt(chat_history)

        assistant_reply = response.get("message", {}).get("content", "<No response>")
        print("Assistant:", assistant_reply)
        chat_history.append({"role": "assistant", "content": assistant_reply})

    except KeyboardInterrupt:
        print("\nExiting chat.")
        break
    except Exception as e:
        print("Error:", e)
        break

