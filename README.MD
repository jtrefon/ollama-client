# Ollama Client

A lightweight Python client for interfacing with a local [Ollama API](http://localhost:11434/api/) using a Deepseek model. This project provides an easy-to-use interface to send instruction and chat prompts to a local Ollama instance.

## Features

- **Instruction Prompt**: Send a system/instruction prompt that receives a generated response.
- **Chat Prompt**: Engage in a conversation by sending and receiving chat messages.
- **Simple Demo**: Includes a demo implementation of both instruction and chat prompts in [`chat.py`](chat.py).

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/jtrefon/ollama-client.git
   cd ollama-client
   ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```
## Usage

### OllamaClient Class
The core functionality is provided by the OllamaClient class in ollama_client.py. The class includes two main methods:

- **instruction_prompt**: Sends a system/instruction prompt and returns the generated response.
- **chat_prompt**: Sends a chat message and returns the generated response.

## Configuration
Customize the client by setting:

- **Model Name**: Use set_model_name(model_name) to change the model. Default is deepseek-r1:32b.
- **Streaming**: Enable or disable streaming with set_streaming(is_streaming).
- **Temperature**: Adjust the response randomness with set_temperature(temperature).
- **Format**: Set the response format using set_format(format).


## Contributing
Contributions are welcome! If you have suggestions or improvements, please feel free to open an issue or submit a pull request. Any enhancements that help others interface with a local Ollama model are appreciated.

## License
GPL-3.0 license

## Acknowledgements
This project is a small contribution to help the community interface with a local Ollama model, particularly but not exclusively the Deepseek model.