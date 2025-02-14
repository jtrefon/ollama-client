import json
import requests

class OllamaClient:
    """
    A client for interfacing with a local Ollama API using the Deepseek model.
    Provides methods to send instruction and chat prompts.
    """
    def __init__(self):
        self.ollama_api_url = "http://localhost:11434/api/"
        self._model_name = "deepseek-r1:32b"
        self._is_streaming = False
        self._ctx_size = 16384
        self._temperature = 0.5
        self._format = "json"
    
    def set_model_name(self, model_name: str) -> None:
        """Set the model name used for generating responses."""
        self._model_name = model_name
    
    def set_streaming(self, is_streaming: bool) -> None:
        """Set whether the response should be streamed or not."""
        self._is_streaming = is_streaming

    def set_temperature(self, temperature : float) -> None:
        """Adjust the temperature parameter for response randomness."""
        self._temperature = temperature

    def set_format(self, format : str) -> None:
        """Set the format of the response."""
        self._format = format

    def instruction_prompt(self, system_message: str, input_example: str) -> dict:
        """
        Send an instruction prompt to the Ollama API.

        Args:
            system_message: The context or instruction for the model.
            input_example: The prompt content or input example.

        Returns:
            A dictionary representing the API's JSON response.
        """
        prompt = {
            "model": self._model_name,
            "prompt": input_example,
            "system": system_message,
            "stream": self._is_streaming,
            "options": {
                "temperature": self._temperature,
                "num_ctx": self._ctx_size,
            }
        }
        return self._send_to_ollama(json.dumps(prompt))

    def chat_prompt(self, messages: list) -> dict:
        """
        Send a chat prompt to the Ollama API.

        Args:
            messages: A list of message dictionaries, e.g.,
                      [{"role": "user", "content": "Hello, world!"}]

        Returns:
            A dictionary representing the API's JSON response.
        """
        prompt = {
            "model": self._model_name,
            "messages": messages,
            "stream": self._is_streaming,
            "options": {
                "temperature": self._temperature,
                "num_ctx": self._ctx_size,
            }
        }
        return self._send_to_ollama(json.dumps(prompt), chat=True)
    
    def _send_to_ollama(self, prompt_json: str, chat: bool=False) -> dict:
        """
        Send the JSON payload to the Ollama API endpoint.

        Args:
            prompt_json: The JSON-formatted prompt payload.
            chat: Determines which endpoint to use (chat or generate).

        Returns:
            The JSON response from the API as a dictionary.

        Raises:
            Exception: If the API request fails.
        """
        headers = {'Content-Type': 'application/json'}
        try:
            response = requests.post(
                self.ollama_api_url + "chat" if chat else self.ollama_api_url + "generate",
                headers=headers,
                data=prompt_json
            )
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Request failed with status code: {response.status_code}. Error: {response.text}")
        except Exception as e:
            raise Exception(f"Request failed with exception: {e}")

