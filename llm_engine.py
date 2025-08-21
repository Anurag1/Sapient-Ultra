import ollama
from PIL import Image
from io import BytesIO
import base64

# In llm_engine.py

class LLMEngine:
    def __init__(self, model_name="mistral", vision_model_name="llava"):
        self.model_name = model_name
        self.vision_model_name = vision_model_name
        print(f"LLM Engine initialized. Using model: '{self.model_name}'.")
        print("Ensuring local model is available...")
        try:
            # --- We comment these lines out after downloading manually ---
            # ollama.pull(self.model_name)
            # ollama.pull(self.vision_model_name)
            # ----------------------------------------------------------
            print(f"Models '{self.model_name}' and '{self.vision_model_name}' are assumed to be available locally.")
        except Exception as e:
            print(f"WARNING: Could not connect to Ollama. Please ensure Ollama is running. Error: {e}") 

    def get_text_response(self, system_prompt, user_prompt):
        print("--- Sending Text Request to Local LLM ---")
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ]
            )
            return response['message']['content']
        except Exception as e:
            return f"Error with local LLM: {e}"

    def get_image_description(self, image_path):
        print("--- Sending Image to Local Vision Model ---")
        try:
            res = ollama.chat(
                model=self.vision_model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': 'Describe this user interface screenshot in one sentence. Focus on the overall impression (e.g., clean, cluttered, simple, complex).',
                        'images': [image_path]
                    }
                ]
            )
            return res['message']['content']
        except Exception as e:
            return f"Error with local Vision Model: {e}"