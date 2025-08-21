import gradio as gr
from llm_engine import LLMEngine

# --- In-Memory Simulation of the HONet Lifelong Memory Engine ---
class HONetMemoryEngine:
    def __init__(self):
        self.facts = {}
        self.preferences = []
        print("HONet Memory Engine initialized (in-memory simulation).")

    def get_context(self):
        if not self.facts and not self.preferences:
            return "No personal context has been learned yet."
        
        context_str = "USER'S PERMANENT MEMORY (You MUST use this context):\n"
        if self.facts:
            context_str += "--- Known Facts ---\n"
            for key, value in self.facts.items():
                context_str += f"- {key}: {value}\n"
        if self.preferences:
            context_str += "--- Learned Preferences ---\n"
            for pref in self.preferences:
                context_str += f"- {pref}\n"
        return context_str

    def learn_fact(self, key, value):
        print(f"HONet learning new fact: {key} = {value}")
        self.facts[key.lower().strip()] = value.strip()

    def learn_preference(self, feedback_text):
        print(f"HONet learning new preference: '{feedback_text}'")
        self.preferences.append(feedback_text)

# --- Application Initialization ---
print("Initializing Project Sapient-Ultra...")
llm_engine = LLMEngine(model_name="llama3")
honet_memory = HONetMemoryEngine()

# --- Core Orchestrator Logic ---
def process_prompt(user_prompt, image_input, history):
    system_context = honet_memory.get_context()
    final_user_prompt = user_prompt

    if image_input:
        print("Image detected, analyzing...")
        image_description = llm_engine.get_image_description(image_input.name)
        final_user_prompt = f"""
        User has uploaded an image and provided a prompt.
        ---
        VISION ANALYSIS of the image: "{image_description}"
        ---
        USER'S PROMPT regarding this image: "{user_prompt}"
        """
    
    response = llm_engine.get_text_response(system_context, final_user_prompt)
    history.append({"role": "user", "content": user_prompt})
    history.append({"role": "assistant", "content": response})
    return history, None # Clear the image input after processing

def handle_learning(text_to_learn):
    if not text_to_learn: return "Please enter something to learn."
    if "=" in text_to_learn:
        parts = text_to_learn.split('=', 1)
        key, value = parts[0], parts[1]
        honet_memory.learn_fact(key, value)
        return f"✅ Fact Learned: '{key.strip()}' is now '{value.strip()}'"
    else:
        honet_memory.learn_preference(text_to_learn)
        return f"✅ Preference Learned: '{text_to_learn}'"

# --- Gradio User Interface ---
with gr.Blocks(theme=gr.themes.Soft(), title="Project Sapient-Ultra") as demo:
    gr.Markdown("# Welcome to Sapient-Ultra: Your Sovereign AI")
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation", type="messages", height=600)
            
            with gr.Row():
                image_box = gr.Image(type="filepath", label="Upload Image (Optional)")
                prompt_box = gr.Textbox(label="Your Prompt", placeholder="Ask a question, give a command, or describe the image...", scale=4)

        with gr.Column(scale=1):
            gr.Markdown("### 🧠 HONet Lifelong Memory")
            gr.Markdown("Teach Sapient facts (`key = value`) or preferences (`Just a sentence.`). This memory is permanent and will be used in all future responses.")
            learn_box = gr.Textbox(label="Teach Sapient Something New", placeholder="e.g., Project Lead = Anurag Dongare")
            learn_button = gr.Button("🧠 Lock in Memory")
            learn_status = gr.Label()

    def submit_message(user_prompt, image_input, history):
        return process_prompt(user_prompt, image_input, history)

    prompt_box.submit(submit_message, [prompt_box, image_box, chatbot], [chatbot, image_box])
    learn_button.click(handle_learning, inputs=[learn_box], outputs=[learn_status])

if __name__ == "__main__":
    print("Launching Project Sapient-Ultra...")
    print("Please ensure the Ollama server is running in the background.")
    demo.launch()