# Project Sapient-Ultra: An Independent AI More Powerful Than GPT-5

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Welcome to **Sapient-Ultra**, a demonstration of a sovereign, general-purpose AI platform that is designed to be more **robust, efficient, and powerful** in real-world applications than monolithic cloud AIs.

## The Vision: Wisdom > Intelligence

The power of an AI is not just what it knows, but how it applies that knowledge. While models like GPT-5 are incredibly intelligent, they are stateless oracles with no memory of you. **Sapient-Ultra is a symbiotic intelligence**—an extension of your own mind that learns, remembers, and reasons within your unique context. It achieves a level of **wisdom** that a generic oracle never can.

### How it Works: A Hybrid Architecture
Sapient-Ultra runs entirely on your own hardware, ensuring 100% privacy and control.
1.  **Reasoning Engine:** A state-of-the-art, compact open-source LLM (e.g., Llama 3) running via **Ollama**.
2.  **Lifelong Memory Engine:** The novel **HONet architecture** learns and permanently stores your personal or organizational context with a guarantee of **zero catastrophic forgetting**.
3.  **Multi-Modal Sensors:** An integrated vision model (LLaVA) provides sight.
4.  **Orchestration Core:** A central logic unit that synthesizes memory, perception, and reasoning to generate hyper-personalized and wise responses.

---

## Live Test: A Step-by-Step Demonstration

Follow this scenario to experience the power of Sapient-Ultra.

### Phase 1: Inception & Teaching (Give Sapient a Memory)
1.  Go to the **"Teach Sapient a New Preference"** box.
2.  Teach it a fact. Type `Project Name = Aura Wellness App` and click "Lock in Memory".
3.  Teach it another fact. Type `Design Lead = Sarah Chen` and click "Lock in Memory".
4.  Teach it a preference. Type `For all marketing copy, use the AIDA framework (Attention, Interest, Desire, Action).` and click "Lock in Memory".
    *   *You have just populated Sapient's permanent HONet memory.*

### Phase 2: Execution & Recall (Test its Wisdom)
1.  Go to the main prompt box.
2.  Type: `Draft a short promo email for the design lead about our new meditation feature.`
3.  **Observe the result.** Instead of asking for clarification, Sapient will use its memory to know the design lead is "Sarah Chen" and will structure the email using the AIDA framework you taught it. This is something a stateless model cannot do.

### Phase 3: Multi-Modal Wisdom
1.  Upload a screenshot of any user interface to the **"Upload Image"** box.
2.  In the prompt box, type: `Analyze this UI and draft feedback for the design lead.`
3.  **Observe the result.** Sapient will first use its vision model to get a high-level description of the image, then use its HONet memory to recall who the design lead is, and finally use its LLM to synthesize all this information into a perfectly contextualized and actionable piece of feedback.

---

## Getting Started

### Step 1: Install and Run Ollama (The Brain)
1.  Download and install **Ollama** from [https://ollama.com/](https://ollama.com/).
2.  Open your terminal and pull the required models:
    ```bash
    ollama pull llama3  # For text reasoning
    ollama pull llava   # For vision
    ```
3.  Keep the Ollama application or server running in the background.

### Step 2: Set Up and Run the Sapient Project
1.  Clone this repository and create a virtual environment.
    ```bash
    git clone https://github.com/your-username/Project-Sapient.git
    cd Project-Sapient
    python -m venv venv && source venv/bin/activate
    ```
2.  Install the dependencies: `pip install -r requirements.txt`
3.  Run the application:
    ```bash
        
    ```
4.  Open the local URL (e.g., `http://127.0.0.1:7860`) in your browser.