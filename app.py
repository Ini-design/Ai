import gradio as gr
import google.generativeai as genai
import os

# Load Gemini API key from HuggingFace Secrets
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found! Add it in HuggingFace → Settings → Secrets")

genai.configure(api_key=api_key)

# Load Gemini model
model = genai.GenerativeModel("gemini-2.5-flash-preview-09-2025")

# Output generation function
def generate_output(prompt, max_length, temperature, top_p, seed):
    if not prompt.strip():
        return "Please enter a prompt."

    # Optional random seed
    generation_config = {
        "max_output_tokens": int(max_length),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    if seed != 0:
        generation_config["seed"] = int(seed)
    response = model.generate_content(
        prompt,
        generation_config=generation_config
    )
    return response.text

# Gradio UI
with gr.Blocks(title="AI Chatbot (Gemini)") as demo:
    gr.Markdown(
        """
        # ✨ AI Chatbot (Powered by Gemini)
        Generate high-quality stories using Google's Gemini model!
        """
    )
    prompt = gr.Textbox(
        label="Enter your prompt here...",
        placeholder="Example: A young explorer discovers a hidden library beneath the city...",
        lines=3
    )
    with gr.Row():
        max_length = gr.Slider(50, 600, value=300, step=25, label="Max Length")
        temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Temperature")
        top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
        seed = gr.Number(value=0, label="Random Seed (0 = random)")
        
    btn = gr.Button("Generate ✨")
    output = gr.Textbox(label="Generated Answer ✨", lines=8)
    
    btn.click(
        generate_output,
        inputs=[prompt, max_length, temperature, top_p, seed],
        outputs=output
    )
demo.launch(debug=True)
