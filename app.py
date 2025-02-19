import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

repo_id = "YasinArafat05/llama2-7b-gym"  

tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=torch.float16, device_map="auto")
model.eval()


def chat(input_text):
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


chatbot_ui = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(lines=2, placeholder="Ask me anything..."),
    outputs="text",
    title="Fine-Tuned LLaMA-2 Chatbot",
    description="A chatbot fine-tuned on custom data, powered by LLaMA-2",
)


chatbot_ui.launch()
