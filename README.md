


<h1 align="center">Circuit</h1>
<p align="center">Fine-tuned Phi-3 for Logical Reasoning</p>

<p align="center">
  <img src="https://i.postimg.cc/Nfnst2F9/Circuit.png" alt="Circuit Logo" style="max-width:100%; height:auto;">
</p>


# Model performance 

## Benchmark


<p align="center">
  <img src="https://i.postimg.cc/85pjRhwf/daata.png" alt="App Screenshot" style="max-width:100%; height:auto;">
</p>

Trained on the [lucasmccabe/logiqa](https://huggingface.co/datasets/lucasmccabe/logiqa) dataset, Circuit enhances the model’s ability to reason through complex problems, answer multi-step logic questions, and provide consistent explanations.


#  Model Details

| Property | Value |
|-----------|--------|
| **Base model** | `microsoft/Phi-3-mini-4k-instruct` |
| **Fine-tuned for** | Logical Reasoning |
| **Dataset** | [`lucasmccabe/logiqa`](https://huggingface.co/datasets/lucasmccabe/logiqa) |
| **Technique** | LoRA fine-tuning, merged for direct use |
| **Formats available** | Full (HF Transformers) + Quantized (`.gguf` for llama.cpp / Ollama) |
| **Project** | **Circuit** |
| **Fine-tuned by** | Rudransh |





#  Model Variants

| Variant | Description | File |
|----------|--------------|------|
|  **Full model** | Merged LoRA with base, compatible with `transformers` | `pytorch_model.bin` |
|  **Quantized model (GGUF)** | Optimized for CPU/GPU inference via `llama.cpp`, `text-generation-webui`, or `Ollama` | `circuit_phi3_q4.gguf` |

#  Example Usage (Transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("rudransh/circuit", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("rudransh/circuit")

prompt = "If all squares are rectangles, and all rectangles have four sides, what can we conclude about squares?"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=150)
print(tokenizer.decode(outputs[0], skip_special_tokens=True)

```


# Training Summary

Base model: Phi-3 Mini 4K Instruct

Dataset: LogiQA (lucasmccabe/logiqa)

Training method: LoRA fine-tuning, later merged

Hardware: NVIDIA RTX 1080

Epochs: ~3

Objective: Improve reasoning consistency and structured explanations



# Acknowledgements

Microsoft
 for Phi-3

Lucas McCabe
 for LogiQA dataset

Fine-tuned and quantized by Rudransh under Project Circuit 

