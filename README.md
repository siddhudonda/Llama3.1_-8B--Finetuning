  # 🦙 LLaMA 3 (8B) + GRPO Fine-Tuning

This repository contains code and experiments for fine-tuning Meta's **LLaMA 3 8B** model using the **GRPO (Generalized Reinforcement Policy Optimization)** method.

## 📓 Notebook Overview

- **Notebook Name**: `Llama3_1_(8B)_GRPO.ipynb`
- **Goal**: Implement and experiment with reinforcement learning-based fine-tuning (GRPO) on the LLaMA 3 8B model.
- **Frameworks Used**: PyTorch, Transformers (HuggingFace), Accelerate, PEFT, BitsAndBytes, TRL
 
## 🚀 Features

- Load and quantize LLaMA 3 8B using 4-bit precision (via `BitsAndBytes`)
- Apply PEFT (Parameter-Efficient Fine-Tuning) with LoRA
- Configure and run GRPO-based training using the TRL (Transformers Reinforcement Learning) library
- Use `accelerate` for distributed and efficient training
- Log and visualize training using `tensorboard`
