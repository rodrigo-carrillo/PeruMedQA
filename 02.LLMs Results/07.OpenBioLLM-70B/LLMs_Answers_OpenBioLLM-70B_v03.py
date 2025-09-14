# Limit number of CPU threads to reduce memory spikes
print("Limit number of CPU threads to reduce memory spikes")
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch._dynamo
torch._dynamo.config.cache_size_limit = 4096
# ====================================================================================================





print("Import libraries")
import os
import re
import gc
import torch
import pandas as pd
from tqdm import tqdm
from time import sleep
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
# ====================================================================================================





print("Load meta-data")
df = pd.read_pickle('/scratch/rmcarri/Examenes_Residentado_Peru_OpenBioLLM/combined_exam_dataset.pkl')
print(df.head())
print(df.shape)
print(df['questions'].iloc[0])

df['question'] = df['questions'] + ' A) ' + df['option_A'] + ' B) ' + df['option_B'] + ' C) ' + df['option_C'] + ' D) ' + df['option_D'] + ' E) ' + df['option_E']
print(df['question'].iloc[0])
df['answer_llm'] = ''

# df = df.sample(n = 5, random_state = 42).reset_index(drop = True)
# ====================================================================================================





print("Download the model")
# Check if CUDA is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use {device}")

if device == "cuda":
    gpu_name = torch.cuda.get_device_name(device="cuda:0")  # Using cuda:0 as default device
    print(f"GPU name: {gpu_name}")
else:
    print("Using CPU - no GPU detected")

model_name = "aaditya/Llama3-OpenBioLLM-70B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set chat template for Llama3 if not present
if tokenizer.chat_template is None:
    tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32, # Use bfloat16 if CUDA is available, otherwise float32
    device_map = 'auto',
    # attn_implementation="flash_attention_2",
    low_cpu_mem_usage = True
    )
# ====================================================================================================





print("User template")
user_prompt = """
Instrucciones: Las siguientes son preguntas de opción múltiple sobre conocimientos médicos.
Resuélvalas paso a paso, comenzando por resumir *internamente* la información disponible y termine con "Respuesta final:" seguido *solo* de la letra correspondiente a la respuesta correcta.
No incluya en su respuesta el razonamiento paso a paso que hizo internamente.
Escriba una sola opción de las cinco como respuesta final.
Pregunta: " {original_question}  "
"""
# ====================================================================================================





print("Loop through all rows and save the LLM response (batched)")

batch_size = 5  # Tune this depending on your GPU memory
answers = []

print("Loop through rows in batches and save LLM responses")

for i in tqdm(range(0, len(df), batch_size)):
    batch_df = df.iloc[i:i + batch_size]

    for idx, row in batch_df.iterrows():
        question = row['question']
        user_message = user_prompt.format(original_question=question)

        messages = [
            {"role": "system", "content": "Eres un asistente médico experto con entrenamiento en Perú."},
            {"role": "user", "content": user_message},
        ]

        inputs = None
        generation = None
        decoded = None
        
        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,   # Tokenizer adds the assistant role automatically
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device)

            # safer terminator handling
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>") 
                if "<|eot_id|>" in tokenizer.get_vocab() else None
            ]
            terminators = [t for t in terminators if t is not None]

            input_len = inputs['input_ids'].shape[-1]

            with torch.inference_mode():
                generation = model.generate(
                    **inputs,
                    max_new_tokens=8192,   # Reduce to 512 for efficiency,
                    do_sample=False,
                    eos_token_id=terminators,
                    temperature=0.0,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
                generation = generation[0][input_len:]

            decoded = tokenizer.decode(generation, skip_special_tokens=True)
            answers.append(decoded.strip())

        except Exception as e:
            print(f"Error at index {idx}: {e}")
            answers.append("ERROR")

        # Clean up memory - only delete if variables were created
        if inputs is not None:
            del inputs
        if generation is not None:
            del generation
        if decoded is not None:
            del decoded
            
        torch.cuda.empty_cache()
        gc.collect()
        sleep(0.1)

df['answer_llm'] = answers
# ====================================================================================================





print("Save the output dataset")
model_name = model_name.split('/')[1]
df['model_basename'] = model_name
df.to_csv(f'/scratch/rmcarri/Examenes_Residentado_Peru_OpenBioLLM/combined_exam_dataset_{model_name}.csv', index = False)
df.to_parquet(f'/scratch/rmcarri/Examenes_Residentado_Peru_OpenBioLLM/combined_exam_dataset_{model_name}.parquet')
df.to_pickle(f'/scratch/rmcarri/Examenes_Residentado_Peru_OpenBioLLM/combined_exam_dataset_{model_name}.pickle')
print("END!!!")
# ====================================================================================================