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





print("Load meta-data")
df = pd.read_pickle('/scratch/rmcarri/Examenes_Residentado_Peru_JSL_MedLlama/combined_exam_dataset.pkl')
print(df.head())
print(df.shape)
print(df['questions'].iloc[0])

df['question'] = df['questions'] + ' A) ' + df['option_A'] + ' B) ' + df['option_B'] + ' C) ' + df['option_C'] + ' D) ' + df['option_D'] + ' E) ' + df['option_E']
print(df['question'].iloc[0])
df['answer_llm'] = ''





print("Download the model")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use {device}")

if device == "cuda":
    gpu_name = torch.cuda.get_device_name(device="cuda:0")
    print(f"GPU name: {gpu_name}")
else:
    print("Using CPU - no GPU detected")

model_name = "johnsnowlabs/JSL-MedLlama-3-8B-v2.0"





print("Loading tokenizer with optimizations...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    padding_side="left",  # Better for batch processing
    truncation_side="left"  # Truncate from left to keep important question parts
)

# Set chat template for Llama3 if not present
if tokenizer.chat_template is None:
    tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token





print("Loading model with speed optimizations...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map='auto',
    # Try without flash_attention_2 first - it can be slower on some setups
    # attn_implementation="flash_attention_2",  # Comment out to test
    low_cpu_mem_usage=True,
    # Additional optimizations
    use_cache=True,
    trust_remote_code=True
)





# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()
print("Model loaded successfully!")

print("Optimized user template")
# Shorter, more direct prompt for faster processing
user_prompt = """
Instrucciones: Las siguientes son preguntas de opción múltiple sobre conocimientos médicos.
Resuélvalas paso a paso, comenzando por resumir *internamente* la información disponible y termine con "Respuesta final:" seguido *solo* de la letra correspondiente a la respuesta correcta. Por ejemplo: 'Respuesta final:X'.
No incluya en su respuesta el razonamiento paso a paso que hizo internamente.
Escriba una sola opción de las cinco como respuesta final.
Pregunta: " {original_question}  "
"""





# Pre-compute terminators outside the loop
print("Setting up generation parameters...")
terminators = [tokenizer.eos_token_id]
if "<|eot_id|>" in tokenizer.get_vocab():
    terminators.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))

print("Processing with TRUE batch inference for maximum speed")
batch_size = 8  # Increased batch size for better GPU utilization
answers = []

import time
start_time = time.time()
total_questions = len(df)

# Process in true batches for maximum speed
for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
    batch_df = df.iloc[i:i + batch_size]
    
    # Prepare all messages in the batch
    batch_messages = []
    for _, row in batch_df.iterrows():
        question = row['question']
        user_message = user_prompt.format(original_question=question)
        
        messages = [
            {"role": "system", "content": "Eres un asistente médico experto con entrenamiento en Perú."},  # Shorter system message
            {"role": "user", "content": user_message}
        ]
        batch_messages.append(messages)
    
    try:
        # Tokenize entire batch at once
        batch_inputs = []
        for messages in batch_messages:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
            batch_inputs.append(inputs['input_ids'])
        
        # Pad sequences to same length for batch processing
        from torch.nn.utils.rnn import pad_sequence
        padded_inputs = pad_sequence(batch_inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = (padded_inputs != tokenizer.pad_token_id).long()
        
        # Move to device
        batch_inputs_dict = {
            'input_ids': padded_inputs.to(model.device),
            'attention_mask': attention_mask.to(model.device)
        }
        
        input_len = padded_inputs.shape[-1]
        
        # Generate for entire batch - THIS IS THE KEY SPEEDUP
        with torch.inference_mode():
            batch_generation = model.generate(
                **batch_inputs_dict,
                max_new_tokens=256,  # SIGNIFICANTLY REDUCED - medical MCQ answers are short
                do_sample=False,  # Deterministic for consistency
                eos_token_id=terminators,
                pad_token_id=tokenizer.pad_token_id,
                temperature=0.0,
                use_cache=True,
                # Speed optimizations
                num_beams=1,  # No beam search
                early_stopping=True,
                output_attentions=False,
                output_hidden_states=False,
                # Batch-specific optimizations
                batch_size=len(batch_messages)
            )
        
        # Decode all outputs in the batch
        for j, generation in enumerate(batch_generation):
            # Extract only the generated part
            generated_tokens = generation[input_len:]
            decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            answers.append(decoded.strip())
        
        # Clean up batch tensors
        del batch_inputs_dict, batch_generation, padded_inputs, attention_mask
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error processing batch starting at index {i}: {e}")
        # Fallback: process individually for this batch
        for _, row in batch_df.iterrows():
            try:
                question = row['question']
                user_message = user_prompt.format(original_question=question)
                messages = [
                    {"role": "system", "content": "Eres un médico experto."},
                    {"role": "user", "content": user_message}
                ]
                
                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(model.device)
                
                input_len = inputs['input_ids'].shape[-1]
                
                with torch.inference_mode():
                    generation = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        eos_token_id=terminators,
                        pad_token_id=tokenizer.pad_token_id,
                        temperature=0.0,
                        use_cache=True
                    )
                
                generated_tokens = generation[0][input_len:]
                decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                answers.append(decoded.strip())
                
                del inputs, generation
                
            except Exception as e2:
                print(f"Error in fallback processing: {e2}")
                answers.append("ERROR")
        
        torch.cuda.empty_cache()
        gc.collect()
    
    # Progress tracking every 50 questions
    if (i + batch_size) % 50 == 0:
        elapsed = time.time() - start_time
        processed = min(i + batch_size, total_questions)
        rate = processed / elapsed * 60  # questions per minute
        estimated_total = (total_questions / processed) * elapsed / 60  # minutes
        print(f"Processed {processed}/{total_questions}. Rate: {rate:.1f} q/min. ETA: {estimated_total:.1f} min")

df['answer_llm'] = answers





# Final timing
total_time = time.time() - start_time
print(f"Total processing time: {total_time/60:.1f} minutes")
print(f"Average time per question: {total_time/len(df):.2f} seconds")
print(f"Questions per minute: {len(df)/(total_time/60):.1f}")





print("Save the output dataset")
model_name_clean = model_name.split('/')[1]
df['model_basename'] = model_name_clean
df.to_csv(f'/scratch/rmcarri/Examenes_Residentado_Peru_JSL_MedLlama/combined_exam_dataset_{model_name_clean}_optimized.csv', index=False)
df.to_parquet(f'/scratch/rmcarri/Examenes_Residentado_Peru_JSL_MedLlama/combined_exam_dataset_{model_name_clean}_optimized.parquet')
df.to_pickle(f'/scratch/rmcarri/Examenes_Residentado_Peru_JSL_MedLlama/combined_exam_dataset_{model_name_clean}_optimized.pickle')
print("END!!!")