import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

def main():
    # === 1. 配置路径和参数 ===
    merged_model_path = "models/qwen2_5-0.5b-instruct-skincare-merged"
    base_model_path = "models/Qwen2.5-0.5B-Instruct"
    test_data_path = "data/val_dataset_final.jsonl"
    output_csv_path = "data/evaluation_results_final_2.csv"

    generation_config = {
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95,
    }

    batch_size = 8
    print("--- 开始模型对比评测 (优化版 + TQDM进度条) ---")

    # === 2. 加载模型和Tokenizer (已修复padding问题) ===
    print(f"正在加载微调模型: {merged_model_path}")
    merged_model = AutoModelForCausalLM.from_pretrained(merged_model_path, device_map="auto", dtype="auto")
    merged_tokenizer = AutoTokenizer.from_pretrained(merged_model_path, padding_side='left')

    print(f"正在加载基础模型: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", dtype="auto")
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side='left')

    if merged_tokenizer.pad_token is None:
        merged_tokenizer.pad_token = merged_tokenizer.eos_token
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    # === 3. 创建Pipeline用于推理 ===
    merged_pipe = pipeline("text-generation", model=merged_model, tokenizer=merged_tokenizer)
    base_pipe = pipeline("text-generation", model=base_model, tokenizer=base_tokenizer)

    # === 4. 加载并准备所有prompts (无变化) ===
    print(f"正在加载并准备所有prompts...")
    test_dataset = load_dataset("json", data_files=test_data_path, split="train")
    
    prompts_and_chosen = []
    merged_prompts_formatted = []
    base_prompts_formatted = []
    for sample in test_dataset:
        prompt_text = sample["prompt"]
        chosen_text = sample["chosen"]
        messages = [{"role": "user", "content": prompt_text}]
        prompts_and_chosen.append({"prompt": prompt_text, "golden_answer": chosen_text})
        merged_prompts_formatted.append(merged_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        base_prompts_formatted.append(base_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    # === 5. --- MODIFIED --- 手动分批并使用tqdm进行批量推理 ===
    
    # --- 处理微调模型 ---
    print("开始对微调模型进行批量推理...")
    merged_outputs_list = []
    # 使用tqdm包裹分批循环
    for i in tqdm(range(0, len(merged_prompts_formatted), batch_size), desc="正在处理微调模型"):
        # 从总列表中切分出一个小批次
        batch_prompts = merged_prompts_formatted[i:i + batch_size]
        # 对这个小批次进行推理
        outputs = merged_pipe(batch_prompts, **generation_config)
        # 收集结果
        merged_outputs_list.extend(outputs)

    # --- 处理基础模型 ---
    print("开始对基础模型进行批量推理...")
    base_outputs_list = []
    # 再次使用tqdm包裹分批循环
    for i in tqdm(range(0, len(base_prompts_formatted), batch_size), desc="正在处理基础模型"):
        batch_prompts = base_prompts_formatted[i:i + batch_size]
        outputs = base_pipe(batch_prompts, **generation_config)
        base_outputs_list.extend(outputs)

    # === 6. 组合结果并保存 (逻辑微调) ===
    print("评测完成，正在组合结果并保存到CSV文件...")
    results = []
    for i in range(len(prompts_and_chosen)):
        finetuned_full_text = merged_outputs_list[i][0]['generated_text']
        finetuned_answer = finetuned_full_text.replace(merged_prompts_formatted[i], "")
        
        base_full_text = base_outputs_list[i][0]['generated_text']
        base_answer = base_full_text.replace(base_prompts_formatted[i], "")

        results.append({
            "prompt": prompts_and_chosen[i]["prompt"],
            "golden_answer": prompts_and_chosen[i]["golden_answer"],
            "finetuned_model_output": finetuned_answer,
            "base_model_output": base_answer
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    print(f"✅ 评测结果已成功保存至: {output_csv_path}")

if __name__ == "__main__":
    main()
