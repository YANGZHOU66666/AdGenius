import json
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 配置变量
model_path = "models/Qwen2.5-0.5B-Instruct"  # 指定模型路径
output_dir = "results/final_test_results.csv"  # 指定输出文件路径

def load_model_and_tokenizer(model_path):
    """加载模型和分词器"""
    print(f"正在加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("模型加载完成")
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=512):
    """生成回答"""
    # 构建输入
    messages = [{"content": prompt, "role": "user"}]
    
    # 应用聊天模板
    input_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 编码输入
    inputs = tokenizer(input_text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码输出
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

def test_model_on_validation_set():
    """在验证集上测试模型"""
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # 读取验证数据集
    val_data = []
    with open("data/val_dataset_final.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                val_data.append(json.loads(line.strip()))
    
    print(f"验证集共有 {len(val_data)} 条数据")
    
    # 生成结果
    results = []
    for i, data in enumerate(val_data):
        prompt = data['prompt']
        print(f"正在处理第 {i+1}/{len(val_data)} 条数据...")
        
        try:
            model_output = generate_response(model, tokenizer, prompt)
            results.append({
                "prompt": prompt,
                "model_output": model_output
            })
        except Exception as e:
            print(f"处理第 {i+1} 条数据时出错: {e}")
            results.append({
                "prompt": prompt,
                "model_output": f"生成失败: {str(e)}"
            })
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(output_dir, index=False, encoding='utf-8')
    print(f"结果已保存到: {output_dir}")
    print(f"共处理 {len(results)} 条数据")

if __name__ == "__main__":
    test_model_on_validation_set()
