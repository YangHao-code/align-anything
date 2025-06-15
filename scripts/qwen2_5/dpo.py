from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from tqdm import tqdm

# 模型路径
dpo_model_path = "/root/autodl-tmp/align-anything/outputs/qwen_2_5_dpo/slice_end"
base_model_path = "/root/autodl-tmp/data/model/model"

# 加载模型与tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dpo_model = AutoModelForCausalLM.from_pretrained(dpo_model_path).to(device)
dpo_tokenizer = AutoTokenizer.from_pretrained(dpo_model_path, padding_side="left")

base_model = AutoModelForCausalLM.from_pretrained(base_model_path).to(device)
base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side="left")

print("Padding side:", base_tokenizer.padding_side)  # 应该输出 "left"

# 加载验证集
val_path = "/root/autodl-tmp/data/data/data/val.jsonl"
output_path = "/root/autodl-tmp/align-anything/eval_outputs_dpo/eval_dpo_vs_base.jsonl"

# 读取全部数据
with open(val_path, "r") as f:
    val_data = [json.loads(line) for line in f]

# Prompt转chat格式
def build_prompt_text(tokenizer, prompt):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 批量生成函数
def batch_generate(model, tokenizer, prompts, max_new_tokens=512):
    texts = [build_prompt_text(tokenizer, p) for p in prompts]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )
    responses = []
    for input_ids, output_ids in zip(inputs.input_ids, outputs):
        gen_ids = output_ids[len(input_ids):]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True)
        responses.append(response)
    return responses

# 主循环
batch_size = 50
with open(output_path, "w") as fout:
    for i in tqdm(range(0, len(val_data), batch_size)):
        batch = val_data[i:i+batch_size]
        questions = [item["question"] for item in batch]

        base_outputs = batch_generate(base_model, base_tokenizer, questions)
        dpo_outputs = batch_generate(dpo_model, dpo_tokenizer, questions)

        for item, base_response, dpo_response in zip(batch, base_outputs, dpo_outputs):
            output_item = {
                "question": item["question"],
                "response_1": base_response,
                "response_2": dpo_response,
                "overall_response": 1
            }
            fout.write(json.dumps(output_item, ensure_ascii=False) + "\n")
