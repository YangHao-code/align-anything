from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from tqdm import tqdm

# 模型路径设置
dpo_model_path = "/root/autodl-tmp/align-anything/outputs/qwen_2_5_dpo/slice_end"
base_model_path = "/root/autodl-tmp/data/model/model"

# 加载模型和tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dpo_model = AutoModelForCausalLM.from_pretrained(dpo_model_path).to(device)
dpo_tokenizer = AutoTokenizer.from_pretrained(dpo_model_path)

base_model = AutoModelForCausalLM.from_pretrained(base_model_path).to(device)
base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# 加载验证集
val_path = "/root/autodl-tmp/data/data/data/val.jsonl"
output_path = "/root/autodl-tmp/align-anything/eval_outputs_dpo/eval_dpo_vs_base.jsonl"

def generate_response(model, tokenizer, prompt):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512)
    outputs = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return response

with open(val_path, "r") as f, open(output_path, "w") as fout:
    for line in tqdm(f):
        item = json.loads(line)
        prompt = item["question"]
        base_response = generate_response(base_model, base_tokenizer, prompt)
        dpo_response = generate_response(dpo_model, dpo_tokenizer, prompt)
        tem = {}
        tem["question"] = item["question"]
        tem["reponse_1"] = base_response
        tem["reponse_2"] = dpo_response
        tem["overall_response"] = 1
        fout.write(json.dumps(tem, ensure_ascii=False) + "\n")