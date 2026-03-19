import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
import os
import pandas as pd
from openai import OpenAI
import re


# DEEPSEEK_API_KEY = "sk-fcc77f045c204c48bb33db5c068d28a9"
#
# # 初始化 DeepSeek 客户端
# client = OpenAI(
#     api_key=DEEPSEEK_API_KEY,  # 确保已在环境变量中配置 API key
#     base_url="https://api.deepseek.com"
# )


# QWen_API_KEY = "sk-933696e342b444168d5ce185a57b4e0c"
# client = OpenAI(
#     api_key=QWen_API_KEY,
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )
YUNWU_API_KEY = "sk-JolyMwHHVHueS5vZtLVaaWCNhTGfaheO5JMtX5YOaMCdnGro"

client = OpenAI(
    api_key = YUNWU_API_KEY,
    base_url="https://yunwu.ai/v1",
)

print("GPU 是否可", torch.cuda.is_available())
print("当前 GPU：", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无")

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)
# 路径
peft_model_id = r"multilabel_mistral"        # 当前目录下的文件夹
base_model_name = "/home/admin/llm_models/LLM-Research/Meta-Llama-3.1-8B"           # 当前目录下的文件夹

input_file = "Evaluation.csv"
output_file = "claude-sonnet-4-6-results-labeled.csv"



tokenizer = AutoTokenizer.from_pretrained(peft_model_id, padding_side="right")



# base_model = AutoModelForSequenceClassification.from_pretrained(
#     base_model_name,
#     num_labels=7,
#     torch_dtype=torch.float16,
#     device_map="cuda"
# )
#
#
#
# model = PeftModel.from_pretrained(
#     base_model,
#     peft_model_id,
#     device_map="cuda"
# )


# model.eval()

def build_pair_text(req_a, req_b):
    return f"Requirement 1: {req_a.strip()}\nRequirement 2: {req_b.strip()}"
def tokenize_for_inference(text, tokenizer, max_length=512):
    return tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )


def predict_conflict_labels(model, tokenizer, req_a, req_b, threshold=0.5):
    text = build_pair_text(req_a, req_b)
    inputs = tokenize_for_inference(text, tokenizer)

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # 多标签分类：sigmoid
    probs = torch.sigmoid(logits).cpu().numpy()[0]

    return probs
LABEL_MAP = {
    0: "Event Conflict",
    1: "Agent Conflict",
    2: "Operation Conflict",
    3: "Input Conflict",
    4: "Output Conflict",
    5: "Constraint Conflict",
    6: "Neutral"
}

def decode_labels(probs, threshold=0.5):
    labels = [
        LABEL_MAP[i]
        for i, p in enumerate(probs)
        if p >= threshold
    ]
    return labels




def build_extraction_prompt(requirement_text: str) -> str:
    return f"""
You are an expert in requirements engineering and semantic analysis.

Extract the semantic tuple from the following Functional Requirement (FR).

====================
Requirement:
\"\"\"{requirement_text}\"\"\"
====================

Identify the following six elements:

- Event: triggering condition or time clause (e.g., when, if). If none, output "null".
- Agent: performer of the operation. If implicit, output "unspecified agent".
- Operation: main action or behavior, usually linked to shall/should/will.
- Input: information or precondition required for the operation. If none, output "null".
- Output: result or effect of the operation. If none, output "null".
- Restriction: constraints on time, quantity, frequency, or resources. If none, output "null".

====================
Output Format (JSON only):

{{
  "Event": "...",
  "Agent": "...",
  "Operation": "...",
  "Input": "...",
  "Output": "...",
  "Restriction": "..."
}}
"""


def extract_semantic_tuple(client, requirement_text):
    prompt = build_extraction_prompt(requirement_text)

    response = client.chat.completions.create(
        # model="deepseek-chat",
        # model="qwen3-max",
        model="claude-sonnet-4-6",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant skilled in requirements engineering and natural language analysis."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.5,
        max_tokens=500,
    )

    return response.choices[0].message.content


# def build_resolution_prompt(req_a, req_b, conflict_label, tuple_a=None, tuple_b=None):
#     return f"""
# You are an expert in requirements conflict analysis.
#
# The following two requirements may have a conflict:
# - Requirement A: \"\"\"{req_a}\"\"\"
# - Requirement B: \"\"\"{req_b}\"\"\"
#
# Task:
# Provide 2–3 concise and actionable resolution suggestions to resolve the conflict between the two requirements.
# Output your suggestions as numbered items, each on a separate line. For example:
#
# 1. First suggestion...
# 2. Second suggestion...
# 3. Third suggestion...
# """

# 标签
def build_resolution_prompt(req_a, req_b, conflict_label, tuple_a=None, tuple_b=None):
    return f"""
You are an expert in requirements conflict analysis and resolution.

Two requirements have been identified as having a potential **{conflict_label}**.

Requirement A:
\"\"\"{req_a}\"\"\"

Requirement B:
\"\"\"{req_b}\"\"\"

Conflict Type Definition:
The conflict belongs to the category of "{conflict_label}".
Please focus specifically on resolving this type of conflict rather than providing general suggestions.

Task:
Provide 2–3 concise, specific, and actionable resolution suggestions
that directly address the {conflict_label} between Requirement A and Requirement B.

Each suggestion should:
- Clearly explain what modification, clarification, or coordination is needed.
- Be practical and implementable in a real system specification.

Output format:
1. First suggestion...
2. Second suggestion...
3. Third suggestion...
"""


# 标签+元组
# def build_resolution_prompt(req_a, req_b, conflict_label, tuple_a=None, tuple_b=None):
#
#
#     if isinstance(conflict_label, list):
#         conflict_label = ", ".join(conflict_label)
#
#     structure_section = ""
#
#     if tuple_a is not None and tuple_b is not None:
#         structure_section = f"""
# Structured Representation:
#
# Requirement A Structure:
# {tuple_a}
#
# Requirement B Structure:
# {tuple_b}
# """
#
#     return f"""
# You are an expert in requirements conflict analysis and resolution.
#
# Two requirements have been identified as having a **{conflict_label}**.
#
# Requirement A:
# \"\"\"{req_a}\"\"\"
#
# Requirement B:
# \"\"\"{req_b}\"\"\"
#
# {structure_section}
#
# Task:
# 1. Analyze the differences between the structured elements.
# 2. Focus specifically on resolving the {conflict_label}.
# 3. Propose 2–3 concise and actionable resolution suggestions.
#
# Each suggestion should:
# - Clearly indicate which structured element should be modified.
# - Suggest concrete modifications rather than vague advice.
#
# Output format:
# 1. ...
# 2. ...
# 3. ...
# """



def resolve_conflict(
    client, req_a, req_b, conflict_label, tuple_a, tuple_b
):
    prompt = build_resolution_prompt(
        req_a, req_b, conflict_label, tuple_a, tuple_b
    )


    response = client.chat.completions.create(
        # model="deepseek-chat",
        # model="qwen3-max",
        model="claude-sonnet-4-6",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant skilled in requirements engineering and conflict resolution."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.5,
        max_tokens=800,
    )

    return response.choices[0].message.content



df = pd.read_csv(input_file, encoding="utf-8")
# df = df.head(2)
for col in df.columns:
    print(col, len(col))
# ==============================
# 2️⃣ 定义无冲突标签集合
# ==============================
NO_CONFLICT_LABELS = {"No Conflict", "Neutral", "None"}

# ==============================
# 3️⃣ 创建结果存储列表
# ==============================
results = []

# ==============================
# 4️⃣ 遍历数据集逐条预测
# ==============================

import time
start_time = time.time()

for idx, row in df.iterrows():
    req_a = str(row["Original_FR"]).strip()
    req_b = str(row["Conflicting_FR"]).strip()
    sample_id = row["ID"]
    conflict_label = row["Label"]

    print("\n" + "=" * 80)
    print(f"Processing ID: {sample_id}")

    # # -------- Step 1: 冲突预测 --------
    # probs = predict_conflict_labels(
    #     model, tokenizer, req_a, req_b, threshold=0.5
    # )
    # conflict_label = decode_labels(probs, threshold=0.5)
    #
    # # -------- Step 2: 冲突判断 --------
    # if not conflict_label or all(label in NO_CONFLICT_LABELS for label in conflict_label):
    #     conclusion = "No Conflict"
    #     resolution = "No conflict detected"
    #     tuple_a = ""
    #     tuple_b = ""
    #     predicted_label = conflict_label
    # else:
    #     conclusion = "Conflict"


    # -------- Step 3: 语义元组抽取 --------
    # tuple_a = extract_semantic_tuple(client, req_a)
    # tuple_b = extract_semantic_tuple(client, req_b)


    tuple_a = ""
    tuple_b = ""
    # -------- Step 4: 冲突消解 --------
    print("conflict_label:",conflict_label)
    resolution = resolve_conflict(
        client, req_a, req_b, conflict_label, tuple_a, tuple_b
    )

    predicted_label = conflict_label

    # -------- Step 5: 保存结果 --------
    results.append({
        "ID": sample_id,
        "Original_FR": req_a,
        "Conflicting_FR": req_b,
        "Predicted_Label": predicted_label,
        # "Conclusion": conclusion,
        "Semantic_Tuple_A": tuple_a,
        "Semantic_Tuple_B": tuple_b,
        "Resolution": resolution
    })


end_time = time.time()
total_time = end_time - start_time

print("\n==============================")
print(f"Total Running Time: {total_time:.2f} seconds")
print("==============================")
# ==============================
# 6️⃣ 生成新表格
# ==============================
result_df = pd.DataFrame(results)

# ==============================
# 7️⃣ 保存为CSV文件
# ==============================
result_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("\n✅ 全部处理完成！")
print(f"结果已保存至: {output_file}")