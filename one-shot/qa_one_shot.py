import re
import json
import argparse
import random
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from vllm import LLM, SamplingParams
import json
import random
from typing import List, Optional

# 全局定义系统提示
SYSTEM_PROMPT = (
"""You are an accurate table Q&A assistant. Please carefully analyze the following table and answer the questions accurately. Note: Only output the answers, do not explain the process.
"table_text":table content;
"statement": question.
Please give the answer directly.
"""
)

TITLE_PROMPT = """
    "Extract key terms from the table and question below. "
    "Return ONLY the most relevant keywords, separated by commas.\n"
"""

demonstration = """
Example:
Table_text:
[["Year", "Title", "Role", "Channel"],
["2015", "Kuch Toh Hai Tere Mere Darmiyaan", "Sanjana Kapoor", "Star Plus"],
["2016", "Kuch Rang Pyar Ke Aise Bhi", "Khushi", "Sony TV"],
["2016", "Gangaa", "Aashi Jhaa", "&TV"],
["2017", "Iss Pyaar Ko Kya Naam Doon 3", "Meghna Narayan Vashishth", "Star Plus"],
["2017–18", "Tu Aashiqui", "Richa Dhanrajgir", "Colors TV"],
["2019", "Laal Ishq", "Pernia", "&TV"],
["2019", "Vikram Betaal Ki Rahasya Gatha", "Rukmani/Kashi", "&TV"],
["2019", "Shaadi Ke Siyape", "Dua", "&TV"]]
Statement: What TV shows was Shagun Sharma seen in 2019?
Answer: Laal Ishq, Vikram Betaal Ki Rahasya Gatha, Shaadi Ke Siyape"""


def validate_model_path(model_path: str):
    """验证模型路径是否有效"""
    path = Path(model_path)
    if not path.exists():
        raise ValueError(f"模型路径不存在: {model_path}")


def process_think_output(generated_text: str):
# 已处理成QWen
    return None,generated_text.strip()


def calculate_qa_acc(data_item: Dict[str, Any], model_prediction: str) -> bool:
    """计算单个QA样本的准确率"""
    if model_prediction is None:
        return False
    ground_truth = data_item.get("answer", "").strip().lower()
    prediction_text = model_prediction.strip().lower()
    return ground_truth == prediction_text


import json
import random
from typing import List, Dict, Optional

def process_jsonl_file(
    file_path: str, 
    sample_size: int = 500, 
    random_seed: Optional[int] = None
) -> List[Dict[str, str]]:
    """
    处理JSONL文件，随机抽取指定数量的条目，返回包含问题、表格、ID和答案的字典列表
    
    参数:
        file_path (str): JSONL文件路径
        sample_size (int): 要抽取的条目数量，默认为100
        random_seed (Optional[int]): 随机种子，如果为None则不固定随机性
        
    返回:
        List[Dict[str, str]]: 包含question, table_str, ids和answer的字典列表
        [{
            'question': str,
            'table': str,
            'id': str,
            'answer': str
        }, ...]
    """
    # 读取所有数据
    with open(file_path, 'r', encoding='utf-8') as f:
        all_lines = [line.strip() for line in f if line.strip()]
    
    # 设置随机种子（如果提供）
    if random_seed is not None:
        random.seed(random_seed)
    
    # 随机抽样
    if len(all_lines) <= sample_size:
        sampled_lines = all_lines
    else:
        sampled_lines = random.sample(all_lines, sample_size)
    
    results = []
    for line in sampled_lines:
        data = json.loads(line)
        
        # 提取所需字段
        entry = {
            'question': data.get('statement', ''),
            'table': data.get('table_text',''),
            'id': data.get('ids', ''),
            'answer': ', '.join(data.get('answer', [])) if isinstance(data.get('answer', []), list) else str(data.get('answer', ''))
        }
        results.append(entry)
    
    return results


def save_results_to_json(results: List[Dict], model_name: str, output_file: str = None) -> str:
    """将评估结果保存为JSON文件"""
    if not output_file:
        model_short_name = os.path.basename(model_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"qa_results_{model_short_name}_{timestamp}.json"
    output_data = {
        "model": model_name,
        "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "system_prompt":SYSTEM_PROMPT,
        "total_samples": len(results),
        "correct_samples": sum(1 for r in results if r.get("is_correct")),
        "accuracy": sum(1 for r in results if r.get("is_correct")) / len(results) if results else 0,
        "results": results
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n评估结果已保存至: {output_file}")
    return output_file


def run_qa_evaluation(
    model_path: str,
    input_file: str,
    batch_size: int = 32,
    max_tokens: int = 256,
    temperature: float = 0.7,
    sample_size: int = None,
    random_seed: int = 42,
    output_file: str = None,
) -> float:
    """
    执行QA评估流程，始终使用聊天格式并内部加载系统提示
    """
    validate_model_path(model_path)
    
    random_data_list = process_jsonl_file(input_file)
    print(f"随机采样 {len(random_data_list)} 个样本进行评估")

    # 加载模型
    print(f"开始加载模型: {model_path}")
    llm = LLM(
        model=model_path,
        max_model_len=32768,
        dtype="half",
        gpu_memory_utilization=0.7,
        trust_remote_code=True
    )
    print("模型加载完成，使用系统提示:", SYSTEM_PROMPT)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.8,
        max_tokens=max_tokens
    )

    total_samples = len(random_data_list)
    total_correct = 0
    all_results: List[Dict[str, Any]] = []

    print(f"开始评估，共 {total_samples} 个样本")
    for idx, item in enumerate(random_data_list):
        print(f"\n处理样本 {idx+1}/{total_samples}...")
        
#         # title_prompt = TITLE_PROMPT + f"table_text: {str(item.get('table', ''))}\n" + f"statement: {item.get('question', '')}"
#         title_prompt = TITLE_PROMPT + f"table_text_header: {str(item.get('table', ''))[0]}\n" + f"statement: {item.get('question', '')}"
        
#         title_resp = llm.chat(
#             [[{"role": "system", "content": SYSTEM_PROMPT},
#               {"role": "user", "content": title_prompt}]],
#             sampling_params=sampling_params
#         )[0].outputs[0].text.strip()
        
        prompt = demonstration + '\n'
        prompt += f'Now Read the following table to answer the given question.'
        prompt += "table_text:\n " + str(item.get("table",""))
        prompt += '\n statement: ' + item.get("question","") + '\nAnswer:'
        
        # 构建聊天会话
        if len(prompt) > 32768:
            print(f"Warning: Instruction too long ({len(prompt)} tokens), truncating.")
            prompt = prompt[:32768]  # 截断为最大长度
        
        # 调用聊天接口（单个请求）
        final_resp = llm.chat(
            [[{"role": "system", "content": SYSTEM_PROMPT},
              {"role": "user", "content": prompt}]],
            sampling_params=sampling_params
        )[0].outputs[0].text.strip()

        correct = calculate_qa_acc(item, final_resp)
        if correct:
            total_correct += 1

        all_results.append({
            "id": idx + 1,
            "id_default": item.get("id",""),
            "instruction": prompt,
            "expected": item.get("answer", ""),
            "prediction": final_resp,
            "is_correct": correct
        })

        print(f"样本 {idx+1}: {'✓' if correct else '✗'} | 输出: {final_resp}")

    final_acc = total_correct / total_samples if total_samples else 0
    print(f"\n最终 ACC: {final_acc:.4f} ({total_correct}/{total_samples})")
    save_results_to_json(all_results, model_path, output_file)
    return final_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 vLLM 进行 QA 评估，使用内部系统提示")
    parser.add_argument("--model", type=str, required=True, help="本地模型路径")
    parser.add_argument("--input", type=str, required=True, help="QA 数据 JSON 文件路径")
    parser.add_argument("--batch_size", type=int, default=32, help="批处理大小")
    parser.add_argument("--max_tokens", type=int, default=256, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--sample_size", type=int, default=None, help="随机采样样本数量，不指定则使用全部数据")
    parser.add_argument("--random_seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", type=str, default=None, help="输出结果 JSON 文件路径，不指定则自动生成")
    args = parser.parse_args()
    
    run_qa_evaluation(
        model_path=args.model,
        input_file=args.input,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        sample_size=args.sample_size,
        random_seed=args.random_seed,
        output_file=args.output,
    )
