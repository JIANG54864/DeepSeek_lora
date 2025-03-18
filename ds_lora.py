from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
import torch
from peft import LoraConfig, TaskType, get_peft_model

# 数据集预处理：将JSON格式转换为Pandas DataFrame，再转换为HuggingFace Dataset格式
df = pd.read_json('./dataset/huanhuan.json')
ds = Dataset.from_pandas(df)

# 初始化分词器并配置填充方向
tokenizer = AutoTokenizer.from_pretrained('./deepseek-ai/deepseek-llm-7b-chat/', use_fast=False, trust_remote_code=True)
tokenizer.padding_side = 'right'

def process_func(example):
    """
    处理单条样本的数据预处理函数，将指令和响应转换为模型输入格式

    参数：
    example : dict
        包含单个样本的字典，包含instruction/input/output三个键

    返回：
    dict
        包含处理后的input_ids, attention_mask和labels的字典
        其中labels将指令部分标记为-100（计算损失时忽略），只计算响应部分的损失
    """
    MAX_LENGTH = 384  # 根据中文分词特性设置的序列最大长度

    # 将用户指令和输入组合编码
    instruction = tokenizer(f"User: {example['instruction']+example['input']}\n\n", add_special_tokens=False)
    # 将助手响应编码
    response = tokenizer(f"Assistant: {example['output']}", add_special_tokens=False)

    # 拼接输入序列：指令 + 响应 + 填充符
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    # 注意力掩码：有效token为1，填充符为1（因eos需要参与计算）
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    # 标签设置：指令部分标记为-100（损失计算忽略），响应部分保留真实token
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    # 序列截断处理
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# 应用预处理函数到整个数据集，并移除原始列
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

# 加载预训练语言模型并配置生成参数
model = AutoModelForCausalLM.from_pretrained(
    './deepseek-ai/deepseek-llm-7b-chat/',
    trust_remote_code=True,
    torch_dtype=torch.half,
    device_map="auto"
)
# 配置生成参数：将填充token设为结束token
model.generation_config = GenerationConfig.from_pretrained('./deepseek-ai/deepseek-llm-7b-chat/')

# 设置填充标记与结束标记相同，避免生成过程中出现未知标记
model.generation_config.pad_token_id = model.generation_config.eos_token_id

# 启用梯度检查点特性（训练时节省显存）
model.enable_input_require_grads()

# 配置LoRA参数（低秩适配器参数）
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 因果语言模型任务类型
    target_modules=[  # 要注入LoRA适配器的模块列表
        "q_proj", "k_proj", "v_proj",  # 注意力机制投影层
        "o_proj", "gate_proj", "up_proj", "down_proj"  # FFN层
    ],
    inference_mode=False,  # 训练模式（False表示启用适配器）
    r=8,  # 低秩矩阵的秩，控制适配器复杂度
    lora_alpha=32,  # 缩放因子，影响适配器输出的权重
    lora_dropout=0.1  # 防止过拟合的dropout比例
)

# 将原始模型转换为参数高效微调模型
model = get_peft_model(model, config)

# 配置训练超参数
args = TrainingArguments(
    output_dir="./output/Ds_lora",  # 模型保存路径
    per_device_train_batch_size=8,  # 每个设备的批处理大小
    gradient_accumulation_steps=2,  # 梯度累积步数（模拟更大批次）
    logging_steps=10,  # 每10步记录一次日志
    num_train_epochs=3,  # 训练轮数
    save_steps=100,  # 每100步保存一次检查点
    learning_rate=1e-4,  # 学习率
    save_on_each_node=True,  # 分布式训练时在每个节点保存
    gradient_checkpointing=True  # 启用梯度检查点节省显存
)

# 初始化训练器
trainer = Trainer(
    model=model,  # 待训练模型
    args=args,  # 训练参数
    train_dataset=tokenized_id,  # 训练数据集
    data_collator=DataCollatorForSeq2Seq(  # 序列填充工具
        tokenizer=tokenizer,
        padding=True  # 启用动态填充
    ),
)

# 启动模型训练
trainer.train()

# 测试训练后的模型
text = "小姐，别的秀女都在求中选，唯有咱们小姐想被撂牌子，菩萨一定记得真真儿的——"

# 构建模型输入（添加对话格式）
inputs = tokenizer(f"User: {text}\n\n", return_tensors="pt")

# 生成文本（限制最大生成长度）
outputs = model.generate(
    **inputs.to(model.device),
    max_new_tokens=100  # 限制生成新token的数量
)

# 解码并打印生成结果（跳过特殊标记）
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
