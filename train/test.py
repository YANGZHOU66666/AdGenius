from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

dataset = load_dataset("trl-lib/Capybara", split="train")

# 打印数据集
print(dataset[0])

training_args = SFTConfig(
    output_dir="models/qwen2.5-0.5b-sft",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    logging_steps=20,
    learning_rate=2e-5,
    report_to="swanlab", 
    )

trainer = SFTTrainer(
    args=training_args,
    model="models/Qwen2.5-0.5B-Instruct",
    train_dataset=dataset,
)

trainer.train()