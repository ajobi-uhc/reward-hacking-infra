from services.finetuning.data_models import UnslothFinetuningJob
from services.llm.data_models import Model

cfg = UnslothFinetuningJob(
    seed=42,
    source_model=Model(id="Qwen/Qwen3-8B", type="open_source"),
    hf_model_name="ajobi882/qwen-8b-reward-hacks",  # Change this to your HuggingFace username
    max_dataset_size=None,
    peft_cfg=UnslothFinetuningJob.PeftCfg(
        r=16,
        lora_alpha=16,
    ),
    train_cfg=UnslothFinetuningJob.TrainCfg(
        n_epochs=3,
        max_seq_length=2048,
        lr=2e-4,
        lr_scheduler_type="linear",
        warmup_steps=10,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_grad_norm=0.3,
    ),
)