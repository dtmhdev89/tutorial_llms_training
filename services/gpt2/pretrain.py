import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.environ.get("ENV_FILE_PATH"))
from tutorial_llms_training.services.gpt2.data_loader import DataLoader
from tutorial_llms_training.services.gpt2.tokenizer_builder import TokenizerBuilder
from tutorial_llms_training.models.gpt2.lm_head_model import LmHeadModel
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling


class Pretrain:
    """Pretrain GPT2"""

    @staticmethod
    def perform():
        """Perform pretrain"""

        data_loader = DataLoader()
        dataset = data_loader.split_train_test()
        print(dataset)

        vocab_path = TokenizerBuilder.bpe_tokenizer(
            dataset=dataset
        )

        tokenizer, _ = TokenizerBuilder.pretrain_tokenizer(vocab_path)

        tokenized_ds = TokenizerBuilder.tokenize(
            tokenizer=tokenizer,
            dataset=dataset
        )

        print(tokenized_ds)

        lm_ds = tokenized_ds.map(
            TokenizerBuilder.group_texts,
            batched=True,
            num_proc=20
        )

        print(lm_ds)

        model = LmHeadModel(
            vocab_size=tokenizer.vocab_size,
            n_positions=512,
            n_ctx=512,
            n_embd=512,
            n_layer=6,
            n_head=8,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        ).model

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        training_args = TrainingArguments(
            output_dir="gpt-small-c4",
            logging_dir="logs",
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=20,
            eval_strategy="steps",
            save_strategy="steps",
            logging_strategy="steps",
            eval_steps=1000,
            save_steps=1000,
            logging_steps=1000,
            save_total_limit=1,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            load_best_model_at_end=True,
            fp16=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=lm_ds["train"],
            eval_dataset=lm_ds["test"],
            processing_class=tokenizer,
            data_collator=data_collator
        )

        trainer.train()

        trainer.push_to_hub(
            commit_message="Training complete",
            token=os.environ.get("HF_TOKEN")
        )
