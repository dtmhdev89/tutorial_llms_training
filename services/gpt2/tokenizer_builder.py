import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFKC
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import PreTrainedTokenizerFast


class TokenizerBuilder:
    """Build tokenizer for training"""

    VOCAB_SIZE = {
        "gpt2": 50257
    }

    SPECIAL_TOKEN = {
        "gpt2": ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    }

    SPECIAL_TOKEN_MAP = {
        "gpt2": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "mask_token": "<mask>",
        }
    }

    BLOCK_SIZE = 512

    @staticmethod
    def bpe_tokenizer(dataset, vocab_size=None):
        """Byte Pair Encoding Tokenizer"""

        if vocab_size is None:
            vocab_size = TokenizerBuilder.VOCAB_SIZE.get("gpt2")

        save_path = os.path.join("data_sources", "gpt_tokenizer.json")
        if os.path.exists(save_path):
            print("Already had the vocab file gpt_tokenizer.json")
            return save_path

        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = ByteLevel()
        tokenizer.normalizer = NFKC()
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=TokenizerBuilder.SPECIAL_TOKEN.get("gpt2")
        )

        tokenizer.train_from_iterator(
            dataset["train"]["text"],
            trainer
        )

        tokenizer.save(save_path)

        return save_path

    @staticmethod
    def pretrain_tokenizer(vocab_path):
        """Pretrain Tokenizer"""

        if os.environ.get("TOKENIZER_SAVED_PATH", None):
            save_path = os.environ.get("TOKENIZER_SAVED_PATH")
        else:
            save_path = os.path.join("data_sources", "gpt-tokenizer")
        
        print("----pretrain_tokenizer: Check if have saved data at: ", save_path)

        if os.path.exists(os.path.join(save_path, "tokenizer.json")):
            print(f"Loading tokenizer from {save_path}")
            tokenizer = PreTrainedTokenizerFast.from_pretrained(save_path)

            return tokenizer, save_path

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=vocab_path
        )

        tokenizer.add_special_tokens(
            TokenizerBuilder.SPECIAL_TOKEN_MAP.get("gpt2")
        )

        tokenizer.save_pretrained(save_path)

        return tokenizer, save_path

    @staticmethod
    def tokenize(tokenizer, dataset):
        """Perform tokenization on dataset"""

        def tokenize_example(example):
            """Perform tokenization on each example"""

            return tokenizer(example["text"])

        tokenizer_ds = dataset.map(
            tokenize_example,
            remove_columns=["text"],
            batched=True,
            num_proc=20
        )
        # output's format: ['input_ids', 'token_type_ids', 'attention_mask']
        return tokenizer_ds

    @staticmethod
    def group_texts(examples, block_size=None):
        if block_size is None:
            block_size = TokenizerBuilder.BLOCK_SIZE

        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size

        result = {
            k: [
                concatenated[k][i: i + block_size]
                for i in range(0, total_length, block_size)
            ]
            for k in concatenated
        }

        # since gpt model we load in train has a filter on input_ids and labels
        # # so that we don't need to filter add the moment
        # # consider this if using any other models
        result["labels"] = result["input_ids"].copy()

        return result
