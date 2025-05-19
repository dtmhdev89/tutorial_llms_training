from transformers import GPT2Config, GPT2LMHeadModel


class LmHeadModel:
    def __init__(
        self,
        vocab_size: int,
        n_positions: int,
        n_ctx: int,
        n_embd: int,
        n_layer: int,
        n_head: int,
        bos_token_id: int,
        eos_token_id: int
    ):
        self._config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_ctx=n_ctx,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id
        )

        self._model = GPT2LMHeadModel(self._config)

    @property
    def config(self):
        """config getter"""

        return self._config
    
    @property
    def model(self):
        """model getter"""

        return self._model
