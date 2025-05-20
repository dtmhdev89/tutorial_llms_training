# tutorial_llms_training
[Tutorial] LLMs training from scratch

###

```bash
# to avoid unallocated memory on cuda
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -m application.pretrain_gpt2_with_small_dataset --train
```
