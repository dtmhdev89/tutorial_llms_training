import os
from tutorial_llms_training.services.gpt2.pretrain import Pretrain
from tutorial_llms_training.services.gpt2.inference import Inference
import argparse


if __name__ == "__main__":

    sys_parser = argparse.ArgumentParser(
        description="Options to run training"
    )

    sys_parser.add_argument(
        "--train",
        action="store_true",
        help="Enable training mode"
    )

    sys_parser.add_argument(
        "--inference",
        action="store_true",
        help="Enable inference mode"
    )

    sys_parser.add_argument(
        "--model-name",
        type=str,
        help="Specify model name for inference"
    )

    args = sys_parser.parse_args()

    if args.train:
        os.environ["WANDB_DISABLED"] = "true"
        Pretrain.perform()

    if args.inference:
        if args.model_name:
            inference = Inference(
                model_name=args.model_name
            )
            tokenizer = inference.tokenizer
            model = inference.model

            decoded_ouput, encoded_output = inference.prompt(
                "I go to"
            )

            print(decoded_ouput)

            perplexity = inference.evaluate_with_perplexity(
                encoded_output=encoded_output
            )

            print(f"Perplexity: {perplexity}")

        else:
            print("Plese specify a model name from Hugging Face")
