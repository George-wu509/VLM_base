"""Optional BPE corpus training for the RANDOM decoder track.

Never replace the tokenizer of a pretrained decoder: its embedding IDs would
lose their meaning. The main laptop track uses the existing SmolLM2 tokenizer.
"""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocab-size", type=int, default=8192)
    args = parser.parse_args()
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train(args.text, trainers.BpeTrainer(vocab_size=args.vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"]))
    out = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>",
                                 bos_token="<bos>", eos_token="<eos>", unk_token="<unk>")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    out.save_pretrained(args.output)

if __name__ == "__main__":
    main()
