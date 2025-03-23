from typing import List
from transformers import PreTrainedTokenizer, StoppingCriteria
from transformers.generation.logits_process import LogitsProcessor
import torch


class StringStoppingCriteria(StoppingCriteria):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        prompt_length: int,
        max_length: int = None,
        min_length: int = 1,
    ):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.max_length = max_length
        self.min_length = min_length

    def __call__(
        self,
        input_ids: torch.LongTensor,
        _,
    ) -> bool:
        if len(input_ids[0]) <= self.prompt_length:
            return False

        last_token_id = input_ids[0][-1]
        last_token = self.tokenizer.decode(last_token_id, skip_special_tokens=True)

        result = '"' in last_token

        if self.max_length is not None:
            # because of tokens this wont work pefectly, we might go 0-10 chars over
            gen_ids = input_ids[0][self.prompt_length :]
            o = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            str_l = len(o)
            if str_l > self.max_length:
                return True

        if self.min_length is not None:
            gen_ids = input_ids[0][self.prompt_length :]
            o = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            str_l = len(o)
            if str_l < self.min_length:
                return False

        return result


class NumberStoppingCriteria(StoppingCriteria):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        prompt_length: int,
        precision: int = 3,
    ):
        self.tokenizer = tokenizer
        self.precision = precision
        self.prompt_length = prompt_length

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> bool:
        decoded = self.tokenizer.decode(
            input_ids[0][self.prompt_length :], skip_special_tokens=True
        )

        if decoded.count(".") > 1:
            return True

        if (
            decoded.count(".") == 1
            and len(decoded.replace(" ", "").split(".")[1]) > self.precision
        ):
            return True

        if (
            len(decoded) > 1
            and "," in decoded
            and any(c.isdigit() for c in decoded.split(",")[0])
        ):
            return True

        if (
            len(decoded) > 1
            and any(c.isdigit() for c in decoded)
            and ("," in decoded or decoded[-1] in (" ", "\n"))
        ):
            return True

        return False


class OutputNumbersTokens(LogitsProcessor):
    def __init__(self, tokenizer: PreTrainedTokenizer, prompt: str):
        self.tokenizer = tokenizer
        self.tokenized_prompt = tokenizer(prompt, return_tensors="pt")
        vocab_size = len(tokenizer)
        self.allowed_mask = torch.zeros(vocab_size, dtype=torch.bool)

        for _, token_id in tokenizer.get_vocab().items():
            token_str = tokenizer.decode(token_id, skip_special_tokens=True).strip()

            if (
                token_str == ""
                or (
                    all(c.isdigit() or c == "." for c in token_str)
                    and token_str.count(".") <= 1
                )
                or (
                    "," in token_str
                    and all(c.isdigit() or c == "." for c in token_str.split(",")[0])
                    and token_str.count(".") <= 1
                )
            ):
                self.allowed_mask[token_id] = True

    def __call__(self, _, scores):
        mask = self.allowed_mask.expand_as(scores)
        scores[~mask] = -float("inf")

        return scores


class IntegerStoppingCriteria(StoppingCriteria):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        prompt_length: int,
        max_digits: int = 15,
    ):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.max_digits = max_digits

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> bool:
        decoded = self.tokenizer.decode(
            input_ids[0][self.prompt_length :], skip_special_tokens=True
        )

        if len(decoded.strip()) > self.max_digits:
            return True

        if (
            len(decoded) > 1
            and "," in decoded
            and any(c.isdigit() for c in decoded.split(",")[0])
        ):
            return True

        if (
            len(decoded) > 1
            and any(c.isdigit() for c in decoded)
            and decoded[-1] in (" ", "\n")
        ):
            return True

        return False


class OutputIntegersTokens(LogitsProcessor):
    def __init__(self, tokenizer: PreTrainedTokenizer, prompt: str):
        self.tokenizer = tokenizer
        self.tokenized_prompt = tokenizer(prompt, return_tensors="pt")
        vocab_size = len(tokenizer)
        self.allowed_mask = torch.zeros(vocab_size, dtype=torch.bool)

        for _, token_id in tokenizer.get_vocab().items():
            token_str = tokenizer.decode(token_id, skip_special_tokens=True).strip()

            if (
                token_str == ""
                or all(c.isdigit() for c in token_str)
                or "," in token_str
                and all(c.isdigit() for c in token_str.split(",")[0])
            ):
                self.allowed_mask[token_id] = True

    def __call__(self, _, scores):
        mask = self.allowed_mask.expand_as(scores)
        scores[~mask] = -float("inf")

        return scores
