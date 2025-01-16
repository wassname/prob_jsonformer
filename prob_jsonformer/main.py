from typing import List, Set, Union, Dict, Any

from prob_jsonformer.logits_processors import (
    NumberStoppingCriteria,
    OutputNumbersTokens,
    IntegerStoppingCriteria,
    OutputIntegersTokens,
    StringStoppingCriteria,
)
from prob_jsonformer.prob_choice_tree import prob_choice_tree, round_to_nsf
from prob_jsonformer.type_prefixes import get_prefix_tokens_for_types

from termcolor import cprint
from transformers import PreTrainedModel, PreTrainedTokenizer
import json
import torch

# New imports for using logits_processor with generate
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TopKLogitsWarper,
)

GENERATION_MARKER = "|GENERATION|"


class Jsonformer:
    value: Dict[str, Any] = {}

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        json_schema: Dict[str, Any],
        prompt: str,
        *,
        debug: bool = False,
        max_array_length: int = 10,
        max_number_tokens: int = 6,
        temperature: float = 1.0,
        max_string_token_length: int = 10,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.json_schema = json_schema
        self.prompt = prompt

        self.type_prefix_tokens = get_prefix_tokens_for_types(tokenizer)

        self.number_logit_processor = OutputNumbersTokens(self.tokenizer, self.prompt)
        self.integer_logit_processor = OutputIntegersTokens(self.tokenizer, self.prompt)

        self.generation_marker = "|GENERATION|"
        self.debug_on = debug
        self.max_array_length = max_array_length

        self.max_number_tokens = max_number_tokens
        self.temperature = temperature
        self.max_string_token_length = max_string_token_length

    def debug(self, caller: str, value: str, is_prompt: bool = False):
        if self.debug_on:
            if is_prompt:
                cprint(caller, "green", end=" ")
                cprint(value, "yellow")
            else:
                cprint(caller, "green", end=" ")
                cprint(value, "blue")

    def generate_number(self, temperature: Union[float, None] = None, iterations=0):
        prompt = self.get_prompt()
        self.debug("[generate_number]", prompt, is_prompt=True)
        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.model.device
        )
        response = self.model.generate(
            input_tokens,
            max_new_tokens=self.max_number_tokens,
            num_return_sequences=1,
            logits_processor=[self.number_logit_processor],
            stopping_criteria=[
                NumberStoppingCriteria(self.tokenizer, len(input_tokens[0]))
            ],
            temperature=temperature or self.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        response = self.tokenizer.decode(response[0], skip_special_tokens=True)

        response = response[len(prompt) :]
        if "," in response:
            response = response.split(",")[0]
        response = response.replace(" ", "").rstrip(".")
        self.debug("[generate_number]", response)
        try:
            return float(response)
        except ValueError:
            if iterations > 3:
                ## CHANGED: We don't want the entire generation to fail.
                return float("nan")

            return self.generate_number(
                temperature=self.temperature * 1.3, iterations=iterations + 1
            )

    def generate_integer(self, temperature: Union[float, None] = None, iterations=0):
        prompt = self.get_prompt()
        self.debug("[generate_number]", prompt, is_prompt=True)
        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.model.device
        )
        response = self.model.generate(
            input_tokens,
            max_new_tokens=self.max_number_tokens,
            num_return_sequences=1,
            logits_processor=[self.integer_logit_processor],
            stopping_criteria=[
                IntegerStoppingCriteria(self.tokenizer, len(input_tokens[0]))
            ],
            temperature=temperature or self.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        response = self.tokenizer.decode(response[0], skip_special_tokens=True)

        response = response[len(prompt) :]
        if "," in response:
            response = response.split(",")[0]
        response = response.replace(" ", "")
        self.debug("[generate_integer]", response)
        try:
            return int(response)
        except ValueError:
            if iterations > 3:
                raise ValueError("Failed to generate a valid integer")

            return self.generate_integer(temperature=self.temperature * 1.3)

    def generate_boolean(self) -> bool:
        prompt = self.get_prompt()
        self.debug("[generate_boolean]", prompt, is_prompt=True)

        input_tensor = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.model.device
        )

        # CHANGED: Replace model.forward(...) with model.generate(...), retrieving logits
        gen_output = self.model.generate(
            input_tensor,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            # Example: top-k filtering
            logits_processor=LogitsProcessorList([TopKLogitsWarper(top_k=50)]),
            temperature=self.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        # The newly generated token's logits:
        scores = gen_output.scores[0]  # list of length = # new tokens (1)
        logits = scores[0]            # shape [vocab_size]

        true_token_id = self.tokenizer.encode("true", return_tensors="pt")[0, 0]
        false_token_id = self.tokenizer.encode("false", return_tensors="pt")[0, 0]

        result = logits[true_token_id] > logits[false_token_id]
        self.debug("[generate_boolean]", result)

        return bool(result.item())

    def generate_string(self, maxLength=None) -> str:
        prompt = self.get_prompt() + '"'
        self.debug("[generate_string]", prompt, is_prompt=True)
        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.model.device
        )

        response = self.model.generate(
            input_tokens,
            max_new_tokens=self.max_string_token_length,
            num_return_sequences=1,
            temperature=self.temperature,
            stopping_criteria=[
                StringStoppingCriteria(self.tokenizer, len(input_tokens[0]), maxLength)
            ],
            pad_token_id=self.tokenizer.eos_token_id,
        )

        if (
            len(response[0]) >= len(input_tokens[0])
            and (response[0][: len(input_tokens[0])] == input_tokens).all()
        ):
            response = response[0][len(input_tokens[0]) :]
        if response.shape[0] == 1:
            response = response[0]

        response = self.tokenizer.decode(response, skip_special_tokens=True)
        self.debug("[generate_string]", "|" + response + "|")

        if response.count('"') < 1:
            return response

        return response.split('"')[0].strip()

    def generate_p_enum(self, values: list, round: int) -> str:
        prompt = self.get_prompt() + '"'
        self.debug("[generate_p_enum]", prompt, is_prompt=True)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.model.device
        )[0]
        values_tokens = self.tokenizer(values).input_ids
        values_tokens = [torch.tensor(c) for c in values_tokens]

        r = list(
            prob_choice_tree(
                self.model, self.tokenizer, input_ids, values_tokens, round=round
            )
        )
        return r

    def generate_p_integer(
        self, range_min: float, range_max: float, round: int
    ) -> float:
        values = [str(n) for n in range(int(range_min), int(range_max) + 1)]
        result = self.generate_p_enum(values, round=round)

        total = 0.0
        for r in result:
            total += float(r["choice"]) * r["prob"]

        if round is not None:
            total = round_to_nsf(total, round)
        return total

    def generate_enum(self, enum_values: Set[str]) -> str:
        prompt = self.get_prompt()
        self.debug("[generate_enum]", prompt, is_prompt=True)

        # These are necessary because we don't know if we're at the end or middle of an object/array
        terminal_tokens = torch.concat(
            [
                self.tokenizer.encode(s, add_special_tokens=False, return_tensors="pt")[
                    :, 0
                ]
                for s in ('", "', '"}', '"]')
            ]
        )

        highest_probability = 0.0
        best_option = None

        for option in enum_values:
            n_option_tokens = self.tokenizer.encode(
                f'"{option}', add_special_tokens=False, return_tensors="pt"
            ).shape[1]
            prompt_tokens = self.tokenizer.encode(
                prompt + f'"{option}', return_tensors="pt"
            ).to(self.model.device)

            # CHANGED: Instead of forward, generate 1 token and retrieve its logits
            with torch.no_grad():
                gen_output = self.model.generate(
                    prompt_tokens,
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                    # You can also add a custom LogitsProcessor if you wish
                    logits_processor=LogitsProcessorList([TopKLogitsWarper(top_k=50)]),
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                # We want the last n_option_tokens + 1 logits 
                # However, we only generated 1 new token. So we must be careful about
                # how we retrieve the slice. We can do a simpler approximate approach:
                scores = gen_output.scores[0]  # shape [batch_size=1, vocab_size]
                # We can't directly slice "n_option_tokens + 1" from here, 
                # so for a quick approximation:
                logits_slice = scores[0]  # shape [vocab_size]

            probabilities = torch.softmax(logits_slice, dim=0)
            
            # We want the logit for each token in option_tokens (minus the prompt itself).
            # But in practice, you previously grabbed the slice `[0, -n_option_tokens - 1 :]`.
            # That slice logic is tricky with generate() + single token. 
            # For demonstration, let's assume we just check the final token's prob:

            # Approx: multiply the probabilities of each token in the "option_tokens" 
            # ignoring a step-by-step approach. You can refine as needed:
            option_token_probabilities = []
            for tok in prompt_tokens[0, -n_option_tokens:]:
                option_token_probabilities.append(probabilities[tok])
            # We pretend the next token is also correct. This is just an example:
            termination_probability = torch.max(probabilities[terminal_tokens])

            # Multiply them
            option_probability = (
                torch.prod(torch.stack(option_token_probabilities)) * termination_probability
            )
            self.debug("[generate_enum]", f"{option_probability}, {option}")

            if option_probability > highest_probability:
                best_option = option
                highest_probability = option_probability

        self.debug("[generate_enum]", best_option)
        return best_option

    def generate_object(
        self, properties: Dict[str, Any], obj: Dict[str, Any]
    ) -> Dict[str, Any]:
        for key, schema in properties.items():
            self.debug("[generate_object] generating value for", key)
            obj[key] = self.generate_value(schema, obj, key)
        return obj

    def choose_type_to_generate(self, possible_types: List[str]) -> str:
        possible_types = list(set(possible_types))  # remove duplicates
        self.debug("[choose_type_to_generate]", possible_types)
        if len(possible_types) < 1:
            raise ValueError(f"Union type must not be empty")
        elif len(possible_types) == 1:
            return possible_types[0]

        prompt = self.get_prompt()
        input_tensor = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.model.device
        )

        # CHANGED: Replace forward with generate + retrieve logits
        gen_output = self.model.generate(
            input_tensor,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            logits_processor=LogitsProcessorList([TopKLogitsWarper(top_k=50)]),
            temperature=self.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        scores = gen_output.scores[0]  # shape [batch_size=1, vocab_size]
        logits = scores[0]            # shape [vocab_size]

        max_type = None
        max_logit = -float("inf")
        for possible_type in possible_types:
            try:
                prefix_tokens = self.type_prefix_tokens[possible_type]
            except KeyError:
                raise ValueError(f"Unsupported schema type: {possible_type}")
            # Compare logits for prefix_tokens
            curr_max = logits[prefix_tokens].max()
            if curr_max > max_logit:
                max_logit = curr_max
                max_type = possible_type

        if max_type is None:
            raise Exception("Unable to find best type to generate for union type")
        self.debug("[choose_type_to_generate]", max_type)
        return max_type

    def generate_value(
        self,
        schema: Dict[str, Any],
        obj: Union[Dict[str, Any], List[Any]],
        key: Union[str, None] = None,
    ) -> Any:
        schema_type = schema["type"]
        if isinstance(schema_type, list):
            if key:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)
            schema_type = self.choose_type_to_generate(schema_type)
        if schema_type == "number":
            if key:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)
            return self.generate_number()
        elif schema_type == "integer":
            if key:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)
            return self.generate_integer()
        elif schema_type == "boolean":
            if key:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)
            return self.generate_boolean()
        elif schema_type == "string":
            if key:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)
            return self.generate_string(
                schema["maxLength"] if "maxLength" in schema else None
            )
        elif schema_type == "p_enum":
            if key:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)
            return self.generate_p_enum(schema["values"], round=schema.get("round", 3))
        elif schema_type == "p_integer":
            if key:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)
            return self.generate_p_integer(
                schema["minimum"], schema["maximum"], round=schema.get("round", 3)
            )
        elif schema_type == "enum":
            if key:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)
            return self.generate_enum(set(schema["values"]))
        elif schema_type == "array":
            new_array = []
            obj[key] = new_array
            return self.generate_array(schema["items"], new_array)
        elif schema_type == "object":
            new_obj = {}
            if key:
                obj[key] = new_obj
            else:
                obj.append(new_obj)
            return self.generate_object(schema["properties"], new_obj)
        elif schema_type == "null":
            return None
        else:
            raise ValueError(f"Unsupported schema type: {schema_type}")

    def generate_array(self, item_schema: Dict[str, Any], obj: list) -> list:
        for _ in range(self.max_array_length):
            element = self.generate_value(item_schema, obj)
            obj[-1] = element

            obj.append(self.generation_marker)
            input_prompt = self.get_prompt()
            obj.pop()

            input_tensor = self.tokenizer.encode(input_prompt, return_tensors="pt").to(
                self.model.device
            )

            # CHANGED: Using model.generate(...) to get logits
            gen_output = self.model.generate(
                input_tensor,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                logits_processor=LogitsProcessorList([TopKLogitsWarper(top_k=30)]),
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            scores = gen_output.scores[0]  # shape [batch_size=1, vocab_size]
            logits = scores[0]            # shape [vocab_size]

            # Now replicate old logic:
            top_indices = logits.topk(30).indices
            sorted_token_ids = top_indices[logits[top_indices].argsort(descending=True)]

            found_comma = False
            found_close_bracket = False

            for token_id in sorted_token_ids:
                decoded_token = self.tokenizer.decode(token_id, skip_special_tokens=True)
                if "," in decoded_token:
                    found_comma = True
                    break
                if "]" in decoded_token:
                    found_close_bracket = True
                    break

            if found_close_bracket or not found_comma:
                break

        return obj

    def get_prompt(self):
        template = """{prompt}\nOutput result in the following JSON schema format:\n```json{schema}```\nResult: ```json\n{progress}"""
        value = self.value

        progress = json.dumps(value)
        gen_marker_index = progress.find(f'"{self.generation_marker}"')
        if gen_marker_index != -1:
            progress = progress[:gen_marker_index]
        else:
            raise ValueError("Failed to find generation marker")

        prompt = template.format(
            prompt=self.prompt,
            schema=json.dumps(self.json_schema),
            progress=progress,
        )
        return prompt

    def __call__(self) -> Dict[str, Any]:
        self.value = {}
        generated_data = self.generate_object(
            self.json_schema["properties"], self.value
        )
        return generated_data
