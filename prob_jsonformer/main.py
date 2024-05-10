from typing import List, Set, Union, Dict, Any

from prob_jsonformer.logits_processors import (
    NumberStoppingCriteria,
    OutputNumbersTokens,
    IntegerStoppingCriteria,
    OutputIntegersTokens,
    StringStoppingCriteria,
)
from prob_jsonformer.choice_tree import choice_tree
from prob_jsonformer.type_prefixes import get_prefix_tokens_for_types

from termcolor import cprint
from transformers import PreTrainedModel, PreTrainedTokenizer
import json
import torch

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
                raise ValueError("Failed to generate a valid number")

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

        input_tensor = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.forward(input_tensor.to(self.model.device))
        logits = output.logits[0, -1]

        true_token_id = self.tokenizer.encode("true", return_tensors="pt")[0, 0]
        false_token_id = self.tokenizer.encode("false", return_tensors="pt")[0, 0]

        if prob:
            result = dict(true=logits[true_token_id], false=logits[false_token_id])
        else:
            result = logits[true_token_id] > logits[false_token_id]

        self.debug("[generate_boolean]", result)

        return result.item()

    def generate_string(self) -> str:
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
                StringStoppingCriteria(self.tokenizer, len(input_tokens[0]))
            ],
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Some models output the prompt as part of the response
        # This removes the prompt from the response if it is present
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

    def generate_choice_probs(self, choices) -> str:
        prompt = self.get_prompt() + '"'
        self.debug("[generate_string_prob]", prompt, is_prompt=True)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.model.device
        )[0]
        choices_tokens = self.tokenizer(choices).input_ids
        choices_tokens = [torch.tensor(c) for c in choices_tokens]

        r = list(choice_tree(self.model, self.tokenizer, input_ids, choices_tokens))
        return r  # json.dumps(r)

    def generate_enum(self, enum_values: Set[str]) -> str:
        prompt = self.get_prompt()
        self.debug("[generate_enum]", prompt, is_prompt=True)

        # These are necessary because we don't know if we're at the end or middle of an object/array
        terminal_tokens = torch.concat([
            self.tokenizer.encode(s, add_special_tokens=False, return_tensors="pt")[:, 0]
            for s in ('", "', '"}', '"]')
        ])

        highest_probability = 0.0
        best_option = None
        for option in enum_values:
            n_option_tokens = self.tokenizer.encode(f'"{option}', add_special_tokens=False, return_tensors="pt").shape[1]
            prompt_tokens = self.tokenizer.encode(prompt + f'"{option}', return_tensors="pt")
            option_tokens = prompt_tokens[0, -n_option_tokens:]

            with torch.no_grad():
                logits = self.model.forward(prompt_tokens.to(self.model.device)).logits[0, -n_option_tokens-1:]
            probabilities = torch.softmax(logits, dim=1)
            option_token_probabilities = probabilities[:-1][torch.arange(probabilities.shape[0]-1), option_tokens]
            termination_probability = torch.max(probabilities[-1, terminal_tokens])
            option_probability = torch.prod(option_token_probabilities) * termination_probability

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
        input_tensor = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.forward(input_tensor.to(self.model.device))
        logits = output.logits[0, -1]

        max_type = None
        max_logit = -float("inf")
        for possible_type in possible_types:
            try: 
                prefix_tokens = self.type_prefix_tokens[possible_type]
            except KeyError:
                raise ValueError(f"Unsupported schema type: {possible_type}")
            max_type_logit = logits[prefix_tokens].max()
            if max_type_logit > max_logit:
                max_type = possible_type
                max_logit = max_type_logit

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
            return self.generate_string()
        elif schema_type == "choice_probs":
            if key:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)
            return self.generate_choice_probs(schema["enum"])
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

    def generate_array(self, item_schema: Dict[str, Any], obj: Dict[str, Any]) -> list:
        for _ in range(self.max_array_length):
            # forces array to have at least one element
            element = self.generate_value(item_schema, obj)
            obj[-1] = element

            obj.append(self.generation_marker)
            input_prompt = self.get_prompt()
            obj.pop()
            input_tensor = self.tokenizer.encode(input_prompt, return_tensors="pt")
            output = self.model.forward(input_tensor.to(self.model.device))
            logits = output.logits[0, -1]

            top_indices = logits.topk(30).indices
            sorted_token_ids = top_indices[logits[top_indices].argsort(descending=True)]

            found_comma = False
            found_close_bracket = False

            for token_id in sorted_token_ids:
                decoded_token = self.tokenizer.decode(token_id)
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
        template = """{prompt}\nOutput result in the following JSON schema format:\n{schema}\nResult: {progress}"""
        progress = json.dumps(self.value)
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
