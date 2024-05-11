# prob_jsonformer: A Bulletproof Way to Generate Probabilistic Structured JSON from Language Models.

This fork has been modified to include the token probabilities. This is not complaint with json schema, but it can be useful for efficient extracting of a range of possible values.

I've also merged some of the recent PR's for enum, integer, null, union. They are not yet included in the upstream Jsonformer. You can see them all below in this example:


~~~
pip install git+https://github.com/wassname/prob_jsonformer.git
~~~

## Example

```python
from prob_jsonformer import Jsonformer
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "databricks/dolly-v2-3b"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

json_schema = {
    "type": "object",
    "properties": {
        # we can return the probability of each choice, even if they are multiple tokens
        "age_probs": {"type": "p_enum", "values": [str(s) for s in range(10, 20)]},
        # we can return the probabilistic weighted mean of a range
        "age_wmean": {"type": "p_integer", "minimum": 10, "maximum": 20},
        # the prob of true and false
        "is_student_probs": {"type": "p_enum", "values": ["true", "false"]},
        "is_student": {"type": "boolean"},
        # we've merged patches for enum, integer, null, union - currently mising from jsonformer
        "name": {"type": "string", "maxLength": 4},
        "age": {"type": "integer"},
        "unit_time": {"type": "number"},
        "courses": {"type": "array", "items": {"type": "string"}},
        "trim": {"type": ["string", "null"]},
        "color": {
            "type": "enum",
            "values": ["red", "green", "blue", "brown", "white", "black"],
        },
    },
}

prompt = "Generate a young person's information based on the following schema:"
jsonformer = Jsonformer(model, tokenizer, json_schema, prompt, temperature=0)
generated_data = jsonformer()

generated_data = {
    "age_probs": [
        {"prob": 0.62353515625, "choice": "10"},
        {"prob": 0.349609375, "choice": "12"},
        {"prob": 0.01123809814453125, "choice": "11"},
        {"prob": 0.00760650634765625, "choice": "16"},
        {"prob": 0.0025482177734375, "choice": "13"},
        {"prob": 0.0025081634521484375, "choice": "15"},
        {"prob": 0.0018062591552734375, "choice": "14"},
        {"prob": 0.00104522705078125, "choice": "18"},
        {"prob": 0.00011551380157470703, "choice": "17"},
        {"prob": 5.042552947998047e-05, "choice": "19"},
    ],
    "age_wmean": 15.544570922851562,
    "is_student_probs": [
        {"prob": 0.962890625, "choice": "true"},
        {"prob": 0.037322998046875, "choice": "false"},
    ],
    "is_student": False,
    "name": "John",
    "age": 17,
    "unit_time": 0.5,
    "courses": ["C++"],
    "trim": None,
    "color": "green",
}
```

 The original [README](https://github.com/1rgs/jsonformer) is includes below.

# ORIGINAL: Jsonformer: A Bulletproof Way to Generate Structured JSON from Language Models.

### Problem: Getting models to output structured JSON is hard

### Solution: Only generate the content tokens and fill in the fixed tokens

[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1rgs/jsonformer/blob/main/Jsonformer_example.ipynb)

![cover](img/cover4.png)

Generating structured JSON from language models is a challenging task. The
generated JSON must be syntactically correct, and it must conform to a schema
that specifies the structure of the JSON.

Current approaches to this problem are brittle and error-prone. They rely on prompt engineering, fine-tuning, and post-processing, but they still fail to generate syntactically correct JSON in many cases.

Jsonformer is a new approach to this problem. In structured data, many tokens are fixed and predictable. Jsonformer is a wrapper around Hugging Face models that fills in the fixed tokens during the generation process, and only delegates the generation of content tokens to the language model. This makes it more efficient and bulletproof than existing approaches.

This currently supports a subset of JSON Schema. Below is a list of the supported schema types:

- number
- boolean
- string
- array
- object

## Example

```python
from jsonformer import Jsonformer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b")
tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b")

json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"},
        "is_student": {"type": "boolean"},
        "courses": {
            "type": "array",
            "items": {"type": "string"}
        }
    }
}

prompt = "Generate a person's information based on the following schema:"
jsonformer = Jsonformer(model, tokenizer, json_schema, prompt)
generated_data = jsonformer()

print(generated_data)
```

### Jsonformer works on complex schemas, even with tiny models. Here is an example of a schema with nested objects and arrays, generated by a 3B parameter model.

```python
{"type": "object", "properties": {"car": {"type": "object", "properties": {"make": {"type": "string"}, "model": {"type": "string"}, "year": {"type": "number"}, "colors": {"type": "array", "items": {"type": "string"}}, "features": {"type": "object", "properties": {"audio": {"type": "object", "properties": {"brand": {"type": "string"}, "speakers": {"type": "number"}, "hasBluetooth": {"type": "boolean"}}}, "safety": {"type": "object", "properties": {"airbags": {"type": "number"}, "parkingSensors": {"type": "boolean"}, "laneAssist": {"type": "boolean"}}}, "performance": {"type": "object", "properties": {"engine": {"type": "string"}, "horsepower": {"type": "number"}, "topSpeed": {"type": "number"}}}}}}}, "owner": {"type": "object", "properties": {"firstName": {"type": "string"}, "lastName": {"type": "string"}, "age": {"type": "number"}}}}}
```

```python
{
  car: {
    make: "audi",
    model: "model A8",
    year: 2016.0,
    colors: [
      "blue"
    ],
    features: {
      audio: {
        brand: "sony",
        speakers: 2.0,
        hasBluetooth: True
      },
      safety: {
        airbags: 2.0,
        parkingSensors: True,
        laneAssist: True
      },
      performance: {
        engine: "4.0",
        horsepower: 220.0,
        topSpeed: 220.0
      }
    }
  },
  owner: {
    firstName: "John",
    lastName: "Doe",
    age: 40.0
  }
}
```

## Features

- Bulletproof JSON generation: Jsonformer ensures that the generated JSON is always syntactically correct and conforms to the specified schema.
- Efficiency: By generating only the content tokens and filling in the fixed tokens, Jsonformer is more efficient than generating a full JSON string and parsing it.
- Flexible and extendable: Jsonformer is built on top of the Hugging Face transformers library, making it compatible with any model that supports the Hugging Face interface.

## Installation

```bash
pip install jsonformer
```

## Development

[Poetry](https://python-poetry.org/docs/#installation) is used for dependency management.

```bash
poetry install
```

```bash
poetry run python -m jsonformer.example
```

## License

Jsonformer is released under the MIT License. You are free to use, modify, and distribute this software for any purpose, commercial or non-commercial, as long as the original copyright and license notice are included.
