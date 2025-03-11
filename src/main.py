import pandas as pd
import yaml
from tqdm import tqdm

from data.preprocess import clean_context
from data.readers import read_json
from models.llm import get_response
from utils import skip_run

# The configuration file
config_path = "configs/config.yaml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)


with skip_run("skip", "read_json_data") as check, check():
    data_path = "data/data.json"
    data = read_json(data_path, key="FactualNarrative")


with skip_run("skip", "input_output_llm_query") as check, check():
    data_path = "data/data.json"
    contexts = read_json(data_path, key="FactualNarrative")

    # Set the GPT model to use
    gpt_model = "qwen2.5:72b"

    io_prompts = yaml.load(open(str("prompts/io.yaml")), Loader=yaml.SafeLoader)
    output = dict()
    result_mapping = {"YES": 1, "NO": 0}
    for context in tqdm(contexts):
        for prompt in io_prompts:
            context = clean_context(context)
            result = get_response(gpt_model, context, io_prompts[prompt]).text
            if prompt in output.keys():
                output[prompt] += result_mapping[result]
            else:
                output[prompt] = result_mapping[result]

    # Save the dictionary
    pd.DataFrame.from_dict(data=output, orient="index").to_csv(
        "data/io_results.csv", header=False
    )


with skip_run("run", "input_output_expanded_llm_query") as check, check():
    data_path = "data/data.json"
    contexts = read_json(data_path, key="FactualNarrative")

    # Set the GPT model to use
    gpt_model = "qwen2.5:72b"

    io_prompts = yaml.load(
        open(str("prompts/io_expanded.yaml")), Loader=yaml.SafeLoader
    )
    io_prompts_base = [
        "organizational_influence",
        "supervisory_factors",
        "preconditions_for_unsafe_acts",
        "unsafe_acts",
    ]
    output = dict()
    result_mapping = {"YES": 1, "NO": 0}
    for context in contexts:
        for prompt in io_prompts_base:
            context = clean_context(context)
            result = get_response(gpt_model, context, io_prompts[prompt]).text

            if result in ["YES"]:
                prompt = prompt + "_detailed"
                result = get_response(gpt_model, context, io_prompts[prompt]).text

            # NOTE: Need to logic to save variable output
            if prompt in output.keys():
                try:
                    output[prompt] += result_mapping[result]
                except KeyError:
                    pass
            else:
                output[prompt] = result_mapping[result]


with skip_run("skip", "cot_llm_query") as check, check():
    data_path = "data/data.json"
    contexts = read_json(data_path, key="FactualNarrative")

    # Set the GPT model to use
    gpt_model = "qwen2.5:72b"

    io_prompts = yaml.load(open(str("prompts/io.yaml")), Loader=yaml.SafeLoader)
    output = dict()
    result_mapping = {"YES": 1, "NO": 0}
    for context in contexts:
        for prompt in io_prompts:
            context = clean_context(context)
            result = get_response(gpt_model, context, io_prompts[prompt])

            if result in ["YES"]:
                prompt = prompt + "_detailed"
                result = get_response(gpt_model, context, io_prompts[prompt]).text

            # NOTE: Need to logic to save variable output
            if prompt in output.keys():
                output[prompt] += result_mapping[result]
            else:
                output[prompt] = result_mapping[result]
