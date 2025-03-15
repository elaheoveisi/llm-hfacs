from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI


def get_response(
    gpt_model: str, context: str, prompt_template: str, model_type="ollama"
):
    """
    Generate a response to a given question based on the provided document."
    """
    try:
        if model_type == "ollama":
            llm = Ollama(
                model=gpt_model,  # Or your desired model
                base_url="http://10.203.13.225:11434",  # Replace with the remote host's IP address
                request_timeout=500,
                num_thread=20,
                num_gpu=2,
            )
        else:
            llm = OpenAI(model=gpt_model, temperature=1.0)

        # Create a prompt template for unstructured markdown output
        prompt_template = PromptTemplate(f"{prompt_template}")
        prompt = prompt_template.format(context=context)

        # Get the response from the model
        return llm.complete(prompt=prompt)
    except Exception as e:
        # Handle any errors that may occur during context generation
        raise RuntimeError(f"Error during context generation: {str(e)}")
