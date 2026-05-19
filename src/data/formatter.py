from unsloth import FastLanguageModel

class Formatter:
    def __init__(self, model_name: str = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit") -> None:
        """
        Initializes the Formatter with a specified language model.
        
        Args:
            model_name (str): The name of the pre-trained language model to use for formatting.
        """
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            load_in_4bit=True,
        )

    def format(self, example):
        """Formats a single example into a structured prompt for evaluation. 
        Args:
            example (dict): A dictionary containing the fields 'task', 'reference_answer', 'answer
            and 'rubric' that describe the evaluation sample.
            
            Returns:
            dict: A dictionary with a single key 'prompt' containing the formatted prompt as a list of
            messages for the language model.
        """
        
        system_propmt = "You are an automated evaluation system."
        user_prompt = f"""
        Task : 
        {example['task']}
        
        Reference Answer:
        {example['reference_answer']}
        
        Student Answer:
        {example['answer']}
        
        Runbric:
        {example['rubric']}
        
         Evaluate the student answer and return:
            - score
            - reasoning
        """
        assistant_prompt = f"""
            {{
            "score": {example['score']},
            "reasoning": "{example['reasoning']}"
            }}
            """
        system = {"role": "system", "content": system_propmt}
        user = {"role": "user", "content": user_prompt}
        assistant = {"role": "assistant", "content": assistant_prompt}
        prompt = [system, user, assistant]

        return {"prompt": prompt}

    def reformat(self, dataset):
        """
        Reformats an entire dataset by applying the formatting to each example.
        Args:
            dataset (datasets.Dataset): A Hugging Face Dataset containing the evaluation samples to be reformatted.
        Returns:
            datasets.Dataset: A new Dataset where each example has been reformatted into a structured prompt."""
        
        reformatted = dataset.map(self.format, remove_columns=dataset.column_names)

        return reformatted

    def encode(self, example):
        """Encodes a single example's prompt into token IDs using the tokenizer.
        Args:
            example (dict): A dictionary containing the 'prompt' key with the formatted prompt to be
            tokenized.
        Returns:
            dict: A dictionary containing the tokenized input IDs and attention mask for the prompt.
        """
        prompt = (
            example["prompt"] if "prompt" in example else self.format(example)["prompt"]
        )
        encoded = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=False,
        )

        return self.tokenizer(encoded, truncation=True)

    def tokenize(self, dataset):
        """Tokenizes an entire dataset by applying the encoding to each example.
        Args:
            dataset (datasets.Dataset): A Hugging Face Dataset containing the evaluation samples to be tokenized
        Returns:
            datasets.Dataset: A new Dataset where each example has been tokenized into input IDs and attention masks.
            """
        return dataset.map(self.encode, remove_columns=dataset.column_names)
