from typing import Callable, Dict, List, Any

def format_for_tokenization(tokenizer) -> Callable:
    """Default formatting function that just passes through the text"""
    def format_func(examples: Dict[str, Any]) -> Dict[str, List[str]]:
        return {"text": examples["text"] if "text" in examples else examples}
    return format_func

def comparison_format(tokenizer) -> Callable:
    """Format function for comparison-style datasets"""
    def format_func(examples: Dict[str, Any]) -> Dict[str, List[str]]:   
        texts = []
        for i in range(len(examples['prompt'])):
            texts.append(
                tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": examples['prompt'],
                        },
                        {
                            "role": "assistant",
                            "content": examples['response'],
                        },
                    ],
                    tokenize=False
                )
            )
        return {"text": texts}
    return format_func

def sharegpt_format(tokenizer) -> Callable:
    """Format function for ShareGPT-style datasets"""
    def format_func(examples: Dict[str, Any]) -> Dict[str, List[str]]:
        texts = []
        for conv in examples['conversations']:
            messages = []
            if isinstance(conv, list):
                for message in conv:
                    if message.get('from') == 'human':
                        messages.append({"role": "user", "content": message.get('value', '')})
                    elif message.get('from') == 'gpt':
                        messages.append({"role": "assistant", "content": message.get('value', '')})
                    elif message.get('from') == 'system':
                        messages.insert(0, {"role": "system", "content": message.get('value', '')})
            
            if not any(msg.get('role') == 'system' for msg in messages):
                messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})
            
            texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
        return {"text": texts}
    return format_func
