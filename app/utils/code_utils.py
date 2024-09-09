import re

def extract_code_and_language(response: str) -> (str, str):
    language_match = re.search(r'Language:\s*(python|javascript)', response, re.IGNORECASE)
    code_block = re.search(r'```(python|javascript)\n(.*?)```', response, re.DOTALL)
    if language_match and code_block:
        language = language_match.group(1).strip().lower()
        code = code_block.group(2).strip()
        return language, code
    else:
        raise ValueError("No valid code block or language found in the response.")

def format_code(code: str) -> str:
    return f"\n{code}\n"