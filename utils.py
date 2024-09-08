import re

def extract_code(response: str) -> str:
    code_block = re.search(r'```python\n(.*?)```', response, re.DOTALL)
    if code_block:
        return code_block.group(1).strip()
    else:
        raise ValueError("No Python code block found in the response.")

def format_code(code: str) -> str:
    return f"""
{code}
"""