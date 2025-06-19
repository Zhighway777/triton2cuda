from zhipuai import ZhipuAI
from prompt import get_full_prompt, get_simple_prompt

# main_prompt = '''Convert the following triton code to equivalent CUDA code.
# ```python

# '''
# suffix_instruction ='''
# ```
# You must ensure that the output code is fully functional and can be run. Your code must provide a class named ModelNew that inherits from torch.nn.Module and has a forward method that takes in the same input tensors as the `Model` class in the triton kernel code and returns the same output tensors.
# '''

# def get_full_prompt(triton_code):
#     return main_prompt + triton_code + suffix_instruction



def triton2cuda(triton_code):
    #API Key, plantform: zhipuai, name:llm_compiler_api
    client = ZhipuAI(api_key="5bf98ea765f642aeb720420e522592f7.DWMrwJ2rfsWPYhHJ") 
    response = client.chat.completions.create(
        model="glm-4-plus",  # 填写需要调用的模型编码
        messages=[
            {
                "role": "system",
                "content": "You are a helpful llm transpiler assistant.",
            },
            #you can use get_full_prompt or get_simple_prompt in prompt.py
            {"role": "user", "content": get_full_prompt(triton_code)},
        ],
    )
    content = response.choices[0].message.content
    # 提取代码块中的内容
    if "```python" in content:
        start = content.find("```python") + len("```python")
        end = content.find("```", start)
        if end != -1:
            return content[start:end].strip()
    elif "```" in content:
        start = content.find("```") + 3
        end = content.find("```", start)
        if end != -1:
            return content[start:end].strip()

    return content
