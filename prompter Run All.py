from openai import OpenAI  # pip install openai
from api_key import api_key # É necessário criar o arquivo api_key.py e a variavel api_key = ""
import os
import re

client = OpenAI(api_key=api_key)

def call_openai(model, output_folder, messages,n):
    for i in range(n):
      completion = client.chat.completions.create(
                    model=model,
                    messages = messages)
      
      # Termina escrevendo o arquivo
      with open(f"{output_folder}{model}/Igor_2_{i+1}.txt","w") as f:
        f.write(completion.choices[0].message.content)


model = "gpt-3.5-turbo"
system_prompts  = ["prompts/system prompt vazio.txt"]
user_prompts    = [f"prompts/Enem 2022 Com texto motivador.txt",f"prompts/Enem 2022 Sem texto motivador.txt"]

n_iter = 50
for s in system_prompts:
   for u in user_prompts:
    system_prompt = "".join(open(s).readlines())
    user_prompt   = "".join(open(u).readlines()) 
    
    pattern = r'prompts\/(.*?)\.txt'
    prompt_type = re.search(pattern, s).group(1)
    user_type   = re.search(pattern, u).group(1)
    output_folder = f"Redações/{prompt_type}/{user_type}/"

    print(output_folder)

    if not os.path.exists(f"{output_folder}{model}/"):
      os.makedirs(f"{output_folder}{model}/")

    messages=[
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_prompt},
    ]
    call_openai(model, output_folder, messages,n_iter)