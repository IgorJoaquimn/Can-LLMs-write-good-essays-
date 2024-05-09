from openai import OpenAI  # pip install openai
from api_key import api_key # É necessário criar o arquivo api_key.py e a variavel api_key = ""
import os

client = OpenAI(api_key=api_key)

# Arquivos com a instrução passada para o modelo
# system_prompt = "".join(open("prompts/Instruções com Cartilha da Redação system.txt").readlines())
system_prompt = "".join(open("prompts/system prompt matias.txt").readlines())

# Prompt de entrada do modelo, ex: escreva a redação com o tema XYZ
user_prompt   = "".join(open("prompts/Enem 2021 Sem texto motivador.txt").readlines()) 

# Pasta onde cada resposta será gravada (se não existir vai dar erro)
output_folder = "Redações/system prompt matias/Enem 2021 Sem Texto Motivador/"

# Modelo/Pasta onde cada resposta será gravada (se não existir vai dar erro)
model = "gpt-3.5-turbo" 
messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
  ]

# Quantidade de redações que serão geradas
n_repetitions = 20

if not os.path.exists(f"{output_folder}{model}/"):
    os.makedirs(f"{output_folder}{model}/")

for i in range(n_repetitions):
  # Chamada da api
  completion = client.chat.completions.create(
    model=model,
    messages = messages
  )
  # Termina escrevendo o arquivo
  with open(f"{output_folder}{model}/Igor_{i+1}.txt","w") as f:
    f.write(completion.choices[0].message.content)