import numpy as np
import pandas as pd
import requests
from conllu import parse

def count_func(sentence, func):
  return len([token for token in sentence if(func(token))])

def profundidade_maxima(no):
  # Se nó não tem filhos, ele é folha
  if(not no.children):
    return 0

  filho_mais_fundo = max([profundidade_maxima(child) for child in no.children])
  return 1 + filho_mais_fundo

def is_clause(token):
  for deprel in ["csubj","ccomp","xcomp","advcl","acl"]:
    if deprel in token["deprel"]:
      return True
  return False

def is_dependent_clause(token):
  for deprel in ["advcl","acl"]:
    if deprel in token["deprel"]:
      return True
  return False

def is_Coordination(token):
  for deprel in ["conj","cc"]:
    if deprel in token["deprel"]:
      return True
  return False

def is_T_Unit(token):
  pass

def count_token(sentence):
  return len(sentence)

def is_lexical_words(token):
  if token["upos"] in ["NOUN","ADJ","VERB"]:
    return True
  if "advmod" in token["deprel"]:
    return True
  return False

corpus_potuguese = "https://www.wordfrequency.info/port/samples/port_40k_lemmas.txt"

r = requests.get(corpus_potuguese)
corpus = r.text
print(corpus[:100])
corpus_lines = corpus.split('\n')[7:]
column_names = ['ID', 'lemma', 'PoS', 'freq', 'texts']
df = pd.DataFrame([line.split('\t') for line in corpus_lines if line.strip()], columns=column_names)
df["freq"]  = df["freq"].apply(lambda x: int(x))
df = df[df["PoS"] == "v"]
df.sort_values(by="freq",inplace=True,ascending=False)
top_200_verbs = list(df["lemma"])

def is_sophisticated_verb(token):
  if token["upos"] != "VERB":
    return False
  if token["lemma"] in top_200_verbs:
    return False
  return True
def analyze_sentences(sentences):
  total_clauses = sum(count_func(sentence, is_clause) for sentence in sentences)
  total_dependent_clauses = sum(count_func(sentence, is_dependent_clause) for sentence in sentences)
  total_coordinated_phrases = sum(count_func(sentence, is_Coordination) for sentence in sentences)
  
  # Count total tokens and total sentences
  total_tokens = sum(count_token(sentence) for sentence in sentences)
  total_sentences = len(sentences)
  
  # Calculate average tree depth and maximum tree depth
  depths = [profundidade_maxima(sentence.to_tree()) for sentence in sentences]
  profundidade_media = np.mean(depths)
  profundidade_max = np.max(depths)
  
  # Calculate Type-Token Ratio (TTR)
  tokens = [token for sentence in sentences for token in sentence if token["upos"] != "PUNCT"]
  types = set(token["form"] for sentence in sentences for token in sentence)
  ttr = len(types) / len(tokens) if tokens else 0
  
  # Calculate lexical density (number of lexical words / total tokens)
  lexical_density = sum(count_func(sentence, is_lexical_words) for sentence in sentences) / total_tokens
  
  # Calculate number of sophisticated verbs (verbs not in top 200 verbs)
  total_sophisticated_verbs = sum(count_func(sentence, is_sophisticated_verb) for sentence in sentences)
  verbs = sum([count_func(sentence,lambda x: x["upos"] == "VERB") for sentence in sentences])

  
  # Calculate Measures of Linguistic Complexity (MLC) and Sentence Complexity (MLS)
  MLC = total_tokens / total_clauses if total_clauses > 0 else 0
  MLS = total_tokens / total_sentences if total_sentences > 0 else 0
  
  # Calculate Dependent Clauses per Clause (DCC) and Coordination per Clause (CPC)
  DCC = total_dependent_clauses / total_clauses if total_clauses > 0 else 0
  CPC = total_coordinated_phrases / total_clauses if total_clauses > 0 else 0
  
  # Return the calculated measures as a dictionary
  return {
      "MLC": MLC,
      "MLS": MLS,
      "DCC": DCC,
      "CPC": CPC,
      "profundidade_media": profundidade_media,
      "profundidade_max": profundidade_max,
      "ttr": ttr,
      "lexical_density": lexical_density,
      "lexical_sophistication": total_sophisticated_verbs/verbs,
      "token_quantity": total_tokens,
  }

def analyze_text(redacao):
  url = 'http://lindat.mff.cuni.cz/services/udpipe/api/process'
  # Dados que a API precisa
  data = {
      'tokenizer': '',
      'tagger': '',
      'parser': '',
      'model': "portuguese-bosque-ud-2.12-230717",
      'data': redacao
  }
  response = requests.post(url, data=data)
  udpipe_output = response.json()["result"]
  sentences = parse(udpipe_output)
  data = analyze_sentences(sentences)
  data["text"] = redacao
  return data

if __name__ == "__main__":
  redacao = "https://raw.githubusercontent.com/IgorJoaquimn/Can-LLMs-write-good-essays-/main/Reda%C3%A7%C3%B5es/system%20prompt%20matias/Redacao001.txt"
  r = requests.get(redacao)
  redacao = r.text
  print(analyze_text(redacao))