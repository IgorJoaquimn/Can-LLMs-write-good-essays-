import numpy as np
import requests
from conllu import parse


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

def count_clause(sentence):
  return len([token for token in sentence if(is_clause(token))])

def is_dependent_clause(token):
  for deprel in ["advcl","acl"]:
    if deprel in token["deprel"]:
      return True
  return False

def count_dependent_clause(sentence):
  return len([token for token in sentence if(is_dependent_clause(token))])

def is_Coordination(token):
  for deprel in ["conj","cc"]:
    if deprel in token["deprel"]:
      return True
  return False

def count_Coordination(sentence):
  return len([token for token in sentence if(is_Coordination(token))])

def is_T_Unit(token):
  pass

def count_token(sentence):
  return len(sentence)

def analyze_sentences(sentences):
  total_clauses = sum(count_clause(sentence) for sentence in sentences)
  total_dependent_clauses = sum(count_dependent_clause(sentence) for sentence in sentences)
  total_coordinated_phrases = sum(count_Coordination(sentence) for sentence in sentences)
  total_tokens = sum(count_token(sentence) for sentence in sentences)
  total_sentences = len(sentences)

  MLC = total_tokens / total_clauses if total_clauses > 0 else 0
  MLS = total_tokens / total_sentences if total_sentences > 0 else 0
  DCC = total_dependent_clauses / total_clauses if total_clauses > 0 else 0
  CPC = total_coordinated_phrases / total_clauses if total_clauses > 0 else 0
  
  # Calculate average and maximum tree depths
  depths = [profundidade_maxima(sentence.to_tree()) for sentence in sentences]
  profundidade_media = np.mean(depths)
  profundidade_max = np.max(depths)
  
  # Calculate Type-Token Ratio
  tokens = [token for sentence in sentences for token in sentence if token["upos"] != "PUNCT"]
  types = set(token["form"] for sentence in sentences for token in sentence)
  ttr = len(types) / len(tokens) if tokens else 0
  
  return {
      "MLC": MLC,
      "MLS": MLS,
      "DCC": DCC,
      "CPC": CPC,
      "profundidade_media": profundidade_media,
      "profundidade_max": profundidade_max,
      "ttr": ttr
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
  redacao = "https://raw.githubusercontent.com/IgorJoaquimn/Can-LLMs-write-good-essays-/main/Reda%C3%A7%C3%B5es/redacoes-prompt-pequeno/Redacao001.txt"
  r = requests.get(redacao)
  redacao = r.text
  print(analyze_text(redacao))