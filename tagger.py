import joblib
from nltk import word_tokenize
import os
import pandas as pd


folder = 'POS-tagger-portuguese-nltk-master/trained_POS_taggers/'
tagger = joblib.load(folder+'POS_tagger_brill.pkl')
phrase = 'O rato roeu a roupa do rei de Roma'

df = pd.DataFrame(columns=['File_Name', 'Text_Content'])

for file_name in os.listdir("corpus/humanos"):
    if file_name.endswith('.txt'):
        file_path = os.path.join("corpus/humanos", file_name)
        with open(file_path, 'r') as file:
            text_content = file.read()
        # Append a new row to the DataFrame
        df = pd.concat([df, pd.DataFrame({'File_Name': [file_name[:-4]], 'Text_Content': [text_content]})], ignore_index=True)

print(df)