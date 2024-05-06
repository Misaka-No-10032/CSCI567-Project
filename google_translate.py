import os
from google.cloud import translate_v2
import pandas as pd
import numpy as np
from tqdm import tqdm
import csv

credential_path = "path/to/file"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path
client = translate_v2.Client()

lang_dict = {'fr': 'French', 'th': 'Thai', 'tr': 'Turkish', 'ur': 'Urdu', 'ru': 'Russian', 'bg': 'Bulgarian', 'de': 'German', 'ar': 'Arabic', 'zh': 'Chinese', 'hi': 'Hindi', 'sw': 'Swahili', 'vi': 'Vietnamese', 'es': 'Spanish', 'el': 'Greek'}

def extract_eng(path='train.csv'):
    df = pd.read_csv(path)
    eng_df = df[df['lang_abv'] == 'en']
    eng_df.to_csv('eng.csv', index=False)
    return eng_df

def translate(eng_df, path='eng.csv'):
    f = open('translated.csv', 'a', newline='', encoding='utf-8')
    writer = csv.writer(f)
    # writer.writerow(['id', 'premise', 'hypothesis', 'lang_abv', 'language', 'label'])

    for id, row in tqdm(eng_df.iterrows(), total=len(eng_df)):
        for lang_label, lang in lang_dict.items():
            new_id = row['id'] + lang_label
            translated_premise = client.translate(row['premise'], target_language=lang_label)['translatedText']
            translated_hypothesis = client.translate(row['hypothesis'], target_language=lang_label)['translatedText']
            writer.writerow([new_id, translated_premise, translated_hypothesis, lang_label, lang, row['label']])

    f.close()

if __name__ == '__main__':
    eng_df = extract_eng()
    translate(eng_df)