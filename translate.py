import requests
import csv
from tqdm import tqdm


def translate_text(text, target_lang, api_key):
    url = "https://api-free.deepl.com/v2/translate"
    data = {"auth_key": api_key, "text": text, "target_lang": target_lang}
    response = requests.post(url, data=data)
    result = response.json()
    return result["translations"][0]["text"]


# api_key = "API_KEY"

target_lang = "ZH"

with open("../train.csv", "r") as fp:
    reader = csv.reader(fp)
    header = next(reader)
    rows = list(reader)

with open("../train_zh_1.csv", "w", newline="") as fp_zh:
    writer = csv.writer(fp_zh)
    writer.writerow(["id", "premise", "hypothesis", "lang_abv", "language", "label"])
    for row in tqdm(rows, desc="Translating"):
        if row[3] == "en":
            id_ = row[0] + "_zh"
            premise = row[1]
            hypothesis = row[2]
            label = row[5]
            lang_abv = "zh"
            language = "Chinese"
            translated_premise = translate_text(premise, target_lang, api_key)
            translated_hypothesis = translate_text(hypothesis, target_lang, api_key)
            writer.writerow(
                [
                    id_,
                    translated_premise,
                    translated_hypothesis,
                    lang_abv,
                    language,
                    label,
                ]
            )
print("Translation complete!")
