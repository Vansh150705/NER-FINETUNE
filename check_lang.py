import json, re

JSONL = r"C:\Users\vansh\OneDrive\Desktop\Bhoj-Data\bhojpuri_pretrain.jsonl"

devanagari = re.compile(r'[\u0900-\u097F]')
latin = re.compile(r'[A-Za-z]')

total = mostly_latin = no_devanagari = 0
junk = []

with open(JSONL, encoding="utf-8") as f:
    for line in f:
        t = json.loads(line)["text"]
        total += 1
        d = len(devanagari.findall(t))
        l = len(latin.findall(t))
        if d == 0:
            no_devanagari += 1
            if len(junk) < 15: junk.append(t[:70])
        elif l > d:
            mostly_latin += 1

print(f"total lines        : {total}")
print(f"no Devanagari      : {no_devanagari}")
print(f"more Latin than Dev: {mostly_latin}")
print("--- suspicious lines ---")
for j in junk: print(j)