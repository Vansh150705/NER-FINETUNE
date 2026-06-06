import json

# Inputs
INPUTS = [
    (r"C:\Users\vansh\Downloads\monolingual.bho", "bho", "bhltr"),
    (r"C:\Users\vansh\Downloads\monolingual-v0.2.bho", "bho", "wiki"),
]
OUTPUT = "bhojpuri_pretrain.jsonl"

# Convert
seen = set()
written = 0
skipped = 0

with open(OUTPUT, "w", encoding="utf-8") as out:
    for path, lang, source in INPUTS:
        with open(path, encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if len(text) < 2:
                    skipped += 1
                    continue
                if text in seen:
                    skipped += 1
                    continue
                seen.add(text)
                out.write(json.dumps({"text": text, "lang": lang, "source": source}, ensure_ascii=False) + "\n")
                written += 1

# Report
print(f"written: {written}")
print(f"skipped (blank/dupe): {skipped}")
print(f"output: {OUTPUT}")