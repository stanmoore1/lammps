import os, json
from bs4 import BeautifulSoup

docs_dir = "kokkos-core-wiki/docs/generated_docs"
output_dir = "kokkos-core-knowledge"
output_path = os.path.join(output_dir, "knowledge.jsonl")
lines = []

for root, _, files in os.walk(docs_dir):
    for file in files:
        if file.endswith(".html"):
            full_path = os.path.join(root, file)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    soup = BeautifulSoup(f, "html5lib")
                    title = soup.title.string.strip() if soup.title else file
                    article = soup.find("article", {"role": "main"})
                    if not article:
                        continue
                    body = article.get_text(" ", strip=True)
                    body = body.replace("\u200b", " ").replace("\xa0", " ")
                    if len(body) > 100:
                        lines.append(json.dumps({
                            "title": title,
                            "text": body,
                            "source": os.path.relpath(full_path, "kokkos-core-wiki")
                        }))
            except Exception as e:
                print(f"[ERROR] Failed to parse {full_path}: {e}")

os.makedirs(output_dir, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as out:
    out.write("\n".join(lines))

print(f"[INFO] Wrote {len(lines)} entries to {output_path}")
