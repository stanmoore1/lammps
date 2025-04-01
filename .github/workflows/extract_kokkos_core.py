import os, json
from bs4 import BeautifulSoup

docs_dir = "kokkos-core-wiki/docs/generated_docs"
output_dir = "kokkos-core-knowledge"
output_path = os.path.join(output_dir, "lammps-gpt-kokkos.jsonl")
lines = []

for root, _, files in os.walk(docs_dir):
    for file in files:
        if file.endswith(".html"):
            full_path = os.path.join(root, file)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    soup = BeautifulSoup(f, "html5lib")
                    article = soup.find("article", {"role": "main"})
                    if not article:
                        continue
                    title = soup.title.string.strip() if soup.title else file
                    body = article.get_text(" ", strip=True)
                    body = body.replace("\u200b", " ").replace("\xa0", " ").replace("\u2019", "'")
                    body = body.replace("\u2014", "-").replace("\u201c", "\"").replace("\u201d", "\"")
                    if len(body) > 10:
                        lines.append(json.dumps({
                            "title": title,
                            "text": body,
                            "source": os.path.relpath(full_path, "kokkos-core-wiki/docs/generated_docs")
                        }))
            except Exception as e:
                print(f"[ERROR] Failed to parse {full_path}: {e}")

os.makedirs(output_dir, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as out:
    out.write("\n".join(lines))

print(f"[INFO] Wrote {len(lines)} entries to {output_path}")
