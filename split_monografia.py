from pathlib import Path
import re

SRC = Path("docs/extracted/monografia_raw.txt")
OUT = Path("docs")

OUT.mkdir(exist_ok=True)

text = SRC.read_text(encoding="utf-8")

# Detectar capítulos
chapters = re.split(
    r'(?=\*\*Capítulo \d+:)',
    text
)

created = 0

for block in chapters:
    match = re.search(r'\*\*Capítulo (\d+):\s*(.*?)\*\*', block)

    if match:
        num = match.group(1)
        title = match.group(2).strip()

        filename = (
            f"{int(num):02d}_"
            + re.sub(r'[^a-zA-Z0-9]+','_',title)
            .strip('_')
            .lower()
            + ".md"
        )

        path = OUT / filename

        content = "# Capítulo " + num + ": " + title + "\n\n"
        content += block

        path.write_text(content, encoding="utf-8")

        print("Creado:", path)
        created += 1

print()
print("Capítulos creados:", created)
