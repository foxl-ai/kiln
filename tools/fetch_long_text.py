"""A long real document for tools/check_long.py: public-domain books from Project Gutenberg, headers and licence
footers stripped, concatenated in order (War and Peace first, so up to ~0.8M tokens are one book).

    python tools/fetch_long_text.py /opt/kiln/data/long.txt
"""

from __future__ import annotations

import sys
import urllib.request

# Project Gutenberg ebook numbers (https://www.gutenberg.org/ebooks/<n>): War and Peace (Tolstoy, Maude translation),
# Anna Karenina (Tolstoy), Les Miserables (Hugo), Moby Dick (Melville), Don Quixote (Cervantes).
BOOKS = (2600, 1399, 135, 2701, 996)
START, END = "*** START OF", "*** END OF"


def fetch(n: int) -> str:
    url = f"https://www.gutenberg.org/cache/epub/{n}/pg{n}.txt"
    with urllib.request.urlopen(url, timeout=120) as r:
        text = r.read().decode("utf-8", errors="replace")
    a = text.find(START)
    a = text.find("\n", a) + 1 if a >= 0 else 0
    b = text.find(END)
    return text[a:b if b >= 0 else len(text)].strip()


def main() -> None:
    out = sys.argv[1]
    parts = [fetch(n) for n in BOOKS]
    with open(out, "w") as f:
        f.write("\n\n".join(parts))
    print(f"{out}: {sum(len(p) for p in parts)} characters from {len(BOOKS)} books")


if __name__ == "__main__":
    main()
