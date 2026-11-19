"""Embed the system's Python (tt/ + site/py/) and site/results.json into site/index.html, so the page runs exactly this code.

    python tools/build_site.py            # after any change in tt/ or site/py/

The page loads Pyodide (Python + numpy in the browser) and writes these files to its virtual disk.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site" / "index.html"
FILES = sorted((ROOT / "tt").glob("*.py")) + sorted((ROOT / "site" / "py").glob("*.py"))
BLOCK = re.compile(r'(<script type="application/json" id="pysrc">)(.*?)(</script>)', re.S)
RES = re.compile(r'(<script type="application/json" id="resdata">)(.*?)(</script>)', re.S)
RESULTS = ROOT / "site" / "results.json"      # hand-curated tables (history, robustness, failures)


def main():
    src = {}
    for f in FILES:
        rel = f.relative_to(ROOT).as_posix()
        src[rel.replace("site/py/", "")] = f.read_text(encoding="utf-8")
    blob = json.dumps(src, ensure_ascii=False).replace("</", "<\\/")
    html = SITE.read_text(encoding="utf-8")
    m = BLOCK.search(html)
    html = html[:m.start(2)] + blob + html[m.end(2):]
    res = json.dumps(json.loads(RESULTS.read_text(encoding="utf-8")), ensure_ascii=False).replace("</", "<\\/")
    m = RES.search(html)
    html = html[:m.start(2)] + res + html[m.end(2):]
    SITE.write_text(html, encoding="utf-8")
    print(f"{len(src)} files, {len(blob) // 1024} KB -> {SITE}")


if __name__ == "__main__":
    main()
