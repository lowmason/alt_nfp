"""Generate API reference pages for mkdocstrings.

Walks the src/alt_nfp package tree and creates a markdown page per module
that uses the `::: module.path` directive to auto-render docstrings.
Also generates the SUMMARY.md navigation file for literate-nav.
"""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

src = Path("src")

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    # Skip __pycache__ and private modules (except __init__)
    if "__pycache__" in parts:
        continue
    if any(part.startswith("_") and part != "__init__" for part in parts):
        continue

    if parts[-1] == "__init__":
        parts = parts[:-1]
        if not parts:
            continue
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")

    if not parts:
        continue

    nav_parts = list(parts)
    nav[nav_parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"# `{ident}`\n\n")
        fd.write(f"::: {ident}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(src))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
