#!/usr/bin/env python3
# tree_txt.py  —  Genera un árbol tipo "tree /f" en un TXT, excluyendo carpetas por nombre.

import os
import sys
import argparse
from pathlib import Path

def build_tree(root: Path, excludes: set, include_files: bool, use_unicode: bool):
    TEE  = "├── " if use_unicode else "+-- "
    END  = "└── " if use_unicode else "\\-- "
    PIPE = "│   " if use_unicode else "|   "
    SPC  = "    "

    lines = [str(root.resolve())]
    counts = {"dirs": 0, "files": 0}

    def walk(dir_path: Path, prefix: str):
        try:
            with os.scandir(dir_path) as it:
                entries = []
                for e in it:
                    name_lower = e.name.lower()
                    # Excluir por nombre exacto (insensible a mayúsculas/minúsculas)
                    if e.is_dir(follow_symlinks=False) and (name_lower in excludes):
                        continue
                    # Evitar bucles por symlinks
                    if e.is_symlink():
                        continue
                    # Filtrar archivos si se pidió solo directorios
                    if not e.is_dir(follow_symlinks=False) and not include_files:
                        continue
                    entries.append(e)

                # Directorios primero, luego archivos; ordenar por nombre
                entries.sort(key=lambda x: (x.is_file(follow_symlinks=False), x.name.lower()))
        except PermissionError:
            lines.append(prefix + TEE + "[Permiso denegado]")
            return

        for i, entry in enumerate(entries):
            is_last = (i == len(entries) - 1)
            branch = END if is_last else TEE
            lines.append(prefix + branch + entry.name)

            if entry.is_dir(follow_symlinks=False):
                counts["dirs"] += 1
                walk(Path(entry.path), prefix + (SPC if is_last else PIPE))
            else:
                counts["files"] += 1

    walk(root, "")
    lines.append("")  # línea en blanco final
    lines.append(f"{counts['dirs']} directorios, {counts['files']} archivos")
    return lines

def main():
    parser = argparse.ArgumentParser(
        description="Genera un árbol en TXT (tipo tree /f) excluyendo carpetas por nombre."
    )
    parser.add_argument("path", nargs="?", default=".", help="Ruta raíz (por defecto: .)")
    parser.add_argument("-o", "--out", default="tree.txt", help="Archivo de salida (por defecto: tree.txt)")
    parser.add_argument(
        "-x", "--exclude", nargs="*", default=["data", "__pycache__", "pycache", ".git"],
        help="Nombres de carpetas a excluir (match por nombre exacto, sin importar mayúsculas)."
    )
    parser.add_argument(
        "--dirs-only", action="store_true",
        help="Solo directorios (no lista archivos)."
    )
    parser.add_argument(
        "--unicode", action="store_true",
        help="Usa caracteres Unicode (├──, └──, │). Por defecto usa ASCII."
    )

    args = parser.parse_args()
    root = Path(args.path)

    if not root.exists():
        print(f"Ruta no encontrada: {root}", file=sys.stderr)
        sys.exit(1)

    excludes = {name.lower() for name in (args.exclude or [])}
    lines = build_tree(root, excludes, include_files=not args.dirs_only, use_unicode=args.unicode)

    out_path = Path(args.out)
    try:
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
    except OSError as e:
        print(f"No se pudo escribir el archivo de salida: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"Árbol generado en: {out_path.resolve()}")

if __name__ == "__main__":
    main()
