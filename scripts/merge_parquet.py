# scripts/merge_parquet.py
from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import List
import pandas as pd

def listar_arquivos(data_dir: Path, pattern: str, recursive: bool) -> List[Path]:
    if recursive:
        return sorted(data_dir.rglob(pattern))
    return sorted(data_dir.glob(pattern))

def main():
    p = argparse.ArgumentParser(description="Merge de arquivos Parquet de data/ em um único .parquet")
    p.add_argument("--data-dir", default="data", help="Pasta onde estão os .parquet (default: data)")
    p.add_argument("--pattern", default="*.parquet", help="Padrão do nome dos arquivos (default: *.parquet)")
    p.add_argument("--recursive", action="store_true", help="Varredura recursiva (usa rglob)")
    p.add_argument("--out", default="data/ALL_MERGED.parquet", help="Caminho de saída (default: data/ALL_MERGED.parquet)")
    p.add_argument("--dedupe-on", nargs="*", default=None, help="Colunas para drop_duplicates (opcional)")
    p.add_argument("--sort-by", nargs="*", default=None, help="Colunas para ordenar antes de salvar (opcional)")
    args = p.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_path = Path(args.out)

    print(f"[merge] data_dir = {data_dir}")
    print(f"[merge] pattern  = {args.pattern}  | recursive = {args.recursive}")
    files = listar_arquivos(data_dir, args.pattern, args.recursive)
    if not files:
        print("[merge] Nenhum arquivo encontrado. Nada a fazer.")
        return

    print(f"[merge] {len(files)} arquivo(s) encontrado(s). Lendo...")
    dfs = []
    all_cols = set()

    for f in files:
        try:
            df = pd.read_parquet(f, engine="pyarrow")
            dfs.append(df)
            all_cols.update(df.columns.tolist())
            print(f"[ok] {f.name}: {len(df):,} linhas")
        except Exception as e:
            print(f"[ERRO] Falha lendo {f}: {e}", file=sys.stderr)

    if not dfs:
        print("[merge] Nenhum dataframe lido com sucesso. Abortando.")
        return

    all_cols = list(all_cols)
    print(f"[merge] Unificando schema para {len(all_cols)} coluna(s).")
    dfs = [d.reindex(columns=all_cols) for d in dfs]

    print("[merge] Concatenando...")
    big = pd.concat(dfs, ignore_index=True)

    if args.dedupe_on:
        antes = len(big)
        big = big.drop_duplicates(subset=args.dedupe_on, keep="last", ignore_index=True)
        print(f"[merge] drop_duplicates({args.dedupe_on}) → {antes:,} → {len(big):,}")

    if args.sort_by:
        big = big.sort_values(by=args.sort_by, kind="stable", ignore_index=True)
        print(f"[merge] Ordenado por: {args.sort_by}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    big.to_parquet(out_path, engine="pyarrow", index=False)
    print(f"[merge] Salvo em: {out_path}  ({len(big):,} linhas)")

if __name__ == "__main__":
    main()
