"""Executa toda a pipeline do projeto, na ordem, a partir dos dados brutos.

Uso:
    python executar_pipeline.py            # executa os 6 notebooks e salva as saídas neles
    python executar_pipeline.py --a-partir 4   # recomeça do notebook 04 (os anteriores já rodaram)

Cada notebook é executado com nbconvert; se algum falhar, a execução para e o erro é exibido.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

RAIZ = Path(__file__).resolve().parent
NOTEBOOKS = [
    "01_gold_local.ipynb",
    "02_enriquecimento_bases.ipynb",
    "03_eda.ipynb",
    "04_modelagem.ipynb",
    "05_interpretabilidade.ipynb",
    "06_aplicacao_estrategica.ipynb",
]


def executar(nome: str) -> None:
    caminho = RAIZ / "notebooks" / nome
    nb = nbformat.read(caminho, as_version=4)
    ep = ExecutePreprocessor(timeout=7200, kernel_name="python3")
    inicio = time.time()
    print(f"▶ {nome} ...", flush=True)
    ep.preprocess(nb, {"metadata": {"path": str(caminho.parent)}})
    nbformat.write(nb, caminho)
    print(f"  ✓ concluído em {time.time() - inicio:,.0f}s", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a-partir", type=int, default=1, help="número do notebook inicial (1 a 6)")
    args = parser.parse_args()
    for nome in NOTEBOOKS[args.a_partir - 1:]:
        try:
            executar(nome)
        except Exception as erro:  # noqa: BLE001
            print(f"  ✗ falhou em {nome}: {erro}", file=sys.stderr)
            return 1
    print("Pipeline concluída. Resultados em reports/, images/ e models/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
