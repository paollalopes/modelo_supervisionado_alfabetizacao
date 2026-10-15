"""Configurações centrais do projeto: caminhos, sementes e constantes de negócio."""

from pathlib import Path

# Raiz do repositório (src/config.py -> src -> raiz)
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
IMAGES_DIR = ROOT_DIR / "images"

for _d in (PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, IMAGES_DIR, EXTERNAL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Semente única para garantir replicabilidade
RANDOM_STATE = 42

# Ponto de corte do INEP: aluno é considerado alfabetizado com proficiência >= 743
CORTE_ALFABETIZACAO = 743

# Códigos de rede de ensino (dicionário do INEP / Base dos Dados)
MAPA_REDE = {0: "Outros", 2: "Estadual", 3: "Municipal", 5: "Pública"}

RAW_FILES = {
    "avaliacao_municipio": "avaliacao_alfabetizacao_municipio.csv",
    "avaliacao_uf": "avaliacao_alfabetizacao_uf.csv",
    "avaliacao_alunos": "avaliacao_alunos.csv",
    "meta_brasil": "meta_alfabetizacao_brasil.csv",
    "meta_municipio": "meta_alfabetizacao_municipio.csv",
    "meta_uf": "meta_alfabetizacao_uf.csv",
}
