import sys
import json
import hashlib
from pathlib import Path

# Librerías desarrolladas por Hugging Face
try:
    from datasets import load_dataset, Dataset
except ImportError:
    print("Error. Libreria datasets no instalada: pip install datasets")
    sys.exit(1)

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Error. Libreria huggingface_hub no instalada: pip install huggingface_hub")
    sys.exit(1)

# Directorio para cache:
# Path.home() devuelve /Users/gerardo
# Por tanto los datasets quedan en la carpeta Users/gerardo/.cache/huggingface/datasets

CACHE_DIR = Path.home() / ".cache" / "huggingface" / "datasets"

# Descarga un dataset con los datos de entrada:
#  path = dataset_name (nombre del dataset) (obligatorio)
#  name = config (nombre de la configuración) (opcional, por defecto None=>ninguno) 
#  split = split (que split cargar) (opcional, por defecto train=>entrenamiento) 
# Imprime varias caracteristicas
# Devuelve el puntero al dataset
def load_and_inspect(dataset_name: str, config: str = None, split: str = "train"):
    kwargs = {"path": dataset_name}
    if config:
        kwargs["name"] = config
    if split:
        kwargs["split"] = split

    ds = load_dataset(**kwargs)
    print(f"Dataset: {dataset_name}")
    print(f"  Split: {split}")
    print(f"  Filas-Rows: {len(ds)}")
    print(f"  Columnas: {ds.column_names}")
    print(f"  Caracteristicas: {ds.features}")
    print(f"  Primera fila: {ds[0]}")
    max_rows = 5
    for i, example in enumerate(ds):
        print(f"   Fila {i+1}: {example}")
        if i >= max_rows - 1:
            break
    return ds

# Descarga dataset por streaming con los datos de entrada
#  path = datasetname (nombre del dataset) (obligatorio)
#  name = config (nombre de la configuracion) (opcional)
#  max_rows = numero máximo de filas a descargar (por defecto 5)
#  Campos fijos: split=train; streaming=True
# Imprime 
def stream_dataset(dataset_name: str, config: str = None, max_rows: int = 5):
    kwargs = {"path": dataset_name, "split": "train", "streaming": True}
    if config:
        kwargs["name"] = config

    ds = load_dataset(**kwargs)
    rows = []
    for i, example in enumerate(ds):
        rows.append(example)
        if i >= max_rows - 1:
            break

    print(f"Recibidos por streaming {len(rows)} filas de {dataset_name}")
    return rows

# Conversion de formato del dataset a csv, json y parquet
# Entrada: ds         = puntero al dataset
#          output_dir = directorio de salida (obligatorio)
#          name       = nombre del fichero de salida convertido
# Imprime una comparación de tamaños de los formatos
# Devuelve punteros a los ficheros convertidos (csv_path, json_path, parquet_path)
def convert_format(ds, output_dir: str, name: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / f"{name}.csv"
    json_path = output_path / f"{name}.json"
    parquet_path = output_path / f"{name}.parquet"

    ds.to_csv(str(csv_path))
    ds.to_json(str(json_path))
    ds.to_parquet(str(parquet_path))

    csv_size = csv_path.stat().st_size
    json_size = json_path.stat().st_size
    parquet_size = parquet_path.stat().st_size

    print(f"Format comparison for {name}:")
    print(f"  CSV:     {csv_size:>10,} bytes")
    print(f"  JSON:    {json_size:>10,} bytes")
    print(f"  Parquet: {parquet_size:>10,} bytes")
    print(f"  Parquet is {csv_size / parquet_size:.1f}x smaller than CSV")

    return {"csv": csv_path, "json": json_path, "parquet": parquet_path}

# Generar splits
# Entrada:
#   ds = dataset (obligatorio)
#   train_ratio = ratio para el split train (por defecto 0.8)
#   val_ratio = ratio para el split validation (por defecto 0.1)
#   seed = semilla (por defecto 42)
# El ratio para test (test_ratio) se calcula como 1 menos los demas
# Devuelve los punteros a los splits (train_ds, val_ds y test_ds)
def make_splits(ds, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42):
    test_ratio = 1.0 - train_ratio - val_ratio
    assert test_ratio > 0, "train_ratio + val_ratio must be less than 1.0"

    test_size = val_ratio + test_ratio
    split1 = ds.train_test_split(test_size=test_size, seed=seed)
    train_ds = split1["train"]

    val_fraction = val_ratio / test_size
    split2 = split1["test"].train_test_split(test_size=(1.0 - val_fraction), seed=seed)
    val_ds = split2["train"]
    test_ds = split2["test"]

    total = len(train_ds) + len(val_ds) + len(test_ds)
    print(f"Splits (seed={seed}):")
    print(f"  Train: {len(train_ds):>6} ({len(train_ds)/total:.1%})")
    print(f"  Val:   {len(val_ds):>6} ({len(val_ds)/total:.1%})")
    print(f"  Test:  {len(test_ds):>6} ({len(test_ds)/total:.1%})")

    return {"train": train_ds, "val": val_ds, "test": test_ds}

# Descargar un modelo de Hugging Face
#   repo_id = nombre del repo (obligatorio)
#   filename = nombre del fichero (obligatorio)
# Devuelve un puntero al fichero descargado
def download_model_file(repo_id: str, filename: str):
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    size = Path(path).stat().st_size
    print(f"Downloaded {filename} from {repo_id}")
    print(f"  Path: {path}")
    print(f"  Size: {size:,} bytes")
    return path

# Imprime información del directorio de Cache
def cache_summary():
    cache_path = CACHE_DIR
    if not cache_path.exists():
        print("No HF cache found yet.")
        return

    total_size = 0
    file_count = 0
    for f in cache_path.rglob("*"):
        if f.is_file():
            total_size += f.stat().st_size
            file_count += 1

    print(f"HF Dataset Cache: {cache_path}")
    print(f"  Files: {file_count}")
    print(f"  Total size: {total_size / (1024 * 1024):.1f} MB")

# Cargar un dataset en formato parquet
def load_from_parquet(path: str):
    ds = Dataset.from_parquet(path)
    print(f"Loaded {len(ds)} rows from {path}")
    return ds

# Cargar un dataset en formato csv
def load_from_csv(path: str):
    ds = Dataset.from_csv(path)
    print(f"Loaded {len(ds)} rows from {path}")
    return ds

# Cargar un dataset en formato json
def load_from_json(path: str):
    ds = Dataset.from_json(path)
    print(f"Loaded {len(ds)} rows from {path}")
    return ds

"""
Una huella digital (fingerprint) y un resumen (digest) son términos que a menudo 
se usan como sinónimos en informática para referirse a una cadena corta de datos 
que identifica de forma única a un archivo o mensaje más grande, pero tienen 
matices distintos
Esta funcion devuelve el digest (resumen) de un dataset
"""
def fingerprint(ds, num_rows: int = 100):
    sample = ds.select(range(min(num_rows, len(ds))))
    content = json.dumps([row for row in sample], default=str).encode()
    digest = hashlib.sha256(content).hexdigest()[:16]
    print(f"Dataset fingerprint (first {num_rows} rows): {digest}")
    return digest

# COMIENZO DEL PROGRAMA

if __name__ == "__main__":
    print("=" * 60)
    print("Utilidad de Gestión de Datos (Data Management)")
    print("=" * 60)

    print("\n--- 1. Carga e inspección de un dataset ---")
    ds = load_and_inspect("glue", config="mrpc", split="train")

    print("\n--- 2. Hacer Streaming de un dataset ---")
    rows = stream_dataset("c4", max_rows=3)
    for row in rows:
        print(f"  {row['text'][:80]}...")

    print("\n--- 3. Conversión de formatos ---")
    small_ds = ds.select(range(500))
    paths = convert_format(small_ds, "/tmp/data_utils_demo", "rotten_tomatoes_sample")

    print("\n--- 4. Crear los splits train/val/test ---")
    splits = make_splits(small_ds, train_ratio=0.8, val_ratio=0.1, seed=42)

    print("\n--- 5. Recarga desde Parquet ---")
    reloaded = load_from_parquet(str(paths["parquet"]))
    print(f"  Columns: {reloaded.column_names}")

    print("\n--- 6. Descargar un modelo ---")
    download_model_file("sentence-transformers/all-MiniLM-L6-v2", "config.json")

    print("\n--- 7. Dataset fingerprint ---")
    fingerprint(ds)

    print("\n--- 8. Cache summary ---")
    cache_summary()

    print("\n" + "=" * 60)
    print("All checks passed. Your data pipeline is ready.")
    print("=" * 60)
