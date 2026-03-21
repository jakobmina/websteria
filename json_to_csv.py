import json
import pandas as pd
import os

# ── Configuración ────────────────────────────────────────────────────────────
input_file = "/home/jako/wesbsteria/websteria/vn_entropy_dataset.json"
output_file = "/home/jako/wesbsteria/websteria/vn_entropy_dataset.csv"
kaggle_dir = "/home/jako/wesbsteria/websteria/h7_kaggle_r"

# ── Carga y Procesamiento ───────────────────────────────────────────────────
if not os.path.exists(input_file):
    print(f"❌ Error: No se encuentra {input_file}")
    exit(1)

with open(input_file, 'r') as f:
    data = json.load(f)

# Extraer muestras
samples = data.get('samples', [])

if not samples:
    print("❌ Error: No hay muestras en el JSON.")
    exit(1)

# Convertir a DataFrame
df = pd.json_normalize(samples)

# Renombrar columnas para consistencia con el notebook de R
# (S_VN.H -> S_H, etc.)
df.columns = [c.replace('S_VN.', 'S_') for c in df.columns]

# ── Guardar ──────────────────────────────────────────────────────────────────
# Guardar en raíz del proyecto
df.to_csv(output_file, index=False)
print(f"✅ CSV generado en: {output_file}")

# Guardar en directorio de Kaggle (si existe)
if not os.path.exists(kaggle_dir):
    os.makedirs(kaggle_dir)

kaggle_csv = os.path.join(kaggle_dir, "vn_entropy_dataset.csv")
df.to_csv(kaggle_csv, index=False)
print(f"✅ CSV copiado a Kaggle dir: {kaggle_csv}")
