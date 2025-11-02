import nbformat

path = "LLM_QLoRA_Continued_Pretraining_Unlabeled.ipynb"
nb = nbformat.read(path, as_version=4)

if "widgets" in nb["metadata"]:
    del nb["metadata"]["widgets"]
    print("Removed 'metadata.widgets' section.")

nbformat.write(nb, path)
print(f"✅ Cleaned and saved {path}")
