import nbformat

path = "DataDesigner_Counsellor_patient.ipynb"
nb = nbformat.read(path, as_version=4)

if "widgets" in nb["metadata"]:
    del nb["metadata"]["widgets"]
    print("Removed 'metadata.widgets' section.")

nbformat.write(nb, path)
print(f"✅ Cleaned and saved {path}")
