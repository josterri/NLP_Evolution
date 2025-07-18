import os
import ast
import subprocess
import sys
import importlib.util
from pathlib import Path

def find_python_files(root_dir):
    return list(Path(root_dir).rglob("*.py"))

def extract_imports_from_file(filepath):
    imports = set()
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            tree = ast.parse(file.read(), filename=str(filepath))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
    except SyntaxError as e:
        print(f"⚠️ SyntaxError in {filepath}, skipping. Error: {e}")
    except Exception as e:
        print(f"⚠️ Could not read {filepath}. Error: {e}")
    return imports

def is_installed(pkg_name):
    try:
        __import__(pkg_name)
        return True
    except ImportError:
        return False

def install_if_needed(pkg_name):
    if is_installed(pkg_name):
        print(f"✅ Already installed: {pkg_name}")
    else:
        print(f"⬇ Installing: {pkg_name}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
            print(f"✅ Installed: {pkg_name}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install: {pkg_name}")

def find_local_modules(root_folder):
    local_modules = set()
    for py_file in find_python_files(root_folder):
        module_name = Path(py_file).stem
        local_modules.add(module_name)
    return local_modules

def write_requirements_file(packages, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for pkg in sorted(packages):
            f.write(f"{pkg}\n")
    print(f"\n📝 requirements.txt written to: {output_path}")

def main(root_folder):
    all_imports = set()
    py_files = find_python_files(root_folder)
    print(f"📁 Found {len(py_files)} Python files in: {root_folder}")

    for file in py_files:
        imports = extract_imports_from_file(file)
        print(f"📄 {file} → {imports}")
        all_imports.update(imports)

    local_modules = find_local_modules(root_folder)
    print(f"\n🚫 Local modules (excluded): {sorted(local_modules)}")

    external_imports = {imp for imp in all_imports if imp not in local_modules}
    print(f"\n📦 External imports to install: {sorted(external_imports)}\n")

    for pkg in sorted(external_imports):
        install_if_needed(pkg)

    requirements_path = os.path.join(root_folder, "requirements.txt")
    write_requirements_file(external_imports, requirements_path)

if __name__ == "__main__":
    project_dir = os.getcwd()  # or specify another folder
    main(project_dir)
