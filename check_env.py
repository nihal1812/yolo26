import ast
import importlib
import sys
from pathlib import Path

# Scan only your actual source folders, not the whole project blindly
SCAN_DIRS = ["src", "models", "config"]
REQUIREMENTS_FILE = Path("requirements.txt")

# import name -> pip package name
IMPORT_TO_PACKAGE = {
    "cv2": "opencv-python",
    "PIL": "pillow",
    "yaml": "PyYAML",
    "zmq": "pyzmq",
    "gi": "PyGObject",  # usually system-installed on Jetson
}

# things that should not be added to requirements.txt
STDLIB_MODULES = {
    "abc", "argparse", "ast", "asyncio", "base64", "collections", "concurrent",
    "contextlib", "copy", "csv", "dataclasses", "datetime", "enum", "functools",
    "glob", "hashlib", "heapq", "http", "importlib", "inspect", "io", "itertools",
    "json", "logging", "math", "multiprocessing", "os", "pathlib", "pickle",
    "platform", "queue", "random", "re", "selectors", "shlex", "shutil", "signal",
    "socket", "sqlite3", "statistics", "string", "struct", "subprocess", "sys",
    "tempfile", "threading", "time", "traceback", "types", "typing", "unittest",
    "urllib", "uuid", "warnings", "weakref", "xml", "zipfile",
}

# modules provided by your own project
LOCAL_MODULES = {
    "config_utils",
}

# modules installed via apt / system packages rather than pip
SYSTEM_MODULES = {
    "gi": "python3-gi / python3-gi-cairo / python3-gst-1.0",
}

PACKAGE_IMPORTS = [
    ("numpy", "numpy"),
    ("yaml", "PyYAML"),
    ("redis", "redis"),
    ("cv2", "opencv-python"),
    ("PIL", "pillow"),
    ("ultralytics", "ultralytics"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("torchaudio", "torchaudio"),
    ("tqdm", "tqdm"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("psutil", "psutil"),
    ("requests", "requests"),
    ("sympy", "sympy"),
    ("mpmath", "mpmath"),
    ("msgpack", "msgpack"),
]

def normalize_req_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")

def read_requirements(req_file: Path):
    reqs = set()
    if not req_file.exists():
        return reqs

    for raw in req_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            continue
        pkg = line.split(";")[0].strip()
        for sep in ["==", ">=", "<=", "~=", "!=", ">", "<"]:
            if sep in pkg:
                pkg = pkg.split(sep)[0].strip()
                break
        if pkg:
            reqs.add(normalize_req_name(pkg))
    return reqs

def import_and_version(module_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, "__version__", "unknown")

def check_python_packages():
    print("=== Python package import checks ===")
    ok = True
    for module_name, package_name in PACKAGE_IMPORTS:
        try:
            version = import_and_version(module_name)
            print(f"[OK]   {module_name:<12} ({package_name}) version={version}")
        except Exception as e:
            ok = False
            print(f"[FAIL] {module_name:<12} ({package_name}) -> {e}")
    print()
    return ok

def check_torch_cuda():
    print("=== Torch / CUDA checks ===")
    ok = True
    try:
        import torch
        import torch.nn as nn

        print(f"Torch version      : {torch.__version__}")
        print(f"CUDA available     : {torch.cuda.is_available()}")
        print(f"cuDNN enabled      : {torch.backends.cudnn.enabled}")
        print(f"cuDNN version      : {torch.backends.cudnn.version()}")

        if torch.cuda.is_available():
            print(f"CUDA device        : {torch.cuda.get_device_name(0)}")

            x = torch.randn(2, 3, device="cuda")
            print(f"[OK]   GPU tensor test passed: shape={tuple(x.shape)}")

            x = torch.randn(1, 3, 224, 224, device="cuda")
            m = nn.Conv2d(3, 16, 3, padding=1).cuda()
            y = m(x)
            print(f"[OK]   GPU conv2d test passed: output_shape={tuple(y.shape)}")
        else:
            ok = False
            print("[FAIL] CUDA is not available")

    except Exception as e:
        ok = False
        print(f"[FAIL] Torch CUDA test -> {e}")

    print()
    return ok

def get_imports_from_py_file(py_file: Path):
    imports = set()
    try:
        source = py_file.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source, filename=str(py_file))
    except Exception:
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top:
                    imports.add(top)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top:
                    imports.add(top)
    return imports

def scan_project_imports():
    all_imports = set()
    py_files = []

    for d in SCAN_DIRS:
        path = Path(d)
        if not path.exists():
            continue
        for py_file in path.rglob("*.py"):
            py_files.append(py_file)
            all_imports.update(get_imports_from_py_file(py_file))

    return all_imports, py_files

def compare_imports_to_requirements(project_imports, requirements):
    print("=== Project import scan ===")
    print(f"Scanned imports: {len(project_imports)}")
    print()

    missing_pip = []
    system_needed = []

    for imp in sorted(project_imports):
        if imp in STDLIB_MODULES:
            continue
        if imp in LOCAL_MODULES:
            continue

        if imp in SYSTEM_MODULES:
            system_needed.append((imp, SYSTEM_MODULES[imp]))
            continue

        pkg_name = IMPORT_TO_PACKAGE.get(imp, imp)
        normalized = normalize_req_name(pkg_name)

        if normalized not in requirements:
            missing_pip.append((imp, pkg_name))

    if missing_pip:
        print("=== Possible missing pip requirements ===")
        for imp, pkg in missing_pip:
            if imp == pkg:
                print(f"- import '{imp}'")
            else:
                print(f"- import '{imp}'  -> package '{pkg}'")
        print()
    else:
        print("=== Pip requirements coverage ===")
        print("No obvious missing third-party pip imports found.")
        print()

    if system_needed:
        print("=== System/Apt-backed modules detected ===")
        for imp, pkg in system_needed:
            print(f"- import '{imp}' -> provided by {pkg}")
        print()

    return missing_pip, system_needed

def suggest_lines(missing_pip):
    if not missing_pip:
        return

    print("=== Suggested lines to add to requirements.txt ===")
    seen = set()
    for _, pkg in missing_pip:
        norm = normalize_req_name(pkg)
        if norm not in seen:
            seen.add(norm)
            print(pkg)
    print()

def main():
    print("check_env.py starting...\n")

    requirements = read_requirements(REQUIREMENTS_FILE)
    if REQUIREMENTS_FILE.exists():
        print(f"Loaded requirements from: {REQUIREMENTS_FILE}")
        print(f"Requirement entries found: {len(requirements)}\n")
    else:
        print("requirements.txt not found\n")

    packages_ok = check_python_packages()
    torch_ok = check_torch_cuda()

    project_imports, py_files = scan_project_imports()
    print(f"Python files scanned: {len(py_files)}")
    print(f"Directories scanned : {', '.join([d for d in SCAN_DIRS if Path(d).exists()])}\n")

    missing_pip, system_needed = compare_imports_to_requirements(project_imports, requirements)
    suggest_lines(missing_pip)

    print("=== Final status ===")
    if packages_ok and torch_ok:
        print("Environment core checks: OK")
    else:
        print("Environment core checks: FAILED")

    if missing_pip:
        print("requirements.txt may be incomplete")
    else:
        print("requirements.txt looks reasonably complete")

if __name__ == "__main__":
    main()
