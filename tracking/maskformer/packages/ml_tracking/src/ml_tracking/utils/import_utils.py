import subprocess
import sys


def check_import_safe(module_name, required_attributes=()):
    code = f"""
try:
    import {module_name} as module
    if not all(hasattr(module, attr) for attr in {required_attributes!r}):
        exit(1)
except Exception:
    exit(1)
exit(0)
"""
    result = subprocess.run([sys.executable, "-c", code])
    return result.returncode == 0
