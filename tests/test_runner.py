# tests/test_runner.py

import importlib.util
import os
import sys

def test_triton2cuda_interface():
    print("✅ Checking tri2cu.py for triton2cuda symbol...")
    spec = importlib.util.spec_from_file_location("tri2cu", os.path.join("tri2cu.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["tri2cu"] = module
    spec.loader.exec_module(module)

    assert hasattr(module, "triton2cuda"), "❌ Missing function: triton2cuda"
    assert callable(module.triton2cuda), "❌ triton2cuda is not callable"

    dummy_code = """
    @triton.jit
    def dummy(...): pass
    """
    cuda_code = module.triton2cuda(dummy_code)
    assert isinstance(cuda_code, str), "❌ Output of triton2cuda must be a string"

    print("🎉 Interface check passed.")

if __name__ == "__main__":
    test_triton2cuda_interface()