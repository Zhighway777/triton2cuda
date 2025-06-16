import os
import sys
import tempfile
import torch
import importlib.util
import sys

folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(folder_path)
sys.path.append(os.path.join(folder_path, "example_submission"))
sys.path.append(os.path.join(folder_path, "data", "ref"))

TEST_NN_MODEL_NAME = 'ModelNew'

from tri2cu import triton2cuda

def eval_simple():
    os.chdir(folder_path)
    triton_code = open("data/triton/vecadd.py", "r").read()
    cuda_code = triton2cuda(triton_code)
    print(cuda_code)
    from vecadd import Model
    from vecadd import get_inputs
    input_tensors = get_inputs()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, "cuda_code.py")
        with open(temp_file, "w") as f:
            f.write(cuda_code)
            
        spec = importlib.util.spec_from_file_location(TEST_NN_MODEL_NAME, temp_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[TEST_NN_MODEL_NAME] = module
        spec.loader.exec_module(module)
        triton_pm = getattr(module, TEST_NN_MODEL_NAME, None)
        if triton_pm is None:
            raise ValueError(f"Could not find class {TEST_NN_MODEL_NAME} in {temp_file}")
    for i, input_tensor in enumerate(input_tensors):
        input_tensor = [inp.detach().clone().cuda() for inp in input_tensor]
        output = Model()(*input_tensor)
        triton_output = triton_pm()(*input_tensor)
        assert torch.allclose(triton_output, output, atol=1e-3, rtol=1e-3)
        print(f"Test {i} passed")
    print(f"Test all passed")

def eval_golden():
    os.chdir(folder_path)
    triton_code = open("data/triton/vecadd.py", "r").read()
    golden_file_path = "data/cuda/vecadd.py"
    cuda_code = triton2cuda(triton_code)
    print(cuda_code)
    from vecadd import Model
    from vecadd import get_inputs
    input_tensors = get_inputs()
    spec = importlib.util.spec_from_file_location(TEST_NN_MODEL_NAME, golden_file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[TEST_NN_MODEL_NAME] = module
    spec.loader.exec_module(module)
    triton_pm = getattr(module, TEST_NN_MODEL_NAME, None)
    for i, input_tensor in enumerate(input_tensors):
        input_tensor = [inp.detach().clone().cuda() for inp in input_tensor]
        output = Model()(*input_tensor)
        triton_output = triton_pm()(*input_tensor)
        assert torch.allclose(triton_output, output, atol=1e-3, rtol=1e-3)
        print(f"Test {i} passed")
    print(f"Test all passed")

if __name__ == "__main__":
    # eval_golden()
    eval_simple()
    