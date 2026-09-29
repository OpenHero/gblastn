"""GPU-free lifecycle regression test with an explicitly mocked CUDA API.

This proves host-side ownership/discovery behavior only, never GPU execution.
"""
import pathlib
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class GpuLifecycleTests(unittest.TestCase):
    def test_lazy_discovery_device_leases_and_reinitialization(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("C++ compiler unavailable")
        with tempfile.TemporaryDirectory(prefix="gblastn-lifecycle-") as temporary:
            binary = pathlib.Path(temporary) / "lifecycle"
            subprocess.run([
                compiler, "-std=c++14", "-pthread",
                "-I" + str(ROOT / "tests/gpu_runtime_mock"),
                "-I" + str(ROOT / "c++/include"),
                str(ROOT / "tests/gpu_runtime_mock/lifecycle.cpp"),
                str(ROOT / "c++/src/algo/blast/gpu_blast/gpu_blast_multi_gpu_utils.cpp"),
                "-o", str(binary),
            ], check=True, capture_output=True, text=True)
            result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
            self.assertEqual(result.stdout, "")
            self.assertIn("GBLASTN_BACKEND backend=cpu", result.stderr)
            self.assertIn("compute_capability=11.0", result.stderr)


if __name__ == "__main__":
    unittest.main()
