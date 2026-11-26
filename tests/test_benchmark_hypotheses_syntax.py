from pathlib import Path
import unittest


class BenchmarkHarnessSyntaxTests(unittest.TestCase):
    def test_paired_cpu_benchmark_script_compiles(self):
        path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_v22_cpu_hypotheses.py"
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")


if __name__ == "__main__":
    unittest.main()
