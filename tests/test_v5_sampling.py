import random
import unittest

from scripts.v5_sampling import sample_v5_batches


class V5SamplingTests(unittest.TestCase):
    def test_exact_share_over_ten_steps(self):
        batches = sample_v5_batches(
            [{"id": "v4"}],
            [{"id": "lore"}],
            [{"id": "conflict"}],
            steps=10,
            rng=random.Random(7),
        )
        self.assertEqual(len(batches), 10)
        self.assertTrue(all(len(batch) == 4 for batch in batches))
        counts = {"v4": 0, "lore": 0, "conflict": 0}
        for batch in batches:
            for kind, _row in batch:
                counts[kind] += 1
        self.assertEqual(counts, {"v4": 27, "lore": 3, "conflict": 10})

    def test_rejects_non_integral_cycle(self):
        with self.assertRaises(ValueError):
            sample_v5_batches([1], [2], [3], steps=11)

    def test_reproducible(self):
        pools = ([{"id": "a"}, {"id": "b"}], [{"id": "l"}], [{"id": "c"}])
        first = sample_v5_batches(*pools, steps=10, rng=random.Random(42))
        second = sample_v5_batches(*pools, steps=10, rng=random.Random(42))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
