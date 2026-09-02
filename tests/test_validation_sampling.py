from __future__ import annotations

import unittest

from pixel_pivr.train import (
    rank_training_indices,
    stratified_file_indices,
    validation_indices,
)


class ValidationSamplingTests(unittest.TestCase):
    def test_stratified_selection_is_balanced_and_unique(self) -> None:
        signature = [{"records": value} for value in (400, 94390, 350, 350, 250)]
        selected = stratified_file_indices(signature, 1000)
        self.assertEqual(len(selected), 1000)
        self.assertEqual(len(set(selected)), 1000)

        offsets = (0, 400, 94790, 95140, 95490, 95740)
        per_file = [
            sum(offsets[index] <= value < offsets[index + 1] for value in selected)
            for index in range(5)
        ]
        self.assertEqual(per_file, [200, 200, 200, 200, 200])
        self.assertGreater(max(selected), 95000)

    def test_all_rows_preserve_concatenated_order(self) -> None:
        signature = [{"records": 2}, {"records": 3}]
        self.assertEqual(validation_indices(signature, 0, "stratified_files"), list(range(5)))

    def test_first_policy_is_explicit(self) -> None:
        signature = [{"records": 2}, {"records": 3}]
        self.assertEqual(validation_indices(signature, 3, "first"), [0, 1, 2])

    def test_shuffled_rank_schedules_cover_every_row_once(self) -> None:
        schedules = [
            rank_training_indices(10, 2, 17, "shuffled", rank, 4, 0, 3)
            for rank in range(4)
        ]
        global_order = [schedules[rank][step] for step in range(3) for rank in range(4)]
        self.assertEqual(len(set(global_order[:10])), 10)
        self.assertEqual(global_order[10:], global_order[:2])


if __name__ == "__main__":
    unittest.main()
