import tempfile
import unittest
from pathlib import Path

from polar_data_loader import PolarDataLoader


def _touch(directory: Path, names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / f"{name}.mdat").touch()


class TurntablePatternTest(unittest.TestCase):
    def _index(self, root: Path):
        loader = PolarDataLoader(
            data_directory=str(root),
            pattern_type="juan_360",
            connect_to_rew=False,
            direct_ir_peak_policy="ir-start",
        )
        return {key: path.stem for key, path in loader._get_file_index().items()}

    def test_full_rotation_maps_both_halves_onto_one_driver(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _touch(root, [f"BTM {a} DEG FRONT" for a in range(0, 360, 10)])
            # Closing repeat of 0 deg; its name does not follow the pattern.
            _touch(root, ["BTM 360_0 0DEG FRON"])
            index = self._index(root)

        self.assertEqual({driver for driver, _, _ in index}, {"BTM"})
        for side in ("F", "R", "LF", "LR"):
            angles = sorted(a for _, a, s in index if s == side)
            self.assertEqual(angles, list(range(0, 91, 10)), side)

        self.assertEqual(index[("BTM", 30, "F")], "BTM 30 DEG FRONT")
        self.assertEqual(index[("BTM", 30, "R")], "BTM 150 DEG FRONT")
        self.assertEqual(index[("BTM", 90, "R")], "BTM 90 DEG FRONT")
        self.assertEqual(index[("BTM", 0, "LF")], "BTM 0 DEG FRONT")
        self.assertEqual(index[("BTM", 30, "LF")], "BTM 330 DEG FRONT")
        self.assertEqual(index[("BTM", 30, "LR")], "BTM 210 DEG FRONT")
        self.assertEqual(index[("BTM", 0, "LR")], "BTM 180 DEG FRONT")
        self.assertEqual(index[("BTM", 90, "LF")], "BTM 270 DEG FRONT")
        self.assertEqual(index[("BTM", 90, "LR")], "BTM 270 DEG FRONT")

    def test_half_sweep_variant_has_no_180_360_side(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _touch(root, [f"BTM {a} DEG FRONT" for a in range(0, 360, 10)])
            _touch(
                root / "EQ",
                [f"BTM {a} DEG FRO EQ" for a in range(0, 91, 10)]
                + [f"BTM {a} DEG REA EQ" for a in range(100, 181, 10)],
            )
            index = self._index(root)

        self.assertEqual({driver for driver, _, _ in index}, {"BTM", "BTM EQ"})
        self.assertEqual({s for d, _, s in index if d == "BTM EQ"}, {"F", "R"})
        self.assertEqual(index[("BTM EQ", 80, "R")], "BTM 100 DEG REA EQ")
        self.assertEqual(index[("BTM EQ", 90, "R")], "BTM 90 DEG FRO EQ")
        self.assertEqual(index[("BTM EQ", 0, "R")], "BTM 180 DEG REA EQ")

    def test_short_front_suffix_keeps_the_sweep_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _touch(root, [f"BTM LM+4 {a} DEG F" for a in range(0, 91, 10)])
            index = self._index(root)

        self.assertEqual({(d, s) for d, _, s in index}, {("BTM LM+4", "F"), ("BTM LM+4", "R")})
        self.assertEqual(sorted(a for _, a, s in index if s == "F"), list(range(0, 91, 10)))
        self.assertEqual(index[("BTM LM+4", 30, "F")], "BTM LM+4 30 DEG F")

    def test_missing_face_word_keeps_the_sweep_together(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            # Only the first capture of this sweep names the face toward the mic.
            _touch(root, ["BTM LM+4 10F 0 DEG F"] + [f"BTM LM+4 10F {a} DEG" for a in range(10, 91, 10)])
            index = self._index(root)

        self.assertEqual({(d, s) for d, _, s in index}, {("BTM LM+4 10F", "F"), ("BTM LM+4 10F", "R")})
        self.assertEqual(sorted(a for _, a, s in index if s == "F"), list(range(0, 91, 10)))
        self.assertEqual(index[("BTM LM+4 10F", 0, "F")], "BTM LM+4 10F 0 DEG F")
        self.assertEqual(index[("BTM LM+4 10F", 30, "F")], "BTM LM+4 10F 30 DEG")


if __name__ == "__main__":
    unittest.main()
