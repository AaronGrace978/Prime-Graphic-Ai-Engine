"""Geometry and persistence checks that do not train the diffusion model."""

import tempfile
import unittest

from prime_engine.schema import load_episode, save_episode
from prime_engine.solidify import psnr
from prime_engine.world import House, capture_the_return


class ReturnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cap, cls.info = capture_the_return(width=64, height=48, seed=0)

    def test_render_is_deterministic(self):
        house = House(seed=0)
        a, da, ia = house.render()
        b, db, ib = house.render()
        self.assertTrue((a == b).all())
        self.assertTrue((da == db).all())
        self.assertTrue((ia == ib).all())

    def test_portals_fire(self):
        self.assertTrue(any(f.portal_crossed for f in self.cap.frames))
        rooms = {f.cell_id for f in self.cap.frames}
        self.assertTrue({0, 1, 2, 3}.issubset(rooms))

    def test_hash_locks_across_the_return(self):
        self.assertNotEqual(self.info["hash_before"], self.info["hash_locked"])
        self.assertEqual(self.info["hash_locked"], self.info["hash_return"])
        self.assertEqual(self.cap.frames[self.info["return_index"]].game_state["props"]["foyer_vase"], "toppled")
        self.assertEqual(self.cap.solidify.cells[0].status, "SOLID")

    def test_guide_matches_the_room_on_revisit(self):
        frame = self.cap.frames[self.info["return_index"]]
        score = psnr(frame.guide_rgb, frame.rgb)
        coverage = float((frame.guide_mask > 0).mean())
        self.assertGreater(coverage, 0.45, f"coverage {coverage:.2f}")
        self.assertGreater(score, 16.0, f"guide PSNR {score:.2f}")

    def test_episode_roundtrip(self):
        frame = self.cap.frames[self.info["return_index"]]
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/ep.npz"
            save_episode(path, self.cap.frames[:3])
            loaded = load_episode(path)
        self.assertEqual(len(loaded), 3)
        self.assertEqual(loaded[0].episode_id, frame.episode_id)
        self.assertEqual(loaded[0].state_hash, self.cap.frames[0].state_hash)
        self.assertEqual(list(loaded[0].rgb.shape), list(self.cap.frames[0].rgb.shape))
        self.assertEqual(loaded[0].depth.dtype.str, "<f2")


if __name__ == "__main__":
    unittest.main()
