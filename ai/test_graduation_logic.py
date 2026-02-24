import sys
import os
import unittest

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.muzero_config import MuZeroConfig
from ai.curriculum import CurriculumManager

class TestGraduationLogic(unittest.TestCase):
    def setUp(self):
        self.config = MuZeroConfig()
        self.cm = CurriculumManager(self.config)

    def test_fixed_step_graduation(self):
        # Stage 1: 15x15 -> 500 steps
        self.assertEqual(self.cm.stages[0].min_steps, 500)
        self.assertFalse(self.cm.check_graduation(step=499))
        self.assertTrue(self.cm.check_graduation(step=500))
        self.assertTrue(self.cm.check_graduation(step=1000))
        
        self.cm.advance()
        # Stage 2: 30x30 -> 1000 steps
        self.assertEqual(self.cm.stages[1].min_steps, 1000)
        self.assertFalse(self.cm.check_graduation(step=999))
        self.assertTrue(self.cm.check_graduation(step=1000))

        self.cm.advance()
        # Stage 3: 50x50 -> 1500 steps
        self.assertEqual(self.cm.stages[2].min_steps, 1500)
        self.assertFalse(self.cm.check_graduation(step=1499))
        self.assertTrue(self.cm.check_graduation(step=1500))
        
        self.cm.advance()
        # Stage 4: 100x100 -> infinity
        self.assertEqual(self.cm.stages[3].board_size, 100)
        self.assertFalse(self.cm.check_graduation(step=999999))
        
    def test_state_dict_roundtrip(self):
        self.cm.games_in_stage = 100
        self.cm.win_rate_buffer = [1.0] * 50
        sd = self.cm.state_dict()
        self.assertIn('games_in_stage', sd)
        self.assertEqual(sd['games_in_stage'], 100)
        cm2 = CurriculumManager(self.config)
        cm2.load_state_dict(sd)
        self.assertEqual(cm2.games_in_stage, 100)

if __name__ == '__main__':
    unittest.main()
