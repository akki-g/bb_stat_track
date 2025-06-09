import unittest
import os
import json
from stats_engine import StatsEngine

class TestStatsEngineSave(unittest.TestCase):
    def test_save_stats_produces_json(self):
        engine = StatsEngine()
        # add some data
        engine.assign_player(1, 'team_a', 'Tester')
        engine.record_shot(1, True, 0, (0, 0))
        filename = 'tmp_stats.json'
        try:
            engine.save_stats(filename)
            with open(filename, 'r') as f:
                data = json.load(f)
            # basic check to ensure structure
            self.assertIn('teams', data)
            self.assertIn('team_a', data['teams'])
        finally:
            if os.path.exists(filename):
                os.remove(filename)

if __name__ == '__main__':
    unittest.main()
