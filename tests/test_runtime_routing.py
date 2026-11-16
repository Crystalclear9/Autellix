import unittest

from autellix.runtime import InferenceEngine


class RuntimeRoutingTests(unittest.TestCase):
    def engine(self, routing):
        engine = InferenceEngine.__new__(InferenceEngine)
        engine.loads = [4, 0, 2]
        engine.threshold = 2048
        engine.routing = routing
        engine.routing_cursor = 0
        return engine

    def test_round_robin_uses_all_healthy_replicas(self):
        engine = self.engine("round-robin")
        self.assertEqual([engine._route({"engine": None}, 3000, [0, 1, 2]) for _ in range(6)],
                         [0, 1, 2, 0, 1, 2])
        self.assertEqual([engine._route({"engine": None}, 3000, [0, 2]) for _ in range(4)],
                         [0, 2, 0, 2])

    def test_least_used_rebalances_long_calls(self):
        engine = self.engine("least-used")
        session = {"engine": 0}
        self.assertEqual(engine._route(session, 3000, [0, 1, 2]), 1)
        engine.loads = [4, 8, 2]
        self.assertEqual(engine._route(session, 3000, [0, 1, 2]), 2)

    def test_program_routing_balances_short_calls_and_keeps_long_history(self):
        engine = self.engine("program")
        session = {"engine": None}
        self.assertEqual(engine._route(session, 2048, [0, 1, 2]), 1)
        self.assertIsNone(session["engine"])
        self.assertEqual(engine._route(session, 2049, [0, 1, 2]), 1)
        engine.loads = [4, 8, 2]
        self.assertEqual(engine._route(session, 3000, [0, 1, 2]), 1)
        self.assertEqual(engine._route(session, 3000, [0, 2]), 2)
