"""A value-aligned control must expose, rather than hide, the objective change."""
import unittest
from rosr import model
from value_coalition import ValueCoalitionPolicy
from rosr.boot import validate_plan


class CoalitionValueTests(unittest.TestCase):
    def test_value_refinement_replaces_low_value_equal_count_acquisition(self):
        tasks={j:model.Task(j,0,0,0,4.2,value,0,2,.2,2) for j,value in [(0,1),(1,10)]}
        s=model.Snapshot(0,tasks,{j:'pending' for j in tasks},[model.legacy.Vehicle(0,0,0,0,2)],{},None,1,0)
        original=model.vehicle_plan(s,'edge_adapter')
        improved=ValueCoalitionPolicy(seconds=1)(s)
        self.assertEqual(original.value,1)
        self.assertEqual(improved.value,10)
        self.assertEqual(original.count,improved.count)
        validate_plan(s,improved)

    def test_control_preserves_seed_plan_value_in_mixed_report_state(self):
        tasks={j:model.Task(j,0,j,0,12,value,0,2,.2,2) for j,value in [(0,4),(1,9),(2,2)]}
        s=model.Snapshot(0,tasks,{0:'ready',1:'pending',2:'pending'},[model.legacy.Vehicle(0,0,0,0,2)],{0:0},None,1,0)
        original=model.vehicle_plan(s,'edge_adapter')
        improved=ValueCoalitionPolicy(seconds=1)(s)
        self.assertGreaterEqual(improved.value,original.value)


if __name__=='__main__':unittest.main()
