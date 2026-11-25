import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"runtime"))
from match_referee import GOAL_INTERIORS, score
from piece_spec import PIECES, RADIUS_M


class MatchRefereeTest(unittest.TestCase):
    def setUp(self):
        self.pos=np.tile([1.5,1.,.0085],(len(PIECES),1))
        self.axes=np.tile([0.,0.,1.],(len(PIECES),1))
        self.floor=np.ones(len(PIECES),dtype=bool)
        self.home_a=np.array([.2,1.])
        self.home_b=np.array([2.8,1.])

    def test_each_team_counts_only_own_pieces(self):
        self.pos[0,:2]=self.home_a
        self.pos[1,:2]=self.home_b
        self.pos[2,:2]=self.home_b
        self.pos[3,:2]=self.home_a
        result=score(self.pos,self.axes,self.floor)
        self.assertEqual((result["A"]["points"],result["B"]["points"]),(10,10))

    def test_airborne_and_boundary_do_not_score(self):
        self.pos[0,:2]=GOAL_INTERIORS[0,:2]+[RADIUS_M,.2]
        self.pos[1,:2]=self.home_b
        self.floor[1]=False
        result=score(self.pos,self.axes,self.floor)
        self.assertEqual((result["A"]["points"],result["B"]["points"]),(0,0))
        self.pos[0,0]+=.001
        self.assertEqual(score(self.pos,self.axes,self.floor)["A"]["points"],10)
        self.pos[0,0]+=.33
        self.assertEqual(score(self.pos,self.axes,self.floor)["A"]["points"],0)
