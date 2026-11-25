import numpy as np


class Pose:
    def _local(self, xy):
        xy = np.asarray(xy, dtype=float)
        return self.field - xy if self.reverse else xy

    def advance_pose(self, encoder, travel):
        self.pos += travel
        self.previous = encoder
