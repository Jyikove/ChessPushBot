"""Discrete scoring destinations. Counts are the vehicle's delivery estimates."""
from math import atan2, dist, pi, remainder
from itertools import chain
CENTERS = tuple(.42 + (i + .5) * (1.57 - .42) / 8 for i in range(8))
SCALE = .35 / (pi / 2)

def cost(start, yaw, path):
    value = 0.
    for end in path:
        d = dist(start, end)
        if d > 1e-10:
            angle = atan2(end[1]-start[1], end[0]-start[0])
            value += d + SCALE * abs(remainder(angle-yaw, 2*pi))
            yaw = angle
        start = end
    return value

def plan(solver, counts, coords, yaw, load, prefix, ref):
    destinations = [(i, (0., y)) for i, y in enumerate(CENTERS) if counts[i] < 4]
    if not destinations:
        return None
    if ref < 0:
        ref = min(cost(coords[0], yaw, path) for _, end in destinations
                  for path in ([[end]] if load else [[q,end] for q in coords[1:]]))
    best = None
    for i, (_, y) in destinations:
        args = [len(coords)-1,4,1.15,0,y,y,load,prefix,ref,yaw,SCALE,*chain.from_iterable(coords)]
        solver.stdin.write(' '.join(map(str,args))+'\n')
        solver.stdin.flush()
        loss,length,_,*route = map(float,solver.stdout.readline().split())
        candidate = (loss,length,i,route)
        if best is None or candidate < best:
            best = candidate
    return best[2], ref, [int(j)-1 for j in best[3]]
