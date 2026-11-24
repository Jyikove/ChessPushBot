"""Discrete scoring destinations. Counts are the vehicle's delivery estimates."""
from math import atan2, dist, pi, remainder
from itertools import chain
from field_spec import GOAL_A_Y0, GOAL_A_Y1

LEGACY_Y_BOUNDS = (.42, 1.57)
LEGACY_PIECE_RADIUS_M = .0235
FIELD_CENTERS = tuple(GOAL_A_Y0+(i+.5)*(GOAL_A_Y1-GOAL_A_Y0)/5 for i in range(5))
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

# Leave lateral space for the piece radius and execution error at each slot edge.
ENTRY_HALF_WIDTH = (1.57-.42)/16 - LEGACY_PIECE_RADIUS_M - .020

def delivery_path(start, center, *, half_width=ENTRY_HALF_WIDTH):
    x, y = start
    entry = min(max(y, center-half_width), center+half_width)
    end = (0., entry)
    if abs(entry-y) < 1e-9:
        return [end]
    # Match the lane while still moving inward. Never return to a staging x.
    bend_x = x - min(abs(entry-y)*3, max(0., x-.26))
    return [(bend_x, entry), end]

def plan(solver, counts, coords, yaw, load, prefix, ref, *, normal_entry=False,
         centers=CENTERS, bounds=LEGACY_Y_BOUNDS, piece_radius=LEGACY_PIECE_RADIUS_M):
    goal_width = bounds[1]-bounds[0]
    entry_half_width = max(0., goal_width/(2*len(centers)) - piece_radius - .020)
    destinations = [(i, (0., y)) for i, y in enumerate(centers) if counts[i] < 4]
    if not destinations:
        return None
    if ref < 0:
        def tail(p, end):
            return delivery_path(p, end[1], half_width=entry_half_width) if normal_entry else [end]
        ref = min(cost(coords[0], yaw, path) for _, end in destinations
                  for path in ([tail(coords[0],end)] if load else
                               [[q,*tail(q,end)] for q in coords[1:]]))
    best = None
    for i, (_, y) in destinations:
        half = entry_half_width if normal_entry else 0.
        args = [len(coords)-1,4,1.15,0,y-half,y+half,load,prefix,ref,yaw,SCALE,*chain.from_iterable(coords)]
        if normal_entry:
            for start in coords:
                path = delivery_path(start, y, half_width=entry_half_width)
                first = atan2(path[0][1]-start[1], path[0][0]-start[0])
                args.extend((first, cost(start, first, path)))
        solver.stdin.write(' '.join(map(str,args))+'\n')
        solver.stdin.flush()
        loss,length,_,*route = map(float,solver.stdout.readline().split())
        candidate = (loss,length,i,route)
        if best is None or candidate < best:
            best = candidate
    return best[2], ref, [int(j)-1 for j in best[3]]
