"""YOLO and image-relative field features for the current camera profile."""
import cv2
import numpy as np

from .box_tracking import BoxTracking
from ..modeling.observations import FenceLine, Piece, StartZone, Zone
from .yolo import Detector, prepare, postprocess, suppress_top_fragments

SIZE = (416, 320)


class Vision:
    def __init__(self, cfg, camera, model_dir, stress=None, seed=0, precision="fp32", threads=2):
        self.cfg, self.camera, self.stress = cfg, camera, stress
        self.detector = Detector(model_dir, SIZE, threads=threads, precision=precision)
        tracking = cfg.get("detector_tracking")
        self.box_tracking = BoxTracking(**tracking) if tracking else None
        self.rng = np.random.default_rng(seed)
        self.k = np.array(camera["K"], dtype=float)
        self.dist = np.array(camera["opencv_distortion"], dtype=float)
        self.rot = np.array(camera["R_ground_optical"], dtype=float)
        self.origin = np.array(camera["camera_base_position_m"], dtype=float)
        self.rx = 640/SIZE[0]
        self.ry = 480/SIZE[1]
        self.floor_step = .015
        self.floor_x = np.arange(.23, 1.22, self.floor_step)
        self.floor_y = np.arange(-.68, .69, self.floor_step)
        x, y = np.meshgrid(self.floor_x, self.floor_y)
        uv, depth = self.project(np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)]))
        uv *= SIZE
        self.floor_visible = ((depth > .03) & (uv[:, 0] >= 2.) & (uv[:, 0] < SIZE[0]-2)
                              & (uv[:, 1] >= SIZE[1]*.34) & (uv[:, 1] < SIZE[1]-2)).reshape(x.shape)
        self.floor_uv = uv.reshape(*x.shape, 2).astype(np.float32)
        self.floor_uv[~self.floor_visible] = -1
        offsets = np.arange(.09, .47, .015)
        self.fence_fits = np.array([(slope, lateral)
                                   for lateral in np.r_[-offsets[::-1], offsets]
                                   for slope in np.arange(-.18, .181, .03)])
        slopes, laterals = self.fence_fits.T[:, :, None]
        y = laterals + slopes*(self.floor_x-.42)
        self.fence_columns = np.arange(len(self.floor_x))
        self.fence_rows = np.rint(
            (y + np.sign(laterals)*np.array([-.045, -.018, .018, .045, .075])[:, None, None]
             - self.floor_y[0])/self.floor_step).astype(int)
        self.fence_visible = self.floor_visible[self.fence_rows, self.fence_columns].all(axis=0)
        # Sample the vertical panel close to the floor, just above the seam.
        wall_points = np.stack(np.broadcast_arrays(
            self.floor_x, y + np.sign(laterals)*.012, .042), axis=-1)
        uv, depth = self.project(wall_points)
        uv *= SIZE
        self.fence_face_uv = uv.reshape(*y.shape, 2).astype(np.float32)
        self.fence_visible &= ((depth > 0.) & (uv[:, 0] >= 1) & (uv[:, 0] < SIZE[0]-1)
                               & (uv[:, 1] >= 1) & (uv[:, 1] < SIZE[1]-1)).reshape(y.shape)
        self.prev_gray = None

    def fence_line(self, image):
        # A wall-floor seam separates two broad surfaces along the body X axis.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        floor = cv2.remap(gray, self.floor_uv, None, cv2.INTER_LINEAR).astype(np.int16)
        inside, inner, outer, outside, far = floor[self.fence_rows, self.fence_columns]
        jump = outer-inner
        contrast = np.abs(jump)
        face = cv2.remap(image, self.fence_face_uv, None, cv2.INTER_LINEAR)
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.int16)
        # The stock scene uses a neutral dark rail, and room lighting a near-white rail.
        rail = ((face_gray <= 90) | (face_gray >= 210)) & (np.ptp(face, axis=2) <= 26)
        support = (self.fence_visible & (contrast >= 26) & (np.abs(inside-inner) < 19)
                   & (np.abs(outside-outer) < 24) & (np.abs(far-outside) < 30)
                   & rail & (np.abs(face_gray-outer) <= 24))
        best, rank = None, 0.
        for polarity in (jump > 0, jump < 0):
            hits = support & polarity
            counts = hits.sum(axis=1)
            for i in np.flatnonzero(counts >= 11):
                x = self.floor_x[hits[i]]
                span = x[-1]-x[0]
                n = int(counts[i])
                if span < .18 or n*self.floor_step < .50*span:
                    continue
                strength = n*np.median(contrast[i, hits[i]])
                if strength > rank:
                    rank, best = strength, (i, x[0], x[-1], n, span)
        if best is None:
            return None
        i, lo, hi, n, span = best
        slope, lateral = self.fence_fits[i]
        uv, _ = self.project([(x, lateral+slope*(x-.42), 0.) for x in (lo, hi)])
        return FenceLine(float(lateral), tuple(map(tuple, uv.tolist())), n, float(span))

    def image_motion(self, gray, pieces):
        prev, prev_mask = self.prev_gray or (None, None)
        w, h = SIZE
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        mask[:int(.25*h)] = 0
        for p in pieces:
            x, y, u, v = np.rint(np.asarray(p.box)*[w,h,w,h]).astype(int)
            cv2.rectangle(mask, (x-5, y-5), (u+5, v+5), 0, -1)
        self.prev_gray = (gray, mask)
        if prev is None:
            return None
        background = mask & prev_mask
        features = cv2.goodFeaturesToTrack(prev, 60, .02, 9, mask=background)
        if features is not None and len(features) >= 5:
            matched, status, _ = cv2.calcOpticalFlowPyrLK(prev, gray, features, None,
                                                          winSize=(17, 17), maxLevel=2)
            valid = status.ravel().astype(bool) & np.isfinite(matched).all(axis=(1, 2))
            if np.count_nonzero(valid) >= 5:
                return float(np.median(np.linalg.norm((matched-features)[valid, 0], axis=1)))
        # A clipped view at a wall may contain edges and too few corners for LK.
        edges = cv2.Canny(prev, 35, 80).astype(bool) & (background != 0)
        if np.count_nonzero(edges) < 60:
            return None
        return float(np.percentile(cv2.absdiff(gray, prev)[edges], 75)/8.)

    def project(self, body):
        optical = (np.asarray(body).reshape(-1, 3)-self.origin)@self.rot
        uv = cv2.projectPoints(optical, np.zeros(3), np.zeros(3), self.k, self.dist)[0].reshape(-1, 2)
        return uv / [640, 480], optical[:, 2]

    def body_rays(self, uv):
        points = cv2.undistortPointsIter(
            np.asarray(uv, dtype=float).reshape(-1, 1, 2), self.k, self.dist, None, None,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 1e-9))
        return np.column_stack([points.reshape(-1, 2), np.ones(len(points))]) @ self.rot.T

    def pieces_from_detections(self, detections):
        if not detections:
            return ()
        boxes = np.asarray([d["box"] for d in detections])
        rays = self.body_rays(np.column_stack([(boxes[:, 0]+boxes[:, 2])/2, boxes[:, 3]]))
        valid = rays[:, 2] < -.012
        ranges = [None]*len(boxes)
        positions = [None]*len(boxes)
        for idx in np.flatnonzero(valid):
            ray = rays[idx]
            ground = self.origin-ray*(self.origin[2]/ray[2])
            if ground[0] > 0 and np.isfinite(ground).all():
                ranges[idx] = float(np.linalg.norm(ground[:2]))
                positions[idx] = tuple(ground[:2].tolist())
        normalized = boxes / [640, 480, 640, 480]
        return tuple(Piece(tuple(box), d["confidence"], distance, position)
                     for box, d, distance, position in
                     zip(normalized, detections, ranges, positions, strict=True))

    def score_strip_support(self, image, slope, intercept, right, left):
        """Look for a saturated outline between the approach floor and score strip."""
        y = np.linspace(right, left, 15)[1:-1]
        x = intercept+slope*y
        offsets = np.r_[-.035, np.linspace(-.012, .012, 7), .035]
        samples = np.column_stack([(x+offsets[:, None]).ravel(),
                                   np.tile(y, len(offsets)), np.zeros(len(offsets)*len(y))])
        uv, depth = self.project(samples)
        u, v = np.rint(uv*np.array(SIZE)).astype(int).T
        valid = (depth > 0) & (0 <= u) & (u < SIZE[0]) & (0 <= v) & (v < SIZE[1])
        pixels = image[v.clip(0,SIZE[1]-1), u.clip(0,SIZE[0]-1)].astype(float)
        blue = ((pixels[:, 0]-pixels[:, 2])/np.maximum(pixels[:, 0], 1.)).reshape(len(offsets), -1)
        levels = pixels.mean(axis=1).reshape(len(offsets), -1)
        valid = valid.reshape(len(offsets), -1)
        near, far = blue[0], blue[-1]
        outline = np.any(valid[1:-1] & (blue[1:-1] >= .27)
                         & (blue[1:-1]-np.maximum(near, far) >= .10), axis=0)
        supported = (valid[0] & valid[-1] & outline
                     & (far-near >= .045) & (far >= .07) & (far <= .40)
                     & (levels[-1] >= .57*levels[0]) & (levels[-1] >= 70.))
        return int(np.count_nonzero(supported))

    def goal(self, image):
        # Relative saturation separates the dark score outline from the pale start patch.
        b, g, r = [v.astype(np.int16) for v in cv2.split(image)]
        mask = (b-r >= .40*b) & (b-g >= 18) & (b >= 50)
        mask[:int(.34*SIZE[1])] = False
        yp, xp = np.nonzero(mask)
        if len(xp) < 12:
            return None
        uv = np.column_stack([xp*self.rx, yp*self.ry]).astype(float)
        rays = self.body_rays(uv)
        valid = rays[:, 2] < -.033
        rays = rays[valid]
        if not len(rays):
            return None
        ground = self.origin - rays*(self.origin[2]/rays[:, 2:3])
        keep = (ground[:, 0] > .12) & (ground[:, 0] < 2.7)
        ground = ground[keep, :2]
        if len(ground) < 12:
            return None
        # Fit the leading edge in body coordinates and verify the strip behind each fit.
        best = None
        width = .014+.035*np.median(ground[:, 0])**2
        for slope in np.sin(np.radians(np.arange(-16, 17, 4))):
            x = ground[:, 0]-slope*ground[:, 1]
            bins = np.round(x/width).astype(int)
            first_bin = int(bins.min())
            histogram = np.bincount(bins-first_bin)
            for _ in range(3):
                peak = int(histogram.argmax())
                if not histogram[peak]:
                    break
                histogram[max(0,peak-1):peak+2] = 0
                inliers = abs(x-(peak+first_bin)*width) <= 1.05*width
                subset = ground[inliers]
                if len(subset) < 12:
                    continue
                right, left = np.quantile(subset[:, 1], [.02, .98])
                span = left-right
                if span < .12:
                    continue
                intercept = float(np.median(x[inliers]))
                support = self.score_strip_support(image,slope,intercept,right,left)
                if support < 6:
                    continue
                quality = span*np.sqrt(len(subset))*support
                if best is None or quality>best[0]:
                    best = (quality,slope,intercept,left,right,len(subset))
        if best is None:
            return None
        _, slope, intercept, left, right, count = best
        margin = min(self.cfg["entry_half_width_m"], .47*(left-right))
        y = float(np.clip(0., right+margin, left-margin))
        x = float(intercept+slope*y)
        pixel, _ = self.project([[x, y, 0.], [intercept+slope*left, left, 0.],
                                 [intercept+slope*right, right, 0.]])
        line = tuple(map(tuple, pixel[1:].tolist()))
        return Zone(tuple(pixel[0].tolist()), y, x, float(left), float(right), count, line)

    def start_zone(self, image):
        """Find the broad pale-blue launch patch in the visible floor."""
        sampled = cv2.remap(image, self.floor_uv, None, cv2.INTER_LINEAR)
        b, g, r = np.moveaxis(sampled.astype(np.float32), -1, 0)
        lit_floor = self.floor_visible & (r >= 70) & (g >= 75)
        if np.count_nonzero(lit_floor) < 80:
            return None
        blue = (b-r)/np.maximum(b, 1.)
        # Nearby ground samples avoid the bright walls in the room presets.
        reference_floor = lit_floor & (self.floor_x[None, :] < .43)
        if np.count_nonzero(reference_floor) < 40:
            return None
        neutral = reference_floor & (r >= np.quantile(r[reference_floor], .75))
        relative_blue = blue-np.median(blue[neutral])
        pale = lit_floor & (relative_blue >= .055) & (relative_blue <= .17) & (b > g+8)
        connected = cv2.morphologyEx(pale.astype(np.uint8), cv2.MORPH_CLOSE,
                                     np.ones((3,3),dtype=np.uint8))
        _, labels, stats, _ = cv2.connectedComponentsWithStats(connected, connectivity=8)
        best = None
        for label, properties in enumerate(stats[1:], start=1):
            if properties[cv2.CC_STAT_AREA] < 90:
                continue
            iy, ix = np.nonzero((labels == label) & pale)
            if len(ix) < 80:
                continue
            xlo, xhi = np.quantile(self.floor_x[ix], [.10, .90])
            ylo, yhi = np.quantile(self.floor_y[iy], [.10, .90])
            # The scoring strip is narrow in the forward direction. The start patch is broad.
            if xhi-xlo < .25 or yhi-ylo < .12:
                continue
            if best is None or len(ix) > best[0]:
                best = len(ix), xlo, xhi, ylo, yhi
        if best is None:
            return None
        _, xlo, xhi, ylo, yhi = best
        marker, _ = self.project([[(xlo+xhi)/2, (ylo+yhi)/2, 0.]])
        return StartZone(tuple(marker[0].tolist()), float(xlo), (float(xhi-xlo), float(yhi-ylo)))

    def sense(self, bgr, *, timestamp=None):
        # This profile uses 640x480 pixels before detector resize and letterbox.
        if bgr.shape != (480, 640, 3) or bgr.dtype != np.uint8:
            raise ValueError("vision requires a 640x480 BGR uint8 camera frame")
        if self.stress:
            bgr = np.clip(bgr.astype(np.float32) + self.rng.normal(
                0, self.stress["camera_pixel_noise_std"], bgr.shape), 0, 255).astype(np.uint8)
        inp, transform = prepare(bgr, SIZE)
        raw, _ = self.detector.forward(inp)
        detections = postprocess(raw, transform, self.cfg["detector_confidence"], self.cfg["detector_nms_iou"])
        fragment_filter = self.cfg.get("detector_top_fragment_filter")
        if fragment_filter:
            detections = suppress_top_fragments(detections, **fragment_filter)
        pieces = self.pieces_from_detections(detections)
        if self.box_tracking is not None:
            pieces = self.box_tracking.update(pieces, timestamp)
        display = cv2.resize(bgr, SIZE, interpolation=cv2.INTER_AREA)
        zone = self.goal(display)
        start_zone = self.start_zone(display) if zone is None else None
        gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
        fence = self.fence_line(display)
        motion = self.image_motion(gray, pieces)
        return pieces, zone, start_zone, fence, motion, display
