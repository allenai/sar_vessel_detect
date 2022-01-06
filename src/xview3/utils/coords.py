import math

class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def scale(self, f):
        return Point(self.x * f, self.y * f)

    def add(self, p):
        return Point(self.x + p.x, self.y + p.y)

    def sub(self, p):
        return Point(self.x - p.x, self.y - p.y)

    def distance(self, p):
        dx = self.x - p.x
        dy = self.y - p.y
        return math.sqrt(dx*dx+dy*dy)

ORIGIN_SHIFT = 2 * math.pi * 6378137 / 2.0

def lonLatToMeters(p):
    mx = p.x * ORIGIN_SHIFT / 180.0
    my = math.log(math.tan((90 + p.y) * math.pi / 360.0)) / (math.pi / 180.0)
    my = my * ORIGIN_SHIFT / 180.0
    return Point(mx, my)

def metersToLonLat(p):
    lon = (p.x / ORIGIN_SHIFT) * 180.0
    lat = (p.y / ORIGIN_SHIFT) * 180.0
    lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180.0)) - math.pi / 2.0)
    return Point(lon, lat)

def getMetersPerPixel(zoom):
    return 2 * math.pi * 6378137 / (2**zoom) / 256

def lonLatToPixel(p, origin, zoom):
    p = lonLatToMeters(p).sub(lonLatToMeters(origin))
    p = p.scale(1 / getMetersPerPixel(zoom))
    p = Point(p.x, -p.y)
    return p

def pixelToLonLat(p, origin, zoom):
    p = Point(p.x, -p.y)
    p = p.scale(getMetersPerPixel(zoom))
    p = metersToLonLat(p.add(lonLatToMeters(origin)))
    return p
