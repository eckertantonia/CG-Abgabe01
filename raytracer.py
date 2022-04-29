import numpy as np
import math as math
from PIL import Image
import tqdm
import threading
import datetime
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Funktion zum Normalisieren von Vektoren
def normalize(zahl):
    normalisiert = zahl/np.linalg.norm(zahl)
    return normalisiert

# Funktion zur Farbberechnung der einzelnen Pixel mit Phong
def phong(obj, ray):
    lichtquelle = np.array([0, 20, 0])
    cIn = np.array([255, 255, 255])
    schnittpunkt = ray.pointAtParameter(obj.intersectionParameter(ray))

    # weil schachbrettmuster
    if isinstance(obj.color, CheckerboardMaterial):
        cA = obj.color.baseColorAt(schnittpunkt)
    else:
        cA = obj.color

    l = normalize(lichtquelle - schnittpunkt)
    n = obj.normalAt(schnittpunkt)
    lr = normalize(l - 2 * np.dot(n, l) * n)
    d = normalize(schnittpunkt-cam.c)

    ln = np.dot(l, n)
    ld = np.dot(lr, -d)

    ka = 0.4
    kd = 0.5
    ks = 0.5
    c_out = cA * ka + cIn * kd * ln + cIn * ks * ld

    #gucken ob da ein schatten ist
    for o in objectlist:
        rayneu = Ray(schnittpunkt, l)
        schnitt = o.intersectionParameter(rayneu)
        if schnitt and schnitt > 0.01:
            #schattenberechnung: farbe * 0.2
            c_dunkel = c_out*0.2
            return (int(c_dunkel[0]), int(c_dunkel[1]), int(c_dunkel[2]))

    return (int(c_out[0]), int(c_out[1]), int(c_out[2]))

# Ray-Klasse aus Skript
class Ray(object):
    def __init__(self, origin, direction):
        self.origin = origin  # point
        self.direction = normalize(direction)  # vector

    def __repr__(self):
        return 'Ray(%s,%s)' % (repr(self.origin), repr(self.direction))

    def pointAtParameter(self, t):
        return self.origin + self.direction * t  # vorher .scale


# Sphere-Klasse aus Skript
class Sphere(object):
    def __init__(self, center, radius, color):
        self.center = center  # point
        self.radius = radius  # scalar
        self.color = color
        self.reflection = 0.5

    def __repr__(self):
        return 'Sphere(%s,%s)' % (repr(self.center), repr(self.radius))

    def intersectionParameter(self, ray):
        co = self.center - ray.origin
        v = np.dot(co, ray.direction)

        discriminant = v*v - np.dot(co, co) + self.radius * self.radius
        if discriminant < 0:
            return None
        else:
            return v - math.sqrt(discriminant)

    def normalAt(self, p):
        return normalize(p - self.center)

    def colorAt(self, obj, ray):
        return phong(obj, ray)

# Plane-Klasse aus Skript
class Plane(object):
    def __init__(self, point, normal, mat):
        self.point = point  # point
        self.normal = normalize(normal)  # vector
        self.color = mat
        self.reflection = 0.5

    def __repr__(self):
        return 'Plane(%s,%s)' % (repr(self.point), repr(self.normal))

    def intersectionParameter(self, ray):
        op = ray.origin - self.point
        a = np.dot(op, self.normal)
        b = np.dot(ray.direction, self.normal)
        if b < 0:
            return -a/b
        else:
            return None

    def normalAt(self, p):
        return self.normal

    def colorAt(self, obj, a):
        return phong(obj, ray)

# Triangle-Klasse aus Skript
class Triangle(object):
    def __init__(self, a, b, c, color):
        self.a = a  # point
        self.b = b  # point
        self.c = c  # point
        self.u = self.b - self.a  # direction vector
        self.v = self.c - self.a  # direction vector
        self.color = color
        self.reflection = 0.2

    def __repr__(self):
        return 'Triangle(%s,%s,%s)' % (repr(self.a), repr(self.b), repr(self.c))

    def intersectionParameter(self, ray):
        w = ray.origin - self.a
        dv = np.cross(ray.direction, self.v)
        dvu = np.dot(dv, self.u)
        if dvu == 0.0:
            return None
        wu = np.cross(w, self.u)
        r = np.dot(dv, w) / dvu
        s = np.dot(wu, ray.direction) / dvu
        if 0<=r and r<=1 and 0<=s and s<=1 and r+s<=1:
            return np.dot(wu, self.v) / dvu
        else:
            return None

    def normalAt(self, p):
        return normalize(np.cross(self.u, self.v))

    def colorAt(self, obj, a):
        return phong(obj, ray)

# Kamera-Klasse aus Skript
class Camera(object):
    def __init__(self, screenWidth, screenHeight):
        # Kamera
        self.c = np.array([0, 3, 0])
        self.e = np.array([0, 1.8, -10])
        self.up = np.array([0, -1, 0])

        #Koordinatensystem
        self.f = normalize(self.c - self.e)
        self.s = normalize(np.cross(self.f, self.up))
        self.u = np.cross(self.s, self.f)

        self.aspectRatio = screenWidth/screenHeight

        #Bild
        self.alpha = np.deg2rad(45)/2
        self.height = 2 * np.tan(self.alpha)
        self.width = self.aspectRatio * self.height

        #Pixel
        self.pixelWidth = self.width / screenWidth
        self.pixelHeight = self.height / screenHeight

    def calcRay(self, x, y):
        xcomp = self.s * (x*self.pixelWidth - self.width/2)
        ycomp = self.u * (y*self.pixelHeight - self.height/2)
        return Ray(self.e, self.f + xcomp + ycomp)

# CheckerboardMaterial aus Skript
class CheckerboardMaterial(object):
    def __init__(self):
        self.baseColor = np.array([255, 255, 255])
        self.otherColor = np.array([0, 0, 0])
        self.ambientCoefficient = 1.0
        self.diffuseCoefficient = 0.6
        self.specularCoefficient = 0.2
        self.checkSize = 1

    def baseColorAt(self, p):
        p = p * (1.0 / self.checkSize)
        if (int(abs(p[0]) + 0.5) + int(abs(p[1]) + 0.5) + int(abs(p[2])+ 0.5)) %2:
            return self.otherColor
        return self.baseColor

# Funktion berechnet den reflektierten Strahl
def computeReflectedRay(obj, ray):
    schnittpunkt = ray.pointAtParameter(obj.intersectionParameter(ray))
    normSchnitt = obj.normalAt(schnittpunkt)
    refRayVec = ray.direction - 2 * np.dot(normSchnitt, ray.direction) * normSchnitt
    return Ray(schnittpunkt, refRayVec)

# Funktion berechnet die Farbe des Schattens
def shade(level, hitPointData):
    obj = hitPointData[0]
    ray = hitPointData[1]
    directColor = phong(obj, ray)
    reflectedRay = computeReflectedRay(obj, ray)
    reflectedColor = traceRay(level+1, reflectedRay)

    if reflectedColor is None:
        return directColor
    color = directColor + np.array(reflectedColor)*obj.reflection
    return (int(color[0]), int(color[1]), int(color[2]))

# Funktion berechnet benoetigte hitpointdata für traceray-Funktion
def intersect(level, ray, maxlevel):
    obj = None
    maxdist = float('inf')
    for object in objectlist:
        hitdist = object.intersectionParameter(ray)
        if hitdist is not None and 0.001 < hitdist < maxdist:
            maxdist = hitdist
            obj = object
    if obj is None:
        return None
    return [obj, ray]

# Funktion berechnet rekursiv den Farbwert der einzelnen Pixel
def traceRay(level, ray):
    maxlevel = 3
    hitPointData = intersect(level, ray, maxlevel)
    if level < maxlevel:
        if hitPointData:
            return shade(level, hitPointData)
        return BACKGROUND_COLOR

# Funktion fürs Threading
def picBerechnung(x):
    #for x in range(0, breite):
    for y in range(imageHeight):
        ray = cam.calcRay(x, y)
        color = BACKGROUND_COLOR
        maxdist = float('inf')
        for object in objectlist:
            hitdist = object.intersectionParameter(ray)
            if hitdist:
                maxdist = hitdist
                color = traceRay(0, ray)
    image.putpixel((x, y), color)


# Funktion fürs Processing -> nicht einsatzbereit
def processing(image, breiteStart, breiteEnd, hoeheStart, hoeheEnd):
    for x in range(breiteStart, breiteEnd):
        for y in range(hoeheStart, hoeheEnd):
            ray = cam.calcRay(x, y)
            color = BACKGROUND_COLOR
            maxdist = float('inf')
            for object in objectlist:
                hitdist = object.intersectionParameter(ray)
                if hitdist:
                    maxdist = hitdist
                    color = traceRay(0, ray)
        image.putpixel((x, y), color)
    return image


if __name__ == '__main__':
    print("Hello")

    imageWidth = 150
    imageHeight = 150
    cam = Camera(imageWidth, imageHeight)

    image = Image.new("RGB", (imageWidth, imageHeight))
    # Colours
    black = np.array([0, 0, 0])
    white = np.array([255, 255, 255])
    red = np.array([255, 0, 0])
    green = np.array([0, 255, 0])
    blue = np.array([0, 0, 255])
    yellow = np.array([255, 255, 0])
    # grey = np.array([180, 180, 180])
    BACKGROUND_COLOR = (0, 0, 0)
    checker = CheckerboardMaterial()
    objectlist = []
    vlist = []
    flist = []

    sq = "nein"

    if sq == "ja":
        sqirrel = open("RayTracer/squirrel_aligned_lowres.obj.txt")

        for x in sqirrel:
            if x.startswith("v"):
                koord = x.split()
                vlist.append(np.array([float(koord[1]), float(koord[2]), float(koord[3])]))
            if x.startswith("f"):
                trikoord = x.split()
                flist.append(np.array([int(trikoord[1]), int(trikoord[2]), int(trikoord[3])]))

        for e in flist:
            triangle = Triangle(vlist[e[0]-1], vlist[e[1]-1], vlist[e[2]-1], yellow)
            objectlist.append(triangle)

    else:

        sph1 = Sphere(np.array([3, -1, 20]), 2, red)
        sph2 = Sphere(np.array([-3, -1, 20]), 2, green)
        sph3 = Sphere(np.array([0, 4, 20]), 2, blue)
        ground = Plane(np.array([0, -4, 0]), np.array([0, 1, 0]), checker)
        triangle = Triangle(np.array([0, 4, 20]), np.array([-3, -1, 20]), np.array([3, -1, 20]), yellow)

        objectlist.append(ground)
        objectlist.append(triangle)
        objectlist.append(sph1)
        objectlist.append(sph2)
        objectlist.append(sph3)

    # Farbberechnung der Pixel vor Schatten und Reflexion
    # for x in tqdm.tqdm(range(imageWidth)):
    #     for y in range(imageHeight):
    #         ray = cam.calcRay(x, y)
    #         maxdist = float('inf')
    #        # color = BACKGROUND_COLOR
    #         for object in objectlist:
    #             hitdist = object.intersectionParameter(ray)  # berechnet schnittpunkt
    #             if hitdist:
    #                 if hitdist < maxdist:
    #                     maxdist = hitdist
    #                     color = object.colorAt(object, ray)
    #         image.putpixel((x, y), color)


    #processing FUNKTIONIERT NICHT
    # tasks = [] #Prozesse

    # for i in range(4): #imageWidth oder liste mit bildabschnitten?
    #     if i == 0:
    #         breiteStart = 0
    #         breiteEnd = imageWidth/ 2
    #         hoeheStart = 0
    #         hoeheEnd = imageHeight / 2
    #     if i == 1:
    #         breiteStart = imageWidth / 2
    #         breiteEnd = imageWidth
    #         hoeheStart = 0
    #         hoeheEnd = imageHeight / 2
    #     if i == 2:
    #         breiteStart = 0
    #         breiteEnd = imageWidth / 2
    #         hoeheStart = imageHeight / 2
    #         hoeheEnd = imageHeight
    #     if i == 3:
    #         breiteStart = imageWidth / 2
    #         breiteEnd = imageWidth
    #         hoeheStart = imageHeight / 2
    #         hoeheEnd = imageHeight
    #     t = multiprocessing.Process(target=processing, args=(image, breiteStart, breiteEnd, hoeheStart, hoeheEnd)) #args für picBerechnung
    #     tasks.append(t)
    #     t.start()
    #
    # print("warte auf ergebnis")
    # for i in range(4):
    #     tasks[i].join()


    # #threading FUNKTIONIERT NICHT
    # threads = []
    # for x in range(imageWidth):
    #     t = threading.Thread(target=picBerechnung, args=(x,))
    #     threads.append(t)
    #     t.start()
    #
    # print("warte auf ergebnis")
    # for i in range(imageWidth):
    #     threads[i].join()

    # ursprüngliche Schleife zum Bildberechnen
    for x in tqdm.tqdm(range(imageWidth)):
        for y in range(imageHeight):
            ray = cam.calcRay(x, y)
            color = BACKGROUND_COLOR
            maxdist = float('inf')
            for object in objectlist:
                hitdist = object.intersectionParameter(ray)
                if hitdist:
                    maxdist = hitdist
                    color = traceRay(0, ray)
            image.putpixel((x, y), color)

    image.save("testing.png", "PNG")

image.show("testing.png")


