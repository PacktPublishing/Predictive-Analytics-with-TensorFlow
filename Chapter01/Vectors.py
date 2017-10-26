import numpy as np
from vectors import Point, Vector
import functools

v1 = Vector(1, 2, 3)
v2 = Vector(10, 20, 30)
print(v1.add(10))
print(v1.sum(v2)) # displays <1 22 33>

print(v1.magnitude())

#We can multiply a vector by a real number.

print(v2.multiply(4)) #=> Vector(4.0, 8.0, 12.0)
print(v1.dot(v2))
print(v1.dot(v2, 180))
print(v1.cross(v2))

print(v1.angle(v2))

print(v1.parallel(v2))
print(v1.perpendicular(v2))
#print(v1.non_parallel(v2))
