import numpy as np              # regular ol' numpy
from trax import layers as tl   # core building block
from trax import shapes         # data signatures: dimensionality and type
from trax import fastmath       # uses jax, offers numpy on steroids

serialLayer = tl.Serial
selectLayer = tl.Select

print("===================1 Layers in series=======================")
addLayer = tl.Fn("Addition", lambda x,y: x + y)
mulLayer = tl.Fn("Multiplication", lambda x,y: x * y)

x1 = np.array([3])
x2 = np.array([4])
x3 = np.array([15])
x4 = np.array([3])
y1 = addLayer((x1, x2))
y2 = mulLayer((y1, x3))
y3 = addLayer((y2, x4))
print("y1 :", y1, "y2 :", y2, "y3 :", y3)

serial = tl.Serial(addLayer, mulLayer, addLayer)
print("-- Serial Model --")
print("name :", addLayer.name, "Inputs", addLayer.n_in, "out", addLayer.n_out, addLayer)
print("name :", mulLayer.name, "Inputs", mulLayer.n_in, "out", mulLayer.n_out, mulLayer)
print("name :", serial.name, "Inputs", serial.n_in, "out", serial.n_out, serial)

xa = (x1, x2, x3, x4)  
serial.init(shapes.signature(xa))  
ya = serial(xa)
print("xa, ya :", xa, ya)

print("===================2 Select Layer=======================")
serial = tl.Serial(tl.Select([0, 1, 0, 1]), addLayer, mulLayer, addLayer)
x = (np.array([3]), np.array([4]))  
serial.init(shapes.signature(x))  
y = serial(x)
print("name :", serial.name, "Inputs", serial.n_in, "out", serial.n_out, serial)
print("x :", x, "y :", y)

serial = tl.Serial(tl.Select([0, 1, 0, 1]), addLayer, tl.Select([0], n_in=2), mulLayer)
x = (np.array([3]), np.array([4]))  
serial.init(shapes.signature(x)) 
y = serial(x)
print("name :", serial.name, "Inputs", serial.n_in, "out", serial.n_out, serial)
print("(x1, x2) :", x)
print("x :", x)
print("y :", y)

print("===================3 Residual Layer (add) =======================")
serial = tl.Serial(tl.Select([0, 1, 0, 1]), tl.Residual(addLayer))
x = (np.array([3]), np.array([4]))
serial.init(shapes.signature(x)) 
y = serial(x)
print("name :", serial.name, "Inputs", serial.n_in, "out", serial.n_out, serial)
print("(x1, x2) :", x)
print("x :", x)
print("y :", y)

print("===================3 Residual Layer (mul) =======================")
serial = tl.Serial(tl.Select([0, 1, 0, 1]), tl.Residual(mulLayer))
x = (np.array([3]), np.array([4]))
serial.init(shapes.signature(x)) 
y = serial(x)
print("name :", serial.name, "Inputs", serial.n_in, "out", serial.n_out, serial)
print("(x1, x2) :", x)
print("x :", x)
print("y :", y)

