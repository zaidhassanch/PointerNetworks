
# @profile
def loop():
    x = 0
    for i in range(10000000):
        x = x + 5
    return x

# @profile
def loop1():
    x = 0
    for i in range(10000):
        x = x + 5
    return x



t = loop()
print(t)
t = loop1()
print(t)