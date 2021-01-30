

kernprof -l main.py                     # will generate main.py.lprof
python3 -m line_profiler main.py.lprof  # will take input main.py.lprof and print following:

Total time: 4.22593 s
File: main.py
Function: loop at line 2

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
     2                                           @profile
     3                                           def loop():
     4         1          1.0      1.0      0.0      x = 0
     5  10000001    2038555.0      0.2     48.2      for i in range(10000000):
     6  10000000    2187372.0      0.2     51.8          x = x + 5
     7         1          0.0      0.0      0.0      return x

Total time: 0.004229 s
File: main.py
Function: loop1 at line 9

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
     9                                           @profile
    10                                           def loop1():
    11         1          0.0      0.0      0.0      x = 0
    12     10001       2040.0      0.2     48.2      for i in range(10000):
    13     10000       2189.0      0.2     51.8          x = x + 5
    14         1          0.0      0.0      0.0      return x

