

- Pointer Network Training / inference worked for one sequence of length 8 (6 lettered words)
- Training was done in parallel, and inference sequentially.
- The sequence was:
    x2 = [
        'behead',
        'chests',
        'worsen',
        'wusses',
        'youths',
        'zanies',
        'detail',
        'femurs']
- However, when tried to train on all words, transformer did not converge

