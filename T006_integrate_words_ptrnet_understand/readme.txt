Epoch [1] loss: 0.5859552025794983
Epoch [1] loss: 0.029461374506354332
Epoch [1] loss: 0.008194823749363422
Epoch [1] loss: 0.003212432377040386
Epoch [1] loss: 0.00209496496245265

Prepare input for our Neural Net
- N lettered words and their sort order

- Generate N lettered words
x1 =['bd','cc','az']

- Sort words to get sort order
  > We have sorting logic already

- Convert letters to arrays
x = [[54, 58], [55, 55], [53, 65]]
y = [2, 0, 1]

- Convert array to letters BACK

- done

- batch support
    [
        [[98, 100], [99, 99], [97, 122]],
        [[98, 100], [99, 99], [97, 122]],
        [[98, 100], [99, 99], [97, 122]],
    ]

    [
        ['bd', 'cc', 'az'],
        ['bd', 'cc', 'az'],
        ['bd', 'cc', 'az'],


    ]
    [
      [2, 0, 1],
      [2, 0, 1],
      [2, 0, 1]
    ]


=================================================================

- Append # to words which are of less than length N



11:25 - 11:40: planning what do want to do today
  11:32 - 12:00: 1
  12:00 - 12:30: 2
  12:30 -  1:00: 3
12:07: All the above done
12:30: We have generated a word of length of our choice

1:00  - 1:15: planning if optimization is required or not



