
def printSentData(vocab, d):
    sh = d.shape
    for i in range(sh[1]):
        print(i, end=" >> ")
        for j in range(sh[0]):
            k = d[j][i].cpu().numpy()[()]
            print(vocab.itos[k], end=" ")
        print()


def printSentTarget(vocab, d, batchsize = 20):
    print("\n = ================ TARGET ") #700, 35x20, 20x35
    count = 0
    sent_length = d.shape[0] // batchsize
    for i in range(batchsize):
        print(i, end=" >> ")
        for j in range(sent_length):
            k = d[j*batchsize+i].cpu().numpy()[()]
            print(vocab.itos[k], end=" ")
        print()
    print()



def printSentOutput(vocab, dd, batchsize = 20):
    # print("\n = ================ TARGET ")  # 700, 35x20, 20x35
    count = 0
    output = dd.cpu().detach().numpy()
    d = output.argmax(axis=-1)
    print("\n = ================ OUTPUT ") #700, 35x20, 20x35
    count = 0
    sent_length = dd.shape[0] // batchsize
    for i in range(batchsize):
        print(i, end=" >> ")
        for j in range(sent_length):
            k = d[j*batchsize+i]
            print(vocab.itos[k], end=" ")
        print()
    print()

