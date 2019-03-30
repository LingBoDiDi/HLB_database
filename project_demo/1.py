#!/usr/bin/python
import numpy as np

def to_word():
    predict = [1,2,3,4,5]
    predict = predict[0]
    predict /= np.sum(predict)
    print(predict)
    #sample = np.random.choice(np.arange(len(predict)), p=predict)
    #if sample > len(vocabs):
        #return vocabs[-1]
    #else:
        #return vocabs[sample]

if __name__ == '__main__':
    to_word()