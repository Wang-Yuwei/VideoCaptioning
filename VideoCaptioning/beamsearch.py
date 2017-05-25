import numpy as np
from operator import itemgetter
import math
class BeamSearch:
    def __init__(self):
        self.BOS = 0
        self.EOS = 1
        self.sentence_pool = []
        self.cost_sentence = []
        self.word_sequence = [{'seq': [self.BOS], 'state': None}]
        self.cost_sequence = [0]
        self.Nsentence = 5
        self.Nsequence = 5

    def next(self):
        if(len(self.cost_sequence) == 0):
            return None, None, None
        nextId = np.argmin(self.cost_sequence)
        word = self.word_sequence[nextId]['seq'][-1]
        state = self.word_sequence[nextId]['state']
        return nextId, word, state

    def add(self, id, next_word, cost, state):
        for i in range(len(next_word)):
            new_sequence = dict()
            new_sequence['seq'] = self.word_sequence[id]['seq'] + [next_word[i]]
            new_sequence['state'] = state
            new_cost = self.cost_sequence[id] - math.log(cost[i])
            if (next_word[i] == self.EOS):
                self.add_pool(new_sequence, new_cost)
            else:
                self.word_sequence.append(new_sequence)
                self.cost_sequence.append(new_cost)
        self.word_sequence.pop(id)
        self.cost_sequence.pop(id)
        self.update()

    def update(self):
        c, w= zip(*sorted(zip(self.cost_sequence,self.word_sequence), key=itemgetter(0), reverse=True))
        self.word_sequence = list(w)
        self.cost_sequence = list(c)
        startIdx = len(self.word_sequence) - self.Nsequence
        if(startIdx > 0):
            self.word_sequence = self.word_sequence[startIdx:]
            self.cost_sequence = self.cost_sequence[startIdx:]
        if (len(self.sentence_pool) == self.Nsentence):
            low_cost = np.min(self.cost_sentence)
            while (len(self.cost_sequence) > 0 and self.cost_sequence[0] > low_cost):
                self.word_sequence.pop(0)
                self.cost_sequence.pop(0)

    def add_pool(self, sentence, cost):
        if(len(self.sentence_pool) == self.Nsentence):
            id = np.argmax(self.cost_sentence)
            max_cost = self.cost_sentence[id]
            if(cost < max_cost):
                self.sentence_pool[id] = sentence
                self.cost_sentence[id] = cost
        else:
            self.sentence_pool.append(sentence)
            self.cost_sentence.append(cost)

    def print(self):
        print("sequence:")
        for i in range(len(self.word_sequence)):
            print("%s %s" %(self.word_sequence[i]['seq'], self.cost_sequence[i].__str__()))
        print("pool:")
        for i in range(len(self.sentence_pool)):
            print("%s %s" %(self.sentence_pool[i], self.cost_sentence[i].__str__()))
