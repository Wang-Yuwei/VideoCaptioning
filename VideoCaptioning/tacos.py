#from utilities import checked_mkdirs

def generate_dict(paragraphs):
    word2idx = {'<start>': 0, '<end>': 1, '<padding>': 2}
    idx2word = ['<start>', '<end>', '<padding>']
    for name in paragraphs:
        for paragraph in paragraphs[name]:
            for sentence in paragraph:
                for word in sentence:
                    if not word in word2idx:
                        word2idx[word] = len(word2idx)
                        idx2word.append(word)
    return word2idx, idx2word

def save_dict(filename, dict):
    with open(filename, 'w+') as f:
        for word in dict:
            print(word, file = f)

def get_max_size(paragraphs):
    max_sentence_number = 0
    max_word_number = 0
    sentence_number = 0
    for name in paragraphs:
        sentence_number += len(paragraphs[name])
        for paragraph in paragraphs[name]:
            max_sentence_number = max(max_sentence_number, len(paragraph))
            for sentence in paragraph:
                max_word_number = max(max_word_number, len(sentence))
    print(sentence_number)
    return max_sentence_number, max_word_number

def save_paragraphs(filename, paragraphs):
    with open(filename, 'w+') as f:
        for name in paragraphs:
            for paragraph in paragraphs[name]:
                print(name, file = f)
                for sentence in paragraph:
                    print(' '.join(sentence), file = f)

def print_sentence(file, sentence, max_word_number, word2idx):
    print(0, end = '', file = file)
    for word in sentence:
        print(' ' + str(word2idx[word]), end = '', file = file)
    print(' 1', end = '', file = file)
    for i in range(max_word_number - len(sentence)):
        print(' 2', end = '', file = file)
    print('', file = file)

def save_indices(filename, paragraphs, max_sentence_number, max_word_number, word2idx):
    with open(filename, 'w+') as file:
        print(max_sentence_number, max_word_number, file = file)
        for name in paragraphs:
            for paragraph in paragraphs[name]:
                print(name, file = file)
                for sentence in paragraph:
                    print_sentence(file, sentence, max_word_number, word2idx)
                print_sentence(file, [], max_word_number, word2idx)
                for i in range(max_sentence_number - len(paragraph)):
                    print(' '.join(['2'] * (max_word_number + 2)), file = file)

def preprocess(path):
#    checked_mkdirs(path + '/result')
    with open(path + '/annosShort-processed.csv') as f:
        result = {}
        for line in f.readlines():
            words = line.split('\t')
            if words[0] not in result:
                result[words[0]] = []
            result[words[0]].append((int(words[4]), words[6]))
    paragraphs = {}
    for name in result:
        paragraphs[name] = []
        temp = {}
        for sentence in result[name]:
            if sentence[0] not in temp:
                temp[sentence[0]] = []
            temp[sentence[0]].append(sentence[1].split(' '))
        for workerid in temp:
            if len(temp[workerid]) > 8:
                continue
            if max(map(len, temp[workerid])) > 15:
                continue
            paragraphs[name].append(temp[workerid])
    word2idx, idx2word = generate_dict(paragraphs)
    save_dict(path + '/result/dict.txt', idx2word)
    max_sentence_number, max_word_number = get_max_size(paragraphs)
    save_paragraphs(path + '/result/paragraphs.txt', paragraphs)
    save_indices(path + '/result/indice.txt', paragraphs, max_sentence_number,
                 max_word_number, word2idx)