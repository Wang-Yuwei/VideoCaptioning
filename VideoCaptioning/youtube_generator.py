import numpy

class YouTubeGenerator(object):
    def __init__(self, words_filename, folder_name, batch_size):
        self.video_list = []
        self.words_list = []
        with open(words_filename) as f:
            for line in f.readlines():
                words = line.split(' ')
                self.video_list.append(words[0])
                self.words_list.append(list(map(int,words[1:])))
        self.max_sentence_length = len(self.words_list[0])
        self.folder_name = folder_name
        self.batch_size = batch_size
        self.words_number = max(map(max, self.words_list))
        self.current = 0
        self.sample_number = 2
        self.feature_shape = [50, 4096]
        #self.sample_number = len(self.video_list)

    def next(self):
        if self.current + self.batch_size >= len(self.video_list):
            self.current = 0
        sentences = numpy.zeros([self.batch_size,
                                 self.max_sentence_length, 
                                 self.words_number])
        length = numpy.zeros([self.batch_size])
        features = []
        for i in range(self.current, self.current + self.batch_size):
            features.append(numpy.load(self.folder_name + '/' + self.video_list[i] + '.npy'))
            sentence_length = 0
            for j in range(self.max_sentence_length):
                sentences[i, j, self.words_list[i][j]] = 1
                if self.words_list[i][j] != 2:
                    sentence_length += 1
            length[i] = sentence_length
        return sentences, length, numpy.stack(features)

    def next1(self):
        if self.current + self.batch_size >= len(self.video_list):
            self.current = 0
        sentences = numpy.zeros([self.batch_size,
                                 self.max_sentence_length])
        length = numpy.zeros([self.batch_size])
        features = []
        sentences = []
        for i in range(self.current, self.current + self.batch_size):
            features.append(numpy.load(self.folder_name + '/' + self.video_list[i] + '.npy'))
            sentences.append(self.words_list[i])
            sentence_length = 0
            for j in range(self.max_sentence_length):
                if self.words_list[i][j] != 2:
                    sentence_length += 1
            length[i] = sentence_length
        return sentences, length, numpy.stack(features)