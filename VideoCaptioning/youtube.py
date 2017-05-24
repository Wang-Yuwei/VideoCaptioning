from utilities import checked_mkdirs

def preprocess_data(path):
    checked_mkdirs(path + '/result')
    description_filename = path + '/video-descriptions.csv'
    mapping_filename = path + '/youtube_mapping.txt'
    train_output_filename = path + '/result/train-result.txt'
    test_output_filename = path + '/result/test-result.txt'
    train_filename = path + '/train.txt'
    train_list = []
    with open(train_filename) as train:
        train_list = list(map(lambda x: x.strip(' \t\n\r'), train.readlines()))
    with open(description_filename, 'r', encoding='utf8') as desc_file, \
        open(mapping_filename, encoding='utf8') as mapping_file, \
        open(train_output_filename, 'w+', encoding='utf8') as train_output, \
        open(test_output_filename, 'w+', encoding='utf8') as test_output:
        youtube_map = {}
        for line in mapping_file.readlines():
            [url, name] = line.split(' ')
            youtube_map[url] = name.strip(' \t\n\r')
        for line in desc_file.readlines():
            fields = line.split(',')
            url = '%s_%s_%s' % (fields[0], fields[1], fields[2])
            sentence = fields[-1].strip(' \t\n\r')[:-1]
            if not url in youtube_map or len(sentence.split(' ')) > 20:
                continue
            name = youtube_map[url]
            if name in train_list:
                print(youtube_map[url] +' ' + sentence, file = train_output)
            else:
                print(youtube_map[url] +' ' + sentence, file = test_output)

def process_dict(path):
    train_filename = path + '/result/train-result.txt'
    test_filename = path + '/result/test-result.txt'
    diction_filename = path + '/result/dict.txt'
    train_output_filename = path + '/result/train-index.txt'
    test_output_filename = path + '/result/test-index.txt'
    with open(train_filename, encoding='utf8') as train_file, \
        open(test_filename, encoding='utf8') as test_file, \
        open(diction_filename, 'w+', encoding='utf8') as diction_output, \
        open(train_output_filename, 'w+', encoding='utf8') as train_output, \
        open(test_output_filename, 'w+', encoding='utf8') as test_output:
        dictionary = {}
        dictionary['<begin>'] = 0
        dictionary['<end>'] = 1
        dictionary['<padding>'] = 2
        dict_list = ['<begin>', '<end>', '<padding>']
        words_list = []
        max_length = 0
        def process_word(input):
            nonlocal max_length
            nonlocal dict_list
            nonlocal dictionary
            words_list = []
            for line in input.readlines():
                words = line.split(' ')
                words = list(filter(lambda x: x != '',
                                    map(lambda x: x.strip(' \t\n\r.!'), words)))
                words_list.append(words)
                max_length = max(max_length, len(words))
            return words_list
        def process_input(words_list, output):
            nonlocal max_length
            nonlocal dict_list
            nonlocal dictionary
            for words in words_list:
                id = words[0]
                words = words[1: ]
                row = [id, 0]
                if len(words) < 2:
                    continue
                for word in words:
                    if word not in dictionary:
                        dict_list.append(word)
                        dictionary[word] = len(dict_list) - 1
                    row.append(dictionary[word])
                row.append(1)
                row = row + [2] * (max_length - len(row))
                print(' '.join(map(str, row)), file = output)
        train_words = process_word(train_file)
        test_words = process_word(test_file)
        max_length += 2
        print('The max length of sentence is %d. ' % max_length)    
        process_input(train_words, train_output)
        process_input(test_words, test_output)
        for word in dict_list:
            print(word, file = diction_output)
    return dictionary, dict_list