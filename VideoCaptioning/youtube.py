def preprocess_data(description_filename, mapping_filename, output_filename):
    with open(description_filename, 'r', encoding='utf8') as desc_file, \
        open(mapping_filename, encoding='utf8') as mapping_file, \
        open(output_filename, 'w+', encoding='utf8') as output:
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
            print(youtube_map[url] +' ' + sentence, file = output)

def process_dict(sentence_filename, diction_filename, output_filename):
    with open(sentence_filename, encoding='utf8') as sentence_file, \
        open(diction_filename, 'w+', encoding='utf8') as diction_output, \
        open(output_filename, 'w+', encoding='utf8') as output:
        dictionary = {}
        dictionary['<begin>'] = 0
        dictionary['<end>'] = 1
        dictionary['<padding>'] = 2
        dict_list = ['<begin>', '<end>', '<padding>']
        words_list = []
        max_length = 0
        for line in sentence_file.readlines():
            words = line.split(' ')
            words_list.append(words)
            max_length = max(max_length, len(words))
        max_length += 2
        print('The max length of sentence is %d. ' % max_length)
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
        for word in dict_list:
            print(word, file = diction_output)
    return dictionary, dict_list