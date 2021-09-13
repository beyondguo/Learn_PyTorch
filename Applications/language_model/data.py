import os
import torch

class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []  # idx2word用一个list即可

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus:
    def __init__(self, path, lang='en'):
        self.lang = lang
        # if lang == 'zh':
        #     import jieba
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, file_path):
        assert os.path.exists(file_path)
        # Add words to dict:
        with open(file_path, 'r', encoding='utf8') as f:
            for line in f:
                if self.lang == 'zh':
                    words = [w for w in line] + ['<eos>']  # 先直接按照字来分
                else:  # 默认英语
                    words = line.split(' ') + ['<eos>']  # 先用空格分词，然后添加 end_of_sentence 符号
                for w in words:
                    self.dictionary.add_word(w)

        with open(file_path, 'r', encoding='utf8') as f:
            idss = []
            for line in f:
                if self.lang == 'zh':
                    words = [w for w in line] + ['<eos>']  # 先直接按照字来分
                else:  # 默认英语
                    words = line.split(' ') + ['<eos>']  # 先用空格分词，然后添加 end_of_sentence 符号
                ids = [self.dictionary.word2idx[w] for w in words]
                idss.append(torch.tensor(ids, dtype=torch.int64))
            return torch.cat(idss)  # 最后是类似这种的东西： tensor([0, 1, 0,  ..., 1, 0, 1])


if __name__ == '__main__':
    # c = Corpus('../../data/wikitext-2')
    c = Corpus('../../data/三国small', lang='zh')
    print(c.train.shape)
    print(c.valid.shape)
    print(c.test.shape)
    print(len(c.dictionary.word2idx))
