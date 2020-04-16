import numpy as np
class WordEmb:
    def __init__(self, filename, sep="\t"):
        self.word_id_map = dict()
        self.id_word_map = dict()
        self.lookup = None
        self.__load(filename, sep)

    def __load(self, filename, sep):
        index = 0
        id_emb_list = list()
        for line in open(filename):
            splits = line.rstrip().split(sep)
            if len(splits) < 10:
                continue
            word = splits[0]
            emb = [float(item) for item in splits[1:]]
            self.word_id_map[word] = index
            self.id_word_map[index] = word
            index += 1
            id_emb_list.append(emb)

        self.lookup = np.array(id_emb_list)

    def most_similarity(self, word, top_k=10):
        if word not in self.word_id_map:
            print("word not in embedding dict!")
            return list()
        v = self.lookup[self.word_id_map[word]]
        sims = np.dot(self.lookup, v)
        sort = sims.argsort()[::-1]
        sort = sort[sort > 0]
        return [(self.id_word_map[i], sims[i]) for i in sort[:top_k]]

    def answer(self, word1, word2, word3, top_k=10):
        if word1 not in self.word_id_map:
            print("word1 not in embedding dict!")
            return list()
        if word2 not in self.word_id_map:
            print("word2 not in embedding dict!")
            return list()
        if word3 not in self.word_id_map:
            print("word3 not in embedding dict!")
            return list()
        v1 = self.lookup[self.word_id_map[word1]]
        v2 = self.lookup[self.word_id_map[word2]]
        v3 = self.lookup[self.word_id_map[word3]]
        v4 = v1 - v2 + v3
        sims = np.dot(self.lookup, v4)
        sort = sims.argsort()[::-1]
        sort = sort[sort > 0]
        return [(self.id_word_map[i], sims[i]) for i in sort[:top_k]]

    def sim(self, word1, word2):
        if word1 not in self.word_id_map:
            print("word1 not in embedding dict!")
            return None
        if word2 not in self.word_id_map:
            print("word2 not in embedding dict!")
            return None
        v1 = self.lookup[self.word_id_map[word1]]
        v2 = self.lookup[self.word_id_map[word2]]
        return np.dot(v1, v2) / (np.linalg.norm(v1) * (np.linalg.norm(v2)))