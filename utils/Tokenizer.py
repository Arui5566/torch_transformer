import json

class Tokenizer:
    def __init__(self):
        self.pad = 0
        self.bos = 1
        self.eos = 2
        self.unk = 4
        self.idx2tok = []
        self.tok2idx = {}

    def build_vocab(self, texts):
        vocab = set()
        for t in texts:
            vocab.update(t)
        vocab = sorted(list(vocab))

        self.idx2tok = ['<pad>', '<bos>', '<eos>', '<unk>'] + vocab
        self.tok2idx = {t: i for i,t in enumerate(vocab)}

    def encode(self, text):
        ids = [self.bos] + [self.tok2idx.get(ch, self.unk) for ch in text] + [self.eos]
        return ids

    def decode(self, ids):
        return ''.join([self.idx2tok[i] for i in ids if i > 3])

    def save(self, path):
        data = {
            "tok2idx": self.tok2idx,
            "idx2tok": self.idx2tok
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = Tokenizer()
        tok.tok2idx = data["tok2idx"]
        tok.idx2tok = data["idx2tok"]
        return tok
