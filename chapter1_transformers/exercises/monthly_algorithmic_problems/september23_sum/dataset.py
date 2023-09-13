import torch as t
from torch.utils.data import Dataset



class SumDataset(Dataset):

    def __init__(self, size: int, num_digits: int, seed: int = 42):
        '''
        We create our non-palindromic examples via the following process (for each sequence):

            1. Generate a random seq of length N/2
            2. Generate another random seq of length N/2, by randomly changing X values of the previous
               seq, where X is some random integer between 0 and N/2 inclusive.
            3. Concatenate the two sequences (flipping the second one)
        
        This makes sure we have a good variety of palindromic numbers, including quite a lot with only
        one number flipped (otherwise it would be too easy for the model to distinguish).
        '''
        self.vocab = [str(i) for i in range(10)] + ["+", "="]
        self.size = size
        self.num_digits = num_digits
        t.manual_seed(seed)  # for reproducible results

        # Generate our sequences, and labels
        a_as_tokens = t.concat([
            t.randint(low=0, high=5, size=(size, 1)),
            t.randint(low=0, high=10, size=(size, num_digits-1)),
        ], dim=1)
        a_as_int = t.sum(a_as_tokens * 10**t.arange(num_digits).flip(0), dim=-1)

        b_as_tokens = t.concat([
            t.randint(low=0, high=5, size=(size, 1)),
            t.randint(low=0, high=10, size=(size, num_digits-1)),
        ], dim=1)
        b_as_int = t.sum(b_as_tokens * 10**t.arange(num_digits).flip(0), dim=-1)

        sum_as_int = a_as_int + b_as_int
        sum_as_tokens = t.stack([
            sum_as_int // 10**i % 10
            for i in reversed(range(num_digits))
        ], dim=1)
        
        self.toks = t.concat([
            a_as_tokens,
            t.full((size, 1), self.vocab.index("+")),
            b_as_tokens,
            t.full((size, 1), self.vocab.index("=")),
            sum_as_tokens,
        ], dim=1)

        self.str_toks = [
            [self.vocab[tok] for tok in toks]
            for toks in self.toks
        ]

    def __getitem__(self, index):
        return self.toks[index]

    def __len__(self):
        return self.size

    def to(self, device: str):
        self.toks = self.toks.to(device)
        return self



