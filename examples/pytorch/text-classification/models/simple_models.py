from torch import nn, hstack
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import CharacterTokenizer, BasicTokenizer

class SimpleTokenizer:
    def __init__(self):
        self.model_max_length = 4
        self.vocab = {'PAD': 0, 'a': 1, 'b': 2, 'c': 3}

    def __call__(self, *args, **kwargs):
        premises = list(map(lambda x: [self.vocab[x]], args[0]))
        hypothesis = []
        is_train = False
        for h in args[1]:
            if len(h) == 2:
                is_train = True
            hypothesis.append([self.vocab[k] for k in h])
        if is_train:
            hypothesis = list(map(lambda x: x + [0] if len(x) == 1 else x, hypothesis))

        return {'premise': premises, 'hypothesis': hypothesis}

class SimpleMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers, out_dim):
        super(SimpleMLP, self).__init__()
        self.embeddings = nn.Embedding(4, in_dim, padding_idx=0)
        self.cat_res = nn.Identity()
        layers = []
        assert n_layers > 0
        # first layer receives concatenation of (premise, hypothesis)
        next_in = in_dim * 2
        for i in range(n_layers - 1):
            layers.append(nn.Linear(next_in, hidden_dim))
            layers.append(nn.ReLU())
            next_in = hidden_dim
        layers.append(nn.Linear(next_in, out_dim))
        self.classifier = nn.Sequential(*layers)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, premise, hypothesis, labels):
        premise = self.embeddings(premise).squeeze()
        hypothesis = self.embeddings(hypothesis).sum(dim=1).squeeze()
        clf_in = hstack([premise, hypothesis])
        clf_in = self.cat_res(clf_in)
        logits = self.classifier(clf_in)
        loss = self.loss_fn(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)

