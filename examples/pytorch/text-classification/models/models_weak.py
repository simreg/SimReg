# StackedSelfAttentionEncoder
import torch
import collections
import os
import pickle
from torch import nn
import logging
from transformers import BasicTokenizer, PreTrainedTokenizer, PreTrainedModel, PretrainedConfig, WEIGHTS_NAME

logger = logging.getLogger(__name__)


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    id_to_tokens = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip('\n')
        vocab[token] = index
        id_to_tokens[index] = token
    return vocab, id_to_tokens


def load_txt_embeddings(path):
    import pandas as pd
    import csv
    words = pd.read_csv(path, sep=" ", index_col=0,
                        na_values=None, keep_default_na=False, header=None,
                        encoding="utf-8", quoting=csv.QUOTE_NONE)
    matrix = words.values
    index_to_word = list(words.index)
    word_to_index = {
        word: ind for ind, word in enumerate(index_to_word)
    }
    print("Loaded", len(index_to_word), "embeddings")
    return matrix, index_to_word, word_to_index


def extract_subset_from_glove(glove_path, dictionary, output_dir):
    import pandas as pd
    import numpy as np
    import pickle
    vocab, index_to_word = load_vocab(dictionary)
    print("Filtering", len(vocab), "embeddings.")
    matrix, _, word_to_index = load_txt_embeddings(glove_path)
    unk_word = matrix.mean(0)
    subset_matrix = np.zeros((len(vocab), matrix.shape[1])) + unk_word[None, :]
    num_unks = 0
    for index, token in index_to_word.items():
        ind = word_to_index.get(token, -1)
        if ind > -1:
            subset_matrix[index] = matrix[ind]
        else:
            num_unks += 1

    print("Filtering done, num unks", num_unks)
    with open(output_dir + "/embeddings.pkl", "wb") as f:
        pickle.dump(dict(word_to_index=vocab, embeddings=subset_matrix), f)


def load_embeddings(path):
    resource = pickle.load(open(path, 'rb'))
    word_to_index = resource['word_to_index']
    matrix = resource['embeddings']
    index_to_word = [(i, w) for w, i in word_to_index.items()]
    return matrix, word_to_index, index_to_word




def convert_examples_to_features_base(examples, max_seq_length, tokenizer):
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        assert example.text_b is not None
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = tokenizer.tokenize(example.text_b)
        if len(tokens_a) > max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
        if len(tokens_b) > max_seq_length:
            tokens_b = tokens_b[:max_seq_length]
        input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)
        input_ids_b = tokenizer.convert_tokens_to_ids(tokens_b)
        input_mask_a = [1] * len(input_ids_a) + [0] * (max_seq_length - len(input_ids_a))
        input_mask_b = [1] * len(input_ids_b) + [0] * (max_seq_length - len(input_ids_b))
        input_ids_a += [0] * (max_seq_length - len(input_ids_a))
        input_ids_b += [0] * (max_seq_length - len(input_ids_b))
        assert len(input_ids_a) == max_seq_length
        assert len(input_ids_b) == max_seq_length
        assert len(input_mask_a) == max_seq_length
        assert len(input_mask_b) == max_seq_length


    return features



class BaselineTokenizer(PreTrainedTokenizer):
    @classmethod
    def from_pretrained(cls, model_name_or_path=None, do_lower_case=False, **kwargs):
        logger.info(f"Initializing BaselineTokenizer with {model_name_or_path} do_lower_case:{do_lower_case}")
        vocab_file = kwargs.pop('vocab_file', None)
        # Load vocab from the checkpoint
        if vocab_file is None:
            assert os.path.exists(model_name_or_path)
            vocab_file = os.path.join(model_name_or_path, "vocab.txt")
        if do_lower_case:
            logger.info("Lower casing is set to True, sure you're doing the right thing?")
        return cls(vocab_file, do_lower_case=do_lower_case)

    def __init__(self, vocab_file, do_lower_case,
                 never_split=None, tokenize_chinese_chars=True):
        super().__init__(max_len=128, vocab_file=vocab_file,
                         unk_token="[UNK]", sep_token="[SEP]",
                         pad_token="[PAD]", cls_token="[CLS]",
                         mask_token="[MASK]",
                         do_lower_case=do_lower_case,
                         never_split=never_split,
                         tokenize_chinese_chars=tokenize_chinese_chars)
        self.tokenizer = BasicTokenizer(
            do_lower_case=do_lower_case, never_split=never_split,
            tokenize_chinese_chars=tokenize_chinese_chars)
        self.vocab, self.ids_to_tokens = load_vocab(vocab_file)
        logger.info("Vocabulary size: %d", len(self.vocab))

    def _convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = ' '.join(tokens).replace(' ##', '').strip()
        return out_string

    def _tokenize(self, text, **kwargs):
        return self.tokenizer.tokenize(text)

    def get_vocab(self):
        return self.vocab

    def save_vocabulary(self, vocab_path, filename_prefix=''):
        """Save the tokenizer vocabulary to a directory or file."""
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, "vocab.txt")
        else:
            vocab_file = vocab_path
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning("Saving vocabulary to {}: vocabulary indices are not consecutive."
                                   " Please check that the vocabulary is not corrupted!".format(vocab_file))
                    index = token_index
                writer.write(token + u'\n')
                index += 1
        return (vocab_file,)


class BaselineConfig(PretrainedConfig):
    def __init__(self, vocab_size_or_config_json_file=None, **kwargs):
        super().__init__(vocab_size_or_config_json_file=vocab_size_or_config_json_file, **kwargs)
        for key, value in kwargs.items():
            self.__dict__[key] = value


class BaselineModel(PreTrainedModel):
    def save_pretrained(self, save_directory, **kwargs):
        super(BaselineModel, self).save_pretrained(save_directory=save_directory, **kwargs)
        # Save class name
        with open(os.path.join(save_directory, "class.txt"), "w") as cf:
            cf.write(self.base_model_prefix + "\n")

    @classmethod
    def from_pretrained(cls, model_name_or_path, *model_args, **kwargs):
        """Model_name_or_path is the type of model to be loaded.
        """
        BASELINE_MODELS_MAP = {
            "bow": BOW,
        }

        weights_file = None
        init_class = BASELINE_MODELS_MAP.get(model_name_or_path)
        config = kwargs.pop("config", None)

        if init_class is not None:
            assert config is not None, "A config is required when initializing a model from scratch."
            vocab = load_vocab(config.vocab_file)[0]
            model = init_class(config, vocab)
        elif os.path.isdir(model_name_or_path):
            with open(os.path.join(model_name_or_path, "class.txt"), "r") as cf:
                init_class = BASELINE_MODELS_MAP.get(cf.readlines()[0].rstrip('\n'))
            # Load config file
            if config is None:
                config = BaselineConfig.from_pretrained(model_name_or_path)
            # Load vocab
            vocab, _ = load_vocab(os.path.join(model_name_or_path, "vocab.txt"))
            model = init_class(config, vocab)
            # Load weights
            weights_file = os.path.join(model_name_or_path, WEIGHTS_NAME)
            if weights_file is not None:
                # Load from a PyTorch state_dict
                state_dict = torch.load(weights_file, map_location='cpu')
                model.load_state_dict(state_dict)
                logger.info("Loaded pretrained model.")

        # Set model in evaluation mode to desactivate DropOut modules by default
        model.eval()
        if kwargs.pop("output_loading_info", False):
            return model, {}
        return model


class BOW(BaselineModel):
    config_class = BaselineConfig
    pretrained_model_archive_map = None
    load_tf_weights = None
    base_model_prefix = "bow"

    def __init__(self, config, vocab):
        super().__init__(config)
        self.config = config
        self.vocab = vocab
        self.dropout = torch.nn.Dropout(config.dropout)
        self.embedding = nn.Embedding(
            len(self.vocab), config.embedding_dim)
        self.proj = nn.Sequential(
            nn.Linear(config.embedding_dim, 512),
            nn.Tanh())
        self.classifier = nn.Sequential(
            nn.Linear(4 * 512, 200),
            nn.Tanh(),
            nn.LayerNorm(200),
            self.dropout,
            nn.Linear(200, config.num_labels),
        )
        self.init_weights()

    def init_weights(self):
        ext_embeddings, ext_word_to_index, _ = load_embeddings(
            self.config.embedding_file)
        embeddings = self.embedding.weight.data.cpu().numpy()
        word_found = 0
        for word, index in self.vocab.items():
            if word in ext_word_to_index:
                embeddings[index] = ext_embeddings[ext_word_to_index[word]]
                word_found += 1
        logger.info('Embeddings found %d / %d', word_found, len(self.vocab))
        embeddings = torch.from_numpy(embeddings).to(self.embedding.weight.device)
        self.embedding.load_state_dict({'weight': embeddings})
        self.embedding.weight.requires_grad = True

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, reduction='mean'):
        # Similarity matrix
        embeds = self.proj(self.embedding(input_ids))
        s1_mask = attention_mask.bool() & (token_type_ids == 0)
        s2_mask = attention_mask.bool() & (token_type_ids == 1)
        s1 = (embeds * s1_mask.unsqueeze(-1)).sum(1) / s1_mask.sum(-1, keepdim=True)
        s2 = (embeds * s2_mask.unsqueeze(-1)).sum(1) / s2_mask.sum(-1, keepdim=True)
        h = torch.cat((s1 * s2, torch.abs_(s1 - s2), s1, s2), 1)
        logits = self.classifier(h)
        outputs = (logits,)
        if labels is not None:
            loss = nn.CrossEntropyLoss(reduction=reduction)(logits, labels)
            outputs = (loss,) + outputs
        return outputs

