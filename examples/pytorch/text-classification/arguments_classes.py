from dataclasses import dataclass, field
from typing import Optional, List

# hack, not most elegant way
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "fever": ("claim", "evidence")
}


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    # EDIT
    indices_dir: Optional[List[str]] = field(default=None, metadata={"help": "list of directories containing split "
                                                                             "indices of the dataset for evaluation (MNLI matched)"})
    mismatched_indices_dir: Optional[List[str]] = field(default=None, metadata={"help": "list of directories containing split "
                                                                             "indices of the dataset for evaluation (MNLI miss-matched)"})
    synthetic_bias_prevalence: float = field(default=0.0,
                                             metadata={'help': 'ratio of samples to be injected with bias token'})
    bias_correlation_prob: float = field(default=0.8, metadata={'help': 'ratio of bias token aligning with true label'})
    hypothesis_only: bool = field(default=False, metadata={'help': 'Use only the hypothesis (second sentence) as input'})
    claim_only: bool = field(default=False, metadata={'help': 'Use only the claim (first sentence) as input'})
    lexical_bias_model: bool = field(default=False, metadata={'help': 'Use lexical bias features'})
    remove_biased_samples_from_train: bool = field(
        default=False,
        metadata={'help': 'remove biased samples from training dataset to measure effect (one time thing)'}
    )
    select_only_biased_samples: bool = field(
        default=False,
        metadata={'help': "select biased samples from training dataset to measure effect (one time thing)"}
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                    validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."
            assert self.synthetic_bias_prevalence <= 0 or not self.hypothesis_only

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    randomly_initialize_model: bool = field(default=False, metadata={"help": "randomly initialize the weights of the main model (used to train bias "
                                                                       "models in unknown-bias settings)"})
    freeze_embeddings: bool = field(default=False, metadata={"help": "freeze embeddings of the  main model (used to train bias models in "
                                                                     "unknown-bias settings)"})

    freeze_encoder: bool = field(default=False, metadata={'help': 'freeze encoder of the main model in BERT based models'})


@dataclass
class WandbArguments:
    tags: Optional[List[str]] = field(default=None, metadata={'help': 'WANDB tags to assign to this run'})
    wandb_group: bool = field(default=False, metadata={'help': 'use run name as group name in W&B'})

