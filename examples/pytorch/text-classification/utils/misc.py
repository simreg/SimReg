from typing import NamedTuple, Optional, Callable, Dict
from datasets import Dataset
from transformers import EvalPrediction, PretrainedConfig
from os import path
import json

ROOT_DIR = path.abspath('../../../')
DATA_DIR = path.join(ROOT_DIR, 'data')

lexical_bias_cache_mapper = {
    'mnli': {
        'train': path.join(DATA_DIR, 'mnli_bias_features_without_neg/mnli_hans_train.cache'),
        'validation_matched': path.join(DATA_DIR, 'mnli_bias_features_without_neg/mnli_hans_validation_matched.cache'),
        'validation_mismatched': path.join(DATA_DIR, 'mnli_bias_features_without_neg/mnli_hans_validation_mismatched.cache'),
        'test_matched': path.join(DATA_DIR, 'mnli_bias_features_without_neg/mnli_hans_test_matched.cache'),
        'test_mismatched': path.join(DATA_DIR, 'mnli_bias_features_without_neg/mnli_hans_test_mismatched.cache'),
        'hans': path.join(DATA_DIR, 'hans_lexical_features_dataset_without_neg/hans_lexical_features_validation.cache')
    },
    'qqp': {
        'train': path.join(DATA_DIR, 'QQP_lexical_features/qqp_lexical_features_train.cache'),
        'validation': path.join(DATA_DIR, 'QQP_lexical_features/qqp_lexical_features_validation.cache'),
        'test': path.join(DATA_DIR, 'QQP_lexical_features/qqp_lexical_features_test.cache'),
        'paws_test': path.join(DATA_DIR, 'QQP_lexical_features/paws_lexical_features_test.cache')
    }
}


class EvaluationSet(NamedTuple):
    set: Dataset
    logging_mode: str
    metrics_func: Optional[Callable[[EvalPrediction], Dict]] = None


def extract_split_name(splits_dir, split_file_name):
    splits_config_path = path.join(splits_dir, 'config.json')
    split_name = split_file_name.split(".")[0]
    if path.isfile(splits_config_path):
        pass
    elif path.isfile(path.join(splits_dir, 'validation_matched_data_extraction_config.json')):
        splits_config_path = path.join(splits_dir, 'validation_matched_data_extraction_config.json')
    elif path.isfile(path.join(splits_dir, 'validation_data_extraction_config.json')):
        splits_config_path = path.join(splits_dir, 'validation_data_extraction_config.json')
    else:
        return split_name
    with open(splits_config_path, 'r') as splits_config_f:
        splits_config = json.load(splits_config_f)
        if 'name_mapper' in splits_config and split_name in splits_config['name_mapper']:
            split_name = splits_config['name_mapper'][split_name]
    return split_name


def is_partial_input_model(config: PretrainedConfig):
    return (hasattr(config, 'hypothesis_only') and config.hypothesis_only) or (hasattr(config, 'claim_only') and config.claim_only)


