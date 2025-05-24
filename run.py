import logging
import random
import numpy as np
import torch
import argparse
from data_utils import load_llm_explanation_dataset, CodeDataset
from transformers import AutoTokenizer
from train import Trainer
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)

    # clip stage args
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr_IF', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--save_epoch', type=int, default=1)
    parser.add_argument('--code_model_path', type=str, default='codet5')
    parser.add_argument('--nl_model_path', type=str, default='t5')
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--savepath', type=str, default='./Results')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_file', type=str, default=None)

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    # parse agrs
    args = parse_args()
    logger.info(vars(args))

    # select device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Device is %s', args.device)

    # set seed
    set_seed(args.seed)

    # 加载数据
    train_data, test_data = load_llm_explanation_dataset("llm_explanations.json", seed=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.code_model_path)

    # 构建 Dataset
    train_dataset = CodeDataset(train_data, tokenizer, max_length=args.max_length)
    test_dataset = CodeDataset(test_data, tokenizer, max_length=args.max_length)

    # CL model
    trainer = Trainer(args)
    trainer.train_classicication([train_dataset, test_dataset])