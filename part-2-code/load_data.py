import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):
    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        self.split = split
        self.datafolder = data_folder

        
        self.bos_token_id = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")

        self.encoder_inputs, self.decoder_inputs, self.decoder_targets = \
            self.process_data(data_folder, split, self.tokenizer)

            
    
    def process_data(self, data_folder, split, tokenizer):
        X_path = os.path.join(data_folder, f"{split}.nl")
        X_lines = load_lines(X_path)

        if split != "test":
            Y_path = os.path.join(data_folder, f"{split}.sql")
            Y_lines = load_lines(Y_path)
        else:
            Y_lines = None

        encoder_inputs = []
        decoder_inputs = []
        decoder_targets = []

        for i, nl in enumerate(X_lines):
            # ------------ PREPROCESS NL ------------
            nl = normalize_nl(nl)
            
            # ------------ ENCODER ------------
            enc = tokenizer(
                nl,
                truncation=True,
                max_length=256,
                padding=False
            )
            encoder_inputs.append(torch.tensor(enc["input_ids"], dtype=torch.long))

            # ------------ DECODER (TRAIN/DEV) ------------
            if split != "test":
                sql = Y_lines[i]
    
                # SQL preprocessing: canonicalize + delexicalize
                sql = canonicalize_sql(sql)
                
    
                dec = tokenizer(
                    sql,
                    truncation=True,
                    max_length=256,
                    padding=False
                )
                sql_ids = dec["input_ids"]
    
                # BOS + SQL tokens
                dec_input_ids = [self.bos_token_id] + sql_ids
    
                # SQL tokens + EOS
                dec_target_ids = sql_ids + [tokenizer.eos_token_id]
    
                decoder_inputs.append(torch.tensor(dec_input_ids, dtype=torch.long))
                decoder_targets.append(torch.tensor(dec_target_ids, dtype=torch.long))

        return encoder_inputs, decoder_inputs, decoder_targets

    def __len__(self):
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        if self.split == "test":
            return self.encoder_inputs[idx]
        else:
            return (
                self.encoder_inputs[idx],
                self.decoder_inputs[idx],
                self.decoder_targets[idx]
            )



# ----------------------- PREPROCESSING HELPERS -----------------------

def normalize_nl(text):
    """Basic NL normalization: lowercase, strip, normalize spaces."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def canonicalize_sql(sql):
    """I found that for preprocessing SQL data it is good to implement an SQL canonicalizer: normalize spacing, uppercase keywords, sort conditions."""
    # Uppercase SQL keywords
    keywords = [
        "select", "from", "join", "on", "where", "group by", "order by", 
        "having", "limit", "and", "or"
    ]
    for kw in keywords:
        sql = re.sub(rf"\b{kw}\b", kw.upper(), sql, flags=re.IGNORECASE)

    # Normalize whitespace
    sql = re.sub(r"\s+", " ", sql).strip()

    # OPTIONAL: sort WHERE conditions for stability
    if "WHERE" in sql:
        before, after = sql.split("WHERE", 1)
        conds = [c.strip() for c in after.split("AND")]
        conds = sorted(conds)
        sql = before + "WHERE " + " AND ".join(conds)

    return sql


# ---------------------------------------------


def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encs, dec_ins, dec_tgts = zip(*batch)

    # pad encoder input idss
    enc_padded = pad_sequence(encs, batch_first=True, padding_value=PAD_IDX)
    enc_mask = (enc_padded != PAD_IDX).long()

    # pad decoder input idss and targets
    dec_in_padded = pad_sequence(dec_ins, batch_first=True, padding_value=PAD_IDX)
    dec_tgt_padded = pad_sequence(dec_tgts, batch_first=True, padding_value=PAD_IDX)

    # first token for generation
    initial_decoder = dec_in_padded[:, 0].unsqueeze(1)

    return enc_padded, enc_mask, dec_in_padded, dec_tgt_padded, initial_decoder
    

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    
    encs = batch

    enc_padded = pad_sequence(encs, batch_first=True, padding_value=PAD_IDX)
    enc_mask = (enc_padded != PAD_IDX).long()

    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    bos_token = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    beg_seq_batch = torch.tensor([[bos_token]] * len(batch), dtype=torch.long)

    return enc_padded, enc_mask, beg_seq_batch

    return enc_padded, enc_mask, beg_seq_batch

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))

    return train_x, train_y, dev_x, dev_y, test_x