import numpy as np 
import pandas as pd
import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from transformers.modeling_outputs import BaseModelOutput
from transformers import PreTrainedTokenizerFast, BatchEncoding

from typing import Mapping, Dict, List
import tqdm
import os

from prompt import *

# helper functions
def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda(non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple([_move_to_cuda(x) for x in maybe_tensor])
        elif isinstance(maybe_tensor, Mapping):
            return type(maybe_tensor)({k: _move_to_cuda(v) for k, v in maybe_tensor.items()})
        else:
            return maybe_tensor

    return _move_to_cuda(sample)

def create_batch_dict(tokenizer: PreTrainedTokenizerFast, input_texts: List[str], always_add_eos: bool, max_length: int = 512) -> BatchEncoding:
    if not always_add_eos:
        return tokenizer(
            input_texts,
            max_length=max_length,
            padding=True,
            pad_to_multiple_of=8,
            return_token_type_ids=False,
            truncation=True,
            return_tensors='pt'
        )
    else:
        batch_dict = tokenizer(
            input_texts,
            max_length=max_length - 1,
            return_token_type_ids=False,
            return_attention_mask=False,
            padding=False,
            truncation=True
        )

        # append eos_token_id to every input_ids
        batch_dict['input_ids'] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]

        return tokenizer.pad(
            batch_dict,
            padding=True,
            pad_to_multiple_of=8,
            return_attention_mask=True,
            return_tensors="pt",
        )

# e5 encoder
cache_dir = '/shared/share_mala/llm-dro/model'
class DenseEncoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct', 
                                                 torch_dtype=torch.float16, 
                                                 cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct', cache_dir=cache_dir)
        self.gpu_count = torch.cuda.device_count()

        self.encoder.eval()
        self.encoder.cuda()

        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)

    @torch.no_grad()
    def encode(self, serialize_df, **kwargs) -> np.ndarray:
        """ Returns a list of embeddings for the given sentences.
        Args:
            input_texts (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """

        input_texts: List[str] = serialize_df['input'].tolist()
        encoded_embeds = []
        batch_size = 64 * self.gpu_count
        max_length = 4096
        for start_idx in tqdm.tqdm(range(0, len(input_texts), batch_size), desc='encoding', mininterval=10):
            batch_input_texts: List[str] = input_texts[start_idx: start_idx + batch_size]
            batch_dict = create_batch_dict(self.tokenizer, 
                                           batch_input_texts, always_add_eos=True, max_length=max_length-1)
            batch_dict = move_to_cuda(batch_dict)

            with torch.cuda.amp.autocast():
                outputs: BaseModelOutput = self.encoder(**batch_dict)
                embeds = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                encoded_embeds.append(embeds.cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0)
    
    def encode_text(self, input_text):
        return self.encode(pd.DataFrame({'input': [input_text]}))[0]


# derive embedding
def embed_ACS_table(task, df, root_dir):
    path = root_dir + '/embed/{}-{}.pkl'.format(task.data_name, task.prompt_method)
    # check if embedded table exists
    if os.path.exists(path) == True:
        return
    else:
        # get detailed prompt
        prompt = get_detailed_prompt(task)
        # add prompt to input
        df['input'] = df['feature'].apply(lambda x: prompt + x)
        # derive embedding
        model = DenseEncoder()
        embedding = model.encode(df)
        df['embedding'] = list(embedding)
        # save embed df
        df.to_pickle(path)

def embed_table(task, df, root_dir):
    if task.task_name in ['income', 'pubcov', 'mobility']:
        embed_ACS_table(task, df, root_dir)
    

def embed_e5(df):
    model = DenseEncoder()
    embedding = model.encode(df)
    return embedding
        

'''old code using openai API

import pandas as pd
import numpy as np
import tiktoken
import os

from openai import OpenAI
client = OpenAI(api_key='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')
def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)
def get_embedding(text: str, model="text-embedding-3-small", embedding_dim = None, **kwargs):
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model, **kwargs)
    # cut dimension (and normalize) if embedding_dim is specified
    if isinstance(embedding_dim, int):
        return normalize_l2(response.data[0].embedding[:embedding_dim])
    else:
        return response.data[0].embedding
def embed_table(serialize_name, embedding_method, serialize_df, root_dir):
    path = root_dir + '/embed/{}-{}.pkl'.format(serialize_name, embedding_method)
    # check if embedded table exists
    if os.path.exists(path) == False:
        if embedding_method == 'textembedding3small':
            # config
            embedding_model = "text-embedding-3-small"
            embedding_encoding = "cl100k_base"
            max_tokens = 8000  # the maximum for text-embedding-3-small is 8191
            encoding = tiktoken.get_encoding(embedding_encoding)
            # check number of tokens
            serialize_df.loc[:, 'n_tokens'] = serialize_df['feature'].apply(lambda x: len(encoding.encode(x)))
            #serialize_df["n_tokens"] = serialize_df['feature'].apply(lambda x: len(encoding.encode(x)))
            if max(serialize_df["n_tokens"]) > max_tokens:
                print('Warning! {} exceeds max tokens'.format(embedding_method))
            # derive text embedding
            serialize_df["embedding"] = serialize_df['feature'].apply(lambda x: get_embedding(x, model=embedding_model))
            # save embed df
            serialize_df.to_pickle(path)
        elif embedding_method == 'textembedding3large':
            # config
            embedding_model = "text-embedding-3-large"
            embedding_encoding = "cl100k_base"
            max_tokens = 8000  # the maximum for text-embedding-3-large is 8191
            encoding = tiktoken.get_encoding(embedding_encoding)
            # check number of tokens
            serialize_df.loc[:, 'n_tokens'] = serialize_df['feature'].apply(lambda x: len(encoding.encode(x)))
            if max(serialize_df["n_tokens"]) > max_tokens:
                print('Warning! {} exceeds max tokens'.format(embedding_method))
            # derive text embedding (cut dimension to 256)
            serialize_df["embedding"] = serialize_df['feature'].apply(lambda x: get_embedding(x, model=embedding_model,
                                                                            embedding_dim=256))
            # save embed df
            serialize_df.to_pickle(path)
'''