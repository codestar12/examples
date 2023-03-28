
# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming dataset conversion scripts for C4 and The Pile."""

## did you forget to pretokenize and concat a streaming dataset? 
## Hey, it happens to the best of us. lucky for you we can take 
## that mds formated dataset and prepare it for deteriministic LLM consumption (yummy)

import os
import platform
import warnings
from argparse import ArgumentParser, Namespace
from typing import Union, Iterable, Dict, Optional
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from streaming import StreamingDataset, MDSWriter
import numpy as np

from tqdm import tqdm

def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        'Convert dataset into MDS format, optionally concatenating and tokenizing'
    )
    parser.add_argument('--local', type=str, required=True)
    parser.add_argument('--remote', type=str, required=False, default=None)
    parser.add_argument('--splits',
                        nargs='+',
                        default=None)
    parser.add_argument('--out_root', type=str, required=True)
    parser.add_argument('--compression', type=str, default=None)

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '--concat_tokens',
        type=int,
        help='Convert text to tokens and concatenate up to this many tokens')

    parser.add_argument('--tokenizer', type=str, required=False, default=None)
    parser.add_argument('--bos_text', type=str, required=False, default=None)
    parser.add_argument('--eos_text', type=str, required=False, default=None)
    parser.add_argument('--no_wrap', default=False, action='store_true')

    parsed = parser.parse_args()

    if os.path.isdir(parsed.out_root) and len(
            set(os.listdir(parsed.out_root)).intersection(set(
                parsed.splits))) > 0:
        raise ValueError(
            f'--out_root={parsed.out_root} contains {os.listdir(parsed.out_root)} which cannot overlap with the requested splits {parsed.splits}.'
        )

    # Make sure we have needed concat options
    if (parsed.concat_tokens is not None and
            isinstance(parsed.concat_tokens, int) and parsed.tokenizer is None):
        parser.error(
            'When setting --concat_tokens, you must specify a --tokenizer')

    # now that we have validated them, change BOS/EOS to strings
    if parsed.bos_text is None:
        parsed.bos_text = ''
    if parsed.eos_text is None:
        parsed.eos_text = ''
    return parsed


class ConcatTokensDatasetStreaming(IterableDataset):
    """An IterableDataset that returns token samples for MDSWriter.

    Returns dicts of {'tokens': bytes}

    To use data created by this class and written to MDS format:

    ```python
        import torch
        from streaming.base import StreamingDataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained('your/tokenizer')
        ds = StreamingDataset(local='mds-data-folder', split='val')

        # note, you need to copy the numpy array because the original is non-writeable
        # and torch does not support non-writeable tensors, so you get a scary warning and
        # if you do try to write to the tensor you get undefined behavior
        tokens = torch.from_numpy(np.frombuffer(ds[0]['tokens'], dtype=np.int64).copy())
        print(tokenizer.decode(tokens))
    ```
    """

    def __init__(self,
                 local: str,
                 remote: str,
                 split: str,
                 tokenizer: PreTrainedTokenizerBase,
                 max_length: int,
                 bos_text: str,
                 eos_text: str,
                 no_wrap: bool,
                 data_subset: Union[str, None] = None):
        self.tokenizer = tokenizer
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.max_length = max_length
        self.bos_text = bos_text
        self.eos_text = eos_text
        self.should_wrap = not no_wrap

        self.streaming_dataset = StreamingDataset(local=local,
                                                  remote=remote, 
                                                  split=split,
                                                  predownload=100_000,
                                                  shuffle=False,) # we don't need to shuffle for this right? 

        self.bos_tokens = self.tokenizer(self.bos_text,
                                         truncation=False,
                                         padding=False,
                                         add_special_tokens=False)['input_ids']
        if len(self.bos_tokens) > 1:
            warnings.warn(
                f'You specified --concat_tokens with --bos_text, but your BOS text is not tokenizing to one token\
                , instead we got {self.bos_tokens}. Quit if this was in error.')

        self.eos_tokens = self.tokenizer(self.eos_text,
                                         truncation=False,
                                         padding=False,
                                         add_special_tokens=False)['input_ids']
        if len(self.eos_tokens) > 1:
            warnings.warn(
                f'You specified --concat_tokens with --eos_text, but your EOS text is not tokenizing to one token\
                , instead we got {self.eos_tokens}. Quit if this was in error.')

        eos_text_provided = self.eos_text != ''
        bos_text_provided = self.bos_text != ''
        test_text = self.tokenizer('')
        if len(test_text['input_ids']) > 0 and (eos_text_provided or
                                                bos_text_provided):
            message = 'both eos and bos' if eos_text_provided and bos_text_provided else (
                'eos_text' if eos_text_provided else 'bos_text')
            warnings.warn(
                f'The provided tokenizer adds special tokens, but you also specified {message}. This may result '
                'in duplicated special tokens. Please be sure this is what you intend.'
            )

    def __iter__(self) -> Iterable[Dict[str, bytes]]:

        buffer = []
        for sample in self.streaming_dataset:
            encoded = self.tokenizer(sample['text'],
                                     truncation=False,
                                     padding=False)
            iids = encoded['input_ids']
            buffer = buffer + self.bos_tokens + iids + self.eos_tokens
            while len(buffer) >= self.max_length:
                concat_sample = buffer[:self.max_length]
                buffer = buffer[self.max_length:] if self.should_wrap else []
                yield {
                    # convert to bytes to store in MDS binary format
                    'tokens': np.asarray(concat_sample).tobytes()
                }

    def __len__(self) -> int:
        """Get the approx length as an IterableDataset.

        Returns:
            int: Dataset length.
        """
        return len(self.streaming_dataset)

def build_dataloader(dataset, batch_size) -> DataLoader:
    # Multiple workers is only supported on linux machines

    if 'linux' in platform.platform().lower():
        num_workers = 64
    else:
        num_workers = 0

    # If using multiple workers, configure each worker to prefetch as many samples as it can, up to
    # the aggregate device batch size
    # If not using workers, the torch DataLoader expects the default value for prefetch_factor,
    # which non-intuitively must be 2.
    prefetch_factor = max(1, 2 * batch_size //
                          num_workers) if num_workers > 0 else 2

    return DataLoader(
        dataset=dataset,
        sampler=None,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

def generate_samples(
        loader: DataLoader,
        truncate_num_samples: Optional[int] = None
) -> Iterable[Dict[str, bytes]]:
    """Generator over samples of a dataloader.

    Args:
       loader (DataLoader): A dataloader emitting batches like {key: [sample0_bytes, sample1_bytes, sample2_bytes, ...]}
       truncate_num_samples (Optional[int]): An optional # of samples to stop at.

    Yields:
        Sample dicts.
    """
    n_samples = 0
    for batch in loader:
        keys = list(batch.keys())
        current_bs = len(batch[keys[0]])
        for idx in range(current_bs):
            if truncate_num_samples is not None and n_samples == truncate_num_samples:
                return
            n_samples += 1
            yield {k: v[idx] for k, v in batch.items()}

def main(args: Namespace) -> None:
    """Main: create C4/pile streaming dataset.

    Args:
        args (Namespace): Commandline arguments.
    """

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # we will enforce length, so suppress warnings about sequences too long for the model
    tokenizer.model_max_length = int(1e30)
    columns = {'tokens': 'bytes'}

    if args.splits is None:
        dataset = ConcatTokensDatasetStreaming(
            local=args.local,
            remote=args.remote,
            split=None,
            max_length=args.concat_tokens,
            tokenizer=tokenizer,
            eos_text=args.eos_text,
            bos_text=args.bos_text,
            no_wrap=args.no_wrap,
        )
        # Get samples

        ds_len = len(dataset)
        loader = build_dataloader(dataset=dataset, batch_size=512)
        samples = generate_samples(loader)


        # Write samples
        print(f'Converting to MDS format...')
        with MDSWriter(out=args.out_root,
                       max_workers=8,
                       progress_bar=True,
                       columns=columns,
                       compression=args.compression) as out:
            for sample in tqdm(samples, total=ds_len):
                out.write(sample)
    else:
        for split_name in args.splits:

            dataset = ConcatTokensDatasetStreaming(
                local=args.local,
                remote=args.remote,
                split=split_name,
                max_length=args.concat_tokens,
                tokenizer=tokenizer,
                eos_text=args.eos_text,
                bos_text=args.bos_text,
                no_wrap=args.no_wrap,
            )
            # Get samples

            ds_len = len(dataset)
            loader = build_dataloader(dataset=dataset, batch_size=512)
            samples = generate_samples(loader)


            # Write samples
            print(f'Converting {split_name} to MDS format...')
            with MDSWriter(dirname=os.path.join(args.out_root, split_name),
                        columns=columns,
                        compression=args.compression) as out:
                for sample in tqdm(samples, desc=split_name, total=ds_len):
                    out.write(sample)


if __name__ == '__main__':
    main(parse_args())
