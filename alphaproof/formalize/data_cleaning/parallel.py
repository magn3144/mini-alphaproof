import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from alphaproof.formalize.data_cleaning.pipeline import CleanResult, Timers, clean_records
from alphaproof.formalize.qwen3 import PARALLELISM_MODES, Qwen3


@dataclass
class ParallelContext:
    """Process information and collectives for one cleaning run."""

    mode: str
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    initialized: bool = False
    max_source_batch_size: int = 0

    @property
    def is_main(self) -> bool:
        """Return whether this process owns file I/O and reporting."""
        return self.rank == 0

    def broadcast(self, value: Any) -> Any:
        """Broadcast a Python value from rank 0 in distributed modes."""
        if not self.initialized:
            return value
        values = [value if self.is_main else None]
        dist.broadcast_object_list(values, src=0)
        return values[0]

    def clean_batch(
            self,
            records: list[dict],
            model: Qwen3,
            timers: Timers,
    ) -> list[CleanResult] | None:
        """Clean a global batch according to the configured parallelism.
        This is where the batch of data is split if data parallelism is enabled."""
        if self.mode != 'data':
            self.max_source_batch_size = max(
                    self.max_source_batch_size,
                    len(records),
            )
            results = clean_records(records, model, timers)
            return results if self.is_main else None

        indexed_records = list(enumerate(records))[self.rank::self.world_size]
        self.max_source_batch_size = max(
                self.max_source_batch_size,
                len(indexed_records),
        )
        local_records = [record for _, record in indexed_records]
        local_results = clean_records(local_records, model, timers)
        indexed_results: list[tuple[int, CleanResult]] = [
                (global_ix, result)
                for (global_ix, _), result in zip(indexed_records, local_results)
        ]
        gathered: list[list[tuple[int, CleanResult]] | None] | None = None
        if self.is_main:
            gathered = []
            for _ in range(self.world_size):
                gathered.append(None)
        dist.gather_object(indexed_results, gathered, dst=0)
        if not self.is_main:
            return None

        merged: list[CleanResult | None] = [None] * len(records)
        assert gathered is not None
        for rank_results in gathered:
            assert rank_results is not None
            for global_ix, result in rank_results:
                if merged[global_ix] is not None:
                    raise RuntimeError(f'Duplicate gathered result index: {global_ix}.')
                merged[global_ix] = result
        if any(result is None for result in merged):
            raise RuntimeError('Missing result while merging data-parallel batch.')
        return [result for result in merged if result is not None]

    def gather(self, value: Any) -> list[Any] | None:
        """Gather one Python value per rank onto rank 0."""
        if not self.initialized:
            return [value]
        gathered: list[Any] | None = (
                [None] * self.world_size if self.is_main else None
        )
        dist.gather_object(value, gathered, dst=0)
        return gathered

    def close(self) -> None:
        """Destroy the process group created for this run."""
        if self.initialized and dist.is_initialized():
            dist.destroy_process_group()


def initialize_parallelism(mode: str) -> ParallelContext:
    """Validate and initialize the requested cleaning execution mode."""
    if mode not in PARALLELISM_MODES:
        choices = ', '.join(sorted(PARALLELISM_MODES))
        raise ValueError(f'parallelism must be one of: {choices}.')
    if mode in {'none', 'balanced'}:
        return ParallelContext(mode=mode)
    if not torch.cuda.is_available():
        raise RuntimeError(f'parallelism={mode} requires CUDA.')

    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    if world_size != 2:
        raise ValueError(f'parallelism={mode} requires exactly two torchrun ranks.')
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    return ParallelContext(
            mode=mode,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            initialized=True,
    )
