# Design choices
 - Used LeanTree for interacting with Lean.
 - Replay buffer samples uniformly.
 - Computes tactic prior by summing token logprobs. This is used as the prior in PUCT.
 - Value head uses mean pooled encoder output.
 - compute_num_simulations() just returns 1000 for now.
 - Data for each run should be stored like this, so runs can be resumed:
 runs/
  2026-06-30_1030_debug/
    config.json
    matchmaker_stats.json
    results.jsonl
    replay_buffer.jsonl
    checkpoints/
      step_0001000.pt
      step_0002000.pt
      latest.pt
 - 