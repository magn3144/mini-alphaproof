# Design choices
 - Used LeanTree for interacting with Lean.
 - Replay buffer samples uniformly.
 - Computes tactic prior by summing token logprobs. This is used as the prior in PUCT.
 - Value head uses mean pooled encoder output.