# Design choices
 - Used LeanTree for interacting with Lean.
 - Replay buffer samples uniformly.
 - Computes tactic prior by summing token logprobs. This is used as the prior in PUCT.
 - Value head uses first hidden vector in encoder output.