## Data cleaning / formalization
- [X] Fix errors in data cleaning script.
- [X] Test data cleaning script with timer and LLM otputs printed to see why it takes so long to execute.
- [X] Organize code in data_cleaning.py
- [X] Understand how data cleaning script batches problems of the same type.
- [X] Add splitter for proof questions.
- [X] Add JSON schema to model in data cleaning pipeline.
- [X] Run a cleaning run.
- [X] Give qwen a max batch size parameter to avoid memory problems.
- [ ] Make it run on two A100s to achieve a much larger batch size.
- [X] Check after the existence theorem has been created that it is not trivial.
- [ ] Generate SFT dataset from finished lean proofs.
- [X] Make formalization pipeline run on two A100 80GB GPUs.
- [ ] Run formalization pipeline.
- [ ] Figure out which licences the datasets / models have.

## Training
- [ ] Train on SFT dataset.
- [ ] Train on formalized theorems using AlphaProof.