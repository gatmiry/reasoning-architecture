To run code 

```bash
python src/train.py trainer.cfg.threshold=0.99 trainer.cfg.function_dim=12
```

If you want to run multiple experiments with different hyperparameters, you can make a jsonl file like in `experiments.jsonl` and then run the following command:

```bash
bash scripts/submit_multiple_experiments/submit_experiments.sh scripts/submit_multiple_experiments/experiments.jsonl
```

This will run all the experiments in the jsonl file in parallel.
