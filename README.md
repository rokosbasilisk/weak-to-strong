# Weak-to-strong generalization

![Our setup and how it relates to superhuman AI alignment](./weak-to-strong-setup.png)

[paper on weak-to-strong generalization](https://cdn.openai.com/papers/weak-to-strong-generalization.pdf).

openai's implementation of weak-to-strong learning setup for binary classification tasks.  


#### Running the Script

The main script of the project is train_weak_to_strong.py. It can be run from the command line using the following command:
```
python train_weak_to_strong.py
```

The script accepts several command-line arguments to customize the training process. Here are some examples:

```
python train_weak_to_strong.py --batch_size 32 --max_ctx 512 --ds_name "sciq" --loss "logconf" --n_docs 1000 --n_test_docs 100 --weak_model_size "gpt2-medium" --strong_model_size "gpt2-large" --seed 42
```

# experiments in weak-2-strong generalization

- add pythia-configs
- try some chessy experiment
- plot scaling laws

