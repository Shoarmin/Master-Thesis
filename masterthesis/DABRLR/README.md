#ENTER TITLE
It has been tested with PyTorch 1.9.0.

To use the code, first get [the datasets](https://drive.google.com/drive/folders/1ta4ZSgRGc6hObZEoYJ8DscO-s4PnL6Ww?usp=sharing), and unzip them in the 'data' directory.
You can see some example usage in ```src/runner.sh``` and use this script to run your own experiments. For example, the first line says:

```bash
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200
```

This runs a federated learning instance with 10 agents for 200 rounds with [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist), where in each round, local training consists of 2 epochs with a batch size of 256. By default, datasets are distributed uniformly between agents and the aggregation function is [FedAvg](https://arxiv.org/pdf/1602.05629.pdf).

In the second line, we see how agents can carry a backdoor attack:

```bash
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --num_corrupt=1  --poison_frac=0.5
python federated.py --data=fedemnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --num_corrupt=1  --poison_frac=0.5
python federated.py --data=cifar10 --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --num_corrupt=1  --poison_frac=0.5
python federated.py --data=tinyimage --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --num_corrupt=1  --poison_frac=0.5
python federated.py --data=reddit --local_ep=2 --bs=20 --num_agents=10 --rounds=200 --num_corrupt=1  --poison_frac=0.5
```

Now, we have a corrupt agent who carries a backdoor attack by poisioning half of his local dataset. The base and target classes for the attack can be specified as argument via ```--base_class (default is 1)``` and ```--target_class (default is 7)```.

Finally, in the third line, we see how we can activate the robust learning rate and try to defend against the attack:

```
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --num_corrupt=1  --poison_frac=0.5 --robustLR_threshold=4
```
When the argument ```--robustLR_threshold``` is set to a positive value, it activates the defense with the given threshold.


Apart from these, one can supply different trojan patterns, use different aggregation functions, and datasets. See ```src/options.py``` and ```src/runner.sh``` for more usage. One thing to note is, when Cifar10 is used, the backdoor pattern is partitioned between the corrupt agents to simulate what's called a [Distributed Backdoor Attack](https://openreview.net/forum?id=rkgyS0VFvr). See ```add_pattern_bd``` method in ```src/utils.py```.
