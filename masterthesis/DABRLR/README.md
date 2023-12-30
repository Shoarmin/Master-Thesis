Backdoor Attacks in a Federated Framework Using Pixel Intensities

To use the code, first get [the datasets](https://drive.google.com/drive/folders/1ta4ZSgRGc6hObZEoYJ8DscO-s4PnL6Ww?usp=sharing), and unzip them in the 'data' directory. The FMNIST and Cifar-10(0) dataset should download automatically when executing the code for the first time and do not need to be downloaded separately.

Be aware, Wandb was used to visualize the results. If you want to not visualize this, comment all the wandb lines out. 

You can run your experiments using the following input:
```bash
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200
```

All arguments after the .py file are arguments used to run an experiment. See the options.py file to have an overview of all possible arguments and their respective input. 

This runs a federated learning instance with 10 agents for 200 rounds with [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist), where in each round, local training consists of 2 epochs with a batch size of 256. By default, datasets are distributed uniformly between agents and the aggregation function is [FedAvg](https://arxiv.org/pdf/1602.05629.pdf).


Some examples for each dataset:
```bash
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --num_corrupt=1 --poison_frac=0.5 --climg_attack=0 --pattern=sig --delta_attack=20 --delta_val=100
python federated.py --data=fedemnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --num_corrupt=1  --poison_frac=0.5
python federated.py --data=cifar10 --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --num_corrupt=4  --poison_frac=0.5
python federated.py --data=tinyimage --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --num_corrupt=4  --poison_frac=0.5
python federated.py --data=reddit --local_ep=2 --bs=20 --num_agents=10 --rounds=200 --num_corrupt=1  --poison_frac=0.5 --poison_sentence="White people are rude"
```