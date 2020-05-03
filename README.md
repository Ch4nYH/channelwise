# channelwise

## Running
```
python main.py --gpu 0
```

## Environment

pytorch==1.2.0

**Generating channel-wise statistics**:
-> models.py@get_channel_stats: calculating mean and standard deviation, shape: (#channels, 2)

**Generating observation**:
-> meta_trainer.py@observe, shape: (#channels, N)


**Generating actions**: meta_trainer.py#213

-> policy.py#36 generating features and rnn states

-> policy.py#37 generating distributions using a **same** linear layer

-> policy.py#46 return actions


**Taking actions**: meta_trainer.py#219

-> optimizers.py#76 setting mask

-> optimizers.py#97 cliping gradients