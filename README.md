# TADAM(Trust region ADAptive Moment estimation)

[Tadam Paper](https://www.sciencedirect.com/science/article/pii/S089360802300504X)

[Tensorflow Implementation of Tadam](https://github.com/dddong2/Tadam/tree/main)

This is the PyTorch implementation of Tadam. 
I have refactored the code so that the interface of Tadam closely matches that of Adam. 
However, Tadam requires two additional parameters beyond those used by Adam:

1. **total_steps**: Defines the total number of iterations for Tadam's gradient descent.
2. **gamma**: Controls how the trust region's size is adjusted.

- If the optimizer's performance is poor (i.e., $\bar{\rho}_n < \gamma$), the trust region is reduced.
- If the optimizer's performance is excellent (i.e., $\bar{\rho}_n > 1 - \gamma$), the trust region is expanded.
- If the optimizer's performance is moderate (i.e., $\gamma \leq \bar{\rho}_n \leq 1 - \gamma$), the trust region remains unchanged.

In `main.py`, you can see the following code:
```python
    total_steps = ARGS.epochs * (60_000 // ARGS.batch_size)
    gamma = 0.25
```

There are two ways to use Tadam:

### Option 1: Use Tadam in `my_own_optim.py`
In `main.py`, you will see:
```python
    optimizer = Tadam(model.parameters(), total_steps=total_steps, gamma=0.25, lr=ARGS.lr)
```

### Option 2: Use Tadam Like Adam
To use Tadam the same way as Adam, save `my_own_optim.py` as `tadam.py` inside your PyTorch package.

For example, I saved it in:
```
/Users/sungchul/opt/anaconda3/lib/python3.9/site-packages/torch/optim/tadam.py
```

To find the path for `torch.optim`, you can run:
```python
import torch.optim
import os

print(os.path.dirname(torch.optim.__file__))
```

Once you’ve saved `tadam.py` inside `torch.optim`, you'll need to register it. To do so:

1. Open `__init__.py` in the `torch.optim` directory.
2. Add the following lines:
```python
from .tadam import Tadam

__all__ = [
    'Optimizer', 'SGD', 'Adam', 'AdamW', 'RMSprop',
    'SparseAdam', 'LBFGS', 'Tadam'  # Include your optimizer here
]
```

Now, you can use Tadam just like Adam. 
Just remember, you need to specify two additional parameters: `total_steps` and `gamma`, in addition to Adam’s parameters:
```python
    optimizer = optim.Tadam(model.parameters(), total_steps=total_steps, gamma=gamma, lr=ARGS.lr)
```

Have Fun!!!




## ℹ️ Summary:

- ### Tadam approximates the loss up to the second order using the Fisher.

- ### Tadam approximates the Fisher and reduces the computational burdens to the O(N) level.

- ### Tadam employs an adaptive trust region scheme to reduce approximate errors and guarantee stability. 

- ### Tadam evaluates how well it minimizes the loss function and uses this information to adjust the trust region dynamically.

<br><br>

## Experiment
-  We use our Tadam to train the deep auto-encoder. The training data sets are MNIST, Fashion-MNIST, CIFAR-10, and celebA. We train each auto-encoder ten times and record the loss's mean and standard deviations. Tadam exhibits a space and time complexity of $O(N)$, placing it on par with other widely used optimizers such as Adam, AMSGrad, Radam, and Nadam.



### Validation loss per epoch

![L2 loss per epoch](/images/loss_mse_step.png)

- Tadam converges faster than the benchmarks.

<br>

### Validation loss by varying $\gamma$

![L2 loss per epoch](/images/loss_mse_gamma_up.png)

- We use the hyper-parameter $\gamma \in (0, 0.25]$ to measure Tadam's training performance and update the $\delta_n$, which controls the trust region size. 
- We evaluate the impact of $\gamma$, we use $\gamma$ values of $0.1$, $0.2$, and $0.25$ while maintaining a fixed learning rate $\eta$ of $0.001$, respectively. We observe that Tadam consistently maintains a relatively stable validation loss across the different $\gamma$ values, suggesting that Tadam's performance is relatively insensitive to the specific choices of $\gamma$.

# Q&A

### Q. I don't quite understand the update equation for v_n in your Algorithm 1. Why is the expression MA(g_n - gbar_n-1)(g_n - gbar_n)? The gbar_n-1 term is a little surprising to me.

A. Initially, we searched for references on how others handle the moving average of the second moment, and we found both MA(g_n - gbar_n)(g_n - gbar_n) and MA(g_n - gbar_n-1)(g_n - gbar_n). We experimented using both representations; the second performed better than the first, and we reported only the second in the paper. g_n is the current gradient, gbar_n is the moving average containing the current, and  gbar_n-1 is without the current. So, MA(g_n - gbar_n-1)(g_n - gbar_n) is a mixture of (backward) Nesterov momentum moving average and more traditional moving average.

### Q. The "for n = 1 to N" loop in Algorithm 1, does n represent the n-th sample, n-th mini-batch, or n-th epoch? 

A. One likely uses adam in the code when one trains a model. To use tadam instead of adam, just add t in front of adam, i.e., change adam to tadam. That is the original intention of our algorithm. One can interpret the for loop in Algorithm 1 in this respect. For our experiment setting, however, to quickly observe the difference between the adam and tadam, we update the model parameters for each mini-batch.
