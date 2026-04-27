# Multiple Instance Learning (MIL) Training Question

## Problem Description

You are provided two datasets in JSON-Lines format, `data/train.jsonl` and `data/val.jsonl`

The package contains a model that is attempting to learn the difference between trajectories labeled 0 and trajectories labeled non-zero

The model outputs a probability for each *instance* in the trajectory. If *any one* instance predicts 0, the entire trajectory is considered 0
If *all* the instances predict non-zero, the entire trajectory is considered non-zero.

There is further structure that indicates differences between trajectories labeled 1 and 2, which we may discuss. For the scope of this problem, both you and the the training
package should see (1,2) as subsets of the same class "non-zero".

## Your Task

In the `train` folder, we have:

* [train/train.py](train/train.py) which contains the training loop.
* [train/mil.py](train/mil.py) which contains the MIL loss and target generation, which you will need to implement.
* [train/dataset.py](train/dataset.py) which contains a `TrajectoryDataset` class that loads the data and provides it in a format suitable for training.
* [train/config/default.yaml](train/config/default.yaml) contains the configuration for the training loop.

In [train/mil.py](train/mil.py) you will find three functions:

```python
def trajectory_mil_target(row_label: int) -> float:
    pass


def mil_hybrid_loss(logits: torch.Tensor, y_mil: float) -> torch.Tensor:
    pass


@torch.no_grad()
def prediction_matches_target(p: torch.Tensor, y_mil: float) -> bool:
    pass
```

The first function, `trajectory_mil_target`, should take in a row label (0, 1, or 2) and return the appropriate target for the MIL loss function.

The second function, `mil_hybrid_loss`, should compute the loss given the model's logits and the MIL target. 

The third function, `prediction_matches_target`, should determine if the model's predictions match the MIL target.

---

To see how they are used, look at the training loop in [train/train.py](train/train.py).

---

[vscode.dev](https://vscode.dev/github/wmarsman/trajectory_eval) can be used to open this for editing in your browser.