# PETs

The final project of Privacy Enchanced Technology in Saarland.

## Authors

Mejbah Uddin Shameem 2577739 \
LIN, YU-DE 2577489 \
David Ahmed 2581749


## Folder Structure

* Membership Inference: Notebook to demonstrate the membership inference attack
* Model Inversion: Notebook to demonstrate the model inversion attack
* Model Stealing: Notebook to demonstrate the model stealing attack

## How to run

Here we attach the commands to run the attacks. The results we demonstrated here are attacks to the RestNet-like models. To see the sophisticated outcome of these three attacks, please check the notebook in each folder.

### Model Inversion
In model inversion, the adversary tries to invert the original data for the training. To run this attack, please do the following:
```python
python3 run.py --attack model_inversion --data <mnist, fashion, cifar10 > --mode production
``` 
|Dataset|Victim Precision|Attack Label|Min Loss|
|:-:|:-:|:-:|:-:|
|Cifar10|60.4%|cat|0.0054|
|MNIST|98.81%|0|0.0012|
|Fashion-MNIST|88.93%|Sneaker|0.0072|

### Model Stealing:
Model stealing creates a new model (to preserve the original model's architecture) and then trains them using random points. You may run a test as follows
```python
python3 run.py --attack model_stealing --data <mnist, fashion, cifar10 > --mode production
```
|Dataset|Victim Precision|Stealing Precision|
|:-:|:-:|:-:|
|Cifar10|60.7%|0.01%|
|MNIST|98.62%|0.98%|
|Fashion-MNIST|88.88%|52.78%|


### Membership Inference:
To run membership inference please use the following command. You can change argument (data) of dataset name that you want to run the attack.
```python
python3 run.py --attack membership_inference --data <mnist, fashion, cifar10 > --mode production
```
|Dataset|Victim Precision|Meta Classifer's Precison|Meta Classifer's Recall|
|:-:|:-:|:-:|:-:|
|Cifar10|56.59%|50%|51%|
|MNIST|97.19%|50%|60%|
|Fashion-MNIST|88.22%|50%|89%|


