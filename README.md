# Test Repo for Block Matrix Multiply

This runs logistic regression over the mnist dataset.

To run normal logistic regression (using `torch.nn.linear`) execute:

```
python butterfly.py --mode linear
```

To run logistic regression with the special diagonal multiply (code from Tri) run:

```
python butterfly.py --mode butterfly
```
