Extension of my previous [micrograd-vector](https://github.com/ckkissane/micrograd-vector) implementation. The main changes are:
* Use Tensor (wraps np.ndarray) instead of Vector (wraps 1D Python list)
* Introduce optim.py, which contains a simple SGD implementation
* Support batching
* Improve nn.Linear init
* Replace Sigmoid with ReLU

numpy does a lot of the dirty work, so it's much faster while also being
easier to read. Check out the mnist example notebook to see it in action.
