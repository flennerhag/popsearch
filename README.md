# Population-based hyper-parameter search

A minimalist hyper-parameter search package aimed at deep learning, where training bad models is costly and hyper-parameter distributions
often "bad" (i.e. discrete, discontinuous, et.c.). 

Based on https://arxiv.org/abs/1711.09846 but differs in that the population and the degree of concurrency is de-coupled, to avoid 
constraining the population size to the number of concurrent jobs. Current implementation restarts job fully 
(as opposed to permuting current job and continuing).

The package is built to be easy to use: simply implement a training function that accepts one argument, the ``State`` variable.
Then in your training loop insert an evaluation: 

```python
state.eval(loss)
``` 

that's it. Use the ``Parameter`` class to define your parameters (use the ``frozen`` argument to fix the value of a parameter) 
and their distributions.  The ``Config`` sets the parameters of the search itself, such as number of concurrent jobs et.c.

See the [example](https://github.com/flennerhag/popsearch/tree/master/example) directory for an example use case.
