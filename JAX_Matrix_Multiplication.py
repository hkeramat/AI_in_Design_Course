# Students are supposed to run the matrix multiplication and compare the JAX with C++, Numpy, and python loop for a range of matrix dimension to see the difference 
# Feel free to play with it 
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

start_time = time.time()
key = random.PRNGKey(0)
x = random.normal(key, (400,400))

jnp.dot(x, x.T).block_until_ready()

print("--- %s seconds ---" % (time.time() - start_time))
