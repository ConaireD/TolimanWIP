I imagine that you are familiar with `python`, but for the sake of
completeness and consistency it is a dynamicly typed, interpretted-bytecode 
language with inbuilt support for functional and object oriented programming. 
At present `python` does not ship with a `jit` runtime for `python` bytecode, 
but this is scheduled for the `python3.12` release. `python` is praised for 
its readability and critized for its speed. Since speed is a necessary evil 
in our `toliman` package we are using `dLux`, which in turn used `jax`, a 
third party `python/numpy` compiler and sutomatic differentiation framework.
At the `toliman` level you will rarely need to interact with `jax` directly.

