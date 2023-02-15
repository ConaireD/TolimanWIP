## Contributing
There is plenty that still needs to be done to make the forwards model flight 
ready. If you wish to add layers to the optical system of the detector then 
you can do this within the `__init__` methods of the respective classes. 
Once you have finished with your changes please run `black` and `pydocstyle`.
Unfortunately you will have to manually make the changes with `pydocstyle`. 
These should come with the full installation of `toliman`.

Two additional features that I am aware of, which need to be added are support 
for Fresnel and a raytraced secondary mirror polish. The Fresnel will require
that an old code for `dLux` be updated to match the latest release. In addition,
once a physical detector is selected it will need to be tested and its 
parameters and noise sources incorporated into this model. 

The tests were created so that after making changes you can simply run the tests
to confirm that the existsing features still work. That said sometimes you will 
need to change and even delete tests depending on the magnitude of the changes.
If you do so please validate the new tests. To run the tests simply invoke 
`pytest` from within the root of the `git` repository. It will automatically 
detect and run all of the tests, which may take some time. 

Other than that, we are using type annotations, but in a relaxed sense, relying 
only on the inbuilt `python` types and avoiding using `typing` as much as 
possible since it reduces the complexity of the code. It was tempting to 
use `jaxtyping`, but this can lead to very long and very confusing function 
signatures for the unwary. It may be something that is pursued in later 
releases. Other aspects of the style should be self apparent from the 
existing code base. 
