## Pytest 
Code validity is very important in modern software development. Modern 
programming languages are increasingly incorporating features to make 
programming safer and more provable. For example, `SPARK` a subset of 
`Ada` implements a contracts to guarantee compile time saftey. `rust`
is another example, implementing a borrowing API for sharing information 
that makes it almost impossible to get memory leaks. There is also 
a steady shift towards the functional programming paradigm, which, among 
other things, minimises side effects. 

??? note "Side Effects"
    A side effect is a change in the state of a program that is not 
    explicitly returned. A program cannot do anything without side 
    effects but they are a common cause of runtime errors.

`python`, being an older language, does not implement many of these 
more modern features. As a result we have to check the validity of our
code the old fashioned way: *rigorous testing*. Like most languages 
`python` has several third party implementations of testing frameworks. 
For the `toliman` project, we chose `pytest`. 

??? note "Why `pytest`"
    A common naming convention for testing frameworks is `xUnit`, where
    `x` specifies the programming language. For example, `jUnit` for 
    `java`. These pacakges usually implement a setup-teardown design.
    You program a function to setup the test/tests and one to tear it
    down. This can quickly become too rigid, preventing you from using 
    multiple setup functions. `pytest` implements a different interface
    inspired by the percieved failings of the `xUnit` ecosystems.

`pytest` is a very flexible framework. It can be invoked using `pytest`
and will automatically detect tests based on file and function/class names.
Alternatively you can run it on a specific file using `pytest path/to/file`.
It allows you to single out test classes and even test functions using 
`pytest path/to/file::TestClass::test_function`. If that was not flexible 
enough you can run tests using a keyword search via the `-k` flag. 

A `fixture` is a `pytest` artifact used to setup and teardown tests. 
`fixture`s are flexible since they can be requested by any other `fixture` 
or any other test. This modularity can make testing code much simpler. 
However, `fixture`s are not treated like a regular function. Instead 
they are executed once and cached when they are requested. In a world 
devoid of side effects this would be fine, but when they become necessary 
this can cause problems. `fixture`s are executed in the order that they
are requested so be careful.

!!! note
    `pytest` leverages the `assert` keyword to determine if a test passes. A
    test can fail and assertion or it can fail because there was an error. 
    Tests that use `assert` are more useful for checking the code is correct.

??? example "Fixtures in the Wild"
    
    ```python
    import pytest

    @pytest.fixture
    def letters_of_the_alphabet() -> list:
        return [chr(i) for i in range(97, 123)]

    def test_str_upper_on_lower_case(letters_of_the_alphabet: list) -> None:
        for char in letters_of_alphabet:
            assert char.upper().isupper()
    ```

    This test is would evaluate as 
    `test_str_upper_on_lower_case(letter_of_the_world())`. In fact, we 
    can use another `pytest` feature to improve this test. In general 
    it is advised that every unit test should test a single case, we are 
    testing twenty-six. `pytest.mark.parametrize` is similar to the inbuilt
    `map` function, but creates unique tests for each case. Using the example
    from earlier,

    ```python
    import pytest

    @pytest.mark.parametrize("char", [chr(i) for i in range(97, 123)])
    def test_str_upper_on_lower_case(char: chr) -> None:
        assert char.upper().isupper()
    ```

If you on the ball you may have noticed that all `fixture`s are evaluated 
before the test is executed. This can lead to problems when tests involve 
side effects. As a result using `yield` in a `fixture` will cause the 
code after `yield` to evaluate once the test has finished. We have used 
this feature extensively in the development of `toliman`, since we deal 
with a lot of file operations. 

??? example "`yield` in `fixture`s"
    ```python 
    import pytest

    @pytest.fixture
    def print_info() -> None:
        print(">>>")
        yield
        print("<<<")

    def test(print_info: None) -> None:
        print("TEST")
    ```
    Will output 
    ```
    >>>
    TEST
    <<<
    ```

??? tip "Writing Unit Tests"
    You should not have to write many unit tests for `toliman`, since they 
    have been programmed already. If you do have to write some unit tests 
    however, there are two guidlines that are helpful to remember: a) 
    FIRST and b) AAAC. FIRST lists the features of a good unit test and
    AAAC explains how one may implement such a test. 

    **FIRST**
    - **F**ast: Typically there are many more tests than functions. Especially 
                if you each test is specific to a single case. In a standard 
                workflow the tests are run to validate any new changes. If
                each test takes a macropspic amount of time, no one will use 
                the tests and they become redundant.
    - **I**ndependent: Each test should be a single *unit* (its in the name).
                       This way the success or failure of a test does not 
                       depend on the other tests.
    - **R**epeatable: Nothing is more frustrating than getting different 
                      results on different machines. Ideally, tests should 
                      not rely on system specific code, making debugging in 
                      a mixed team easier.
    - **S**elf-validating: This just means that the tests should pass or fail 
                           without the tester having to check. This sounds 
                           silly, but for complex outputs it can be difficult
                           to automatically determine if they are correct. 
    - **T**horough: This should be obvious, tests are not that useful if 
                    they only catch some of the bugs. Again, implementing 
                    thorough tests is time consuming and can often feal
                    very unrewarding.

    **AAAC**
    - **A**rrange: Set up the evironment of the test.
    - **A**ct: Perform the test action.
    - **A**ssert: Check if the test action produced the correct output.
    - **C**lean Up: Revert any side effects or free memory.
 
### Plugins 
`pytest` is popular, so users have created plugins to enhance the functionality
as they needed. We use several plugins for `toliman` but **none** are strictly
necessary. 

#### Pytest Xdist 
`pytest-xdist` lets us run our tests across multiple processes.
This plugin helps us make our unit tests fast (**F**IRST). Some of the 
programs we are testing are necessarily time consuming so running them
in parallel is very useful. Unfortunately it is difficult to use this 
plugin when the tests have side effects. In general side effects make
test co-dependent (F**I**RST). To invoke tests in a parallel way use 
`pytest ... -n num_processes`. The only tests in the `toliman` test 
suite that can be run on separate processes are the tests in 
`tests/test_toliman.py`.

#### Pytest Timer 
`pytest-timer` is a reporting tool. It tells us how long each test took.
This helps with **F**IRST by letting us identify which tests/functions 
need to be optimised or isolated. Once installed the plugin justs 
adds a table to the `pytest` ouput and you are not required to interact 
with it in any way. 

!!! tip 
    Some things are going to be slow. Not every program can be written so 
    that it executes in a fraction of a second. We can identify tests 
    that are slow using `pytest.mark.slow`. We can then leave these tests
    out when running pytest using `pytest ... -m "not slow"`. 

#### Pytest Cov 
`pytest-cov` addresses FIRS**T**. It shows us how much of our code is 
tested. Like `pytest-timer` it is a diagnostics tool. To generate a 
coverage report add the `--cov` flag to your `pytest` call. This will 
be ouput in the terminal giving a percentage by file. If you wish to 
see exactly where is not tested it can be used to generate a `html` 
report showing what lines were and were not tested.

#### Pytest Sugar 
While most of the plugins that we recommend have some functional value,
`pytest-sugar` purely aesthetic. This plugin beautifies the ouput of 
`pytest` and runs automatically once installed. I highly recommend it
if you plan to be running the tests much.

### Resources
- [https://docs.pytest.org/en/7.1.x/contents.html](https://docs.pytest.org/en/7.1.x/contents.html)
- [https://pytest-xdist.readthedocs.io/en/latest/](https://pytest-xdist.readthedocs.io/en/latest/)
- [https://pytest-cov.readthedocs.io/en/latest/](https://pytest-cov.readthedocs.io/en/latest/)
- [https://pypi.org/project/pytest-timer/](https://pypi.org/project/pytest-timer/)
- [https://pypi.org/project/pytest-sugar/](https://pypi.org/project/pytest-sugar/)

