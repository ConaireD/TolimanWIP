## Object Oriented Programming
Object oriented programming (OOP) is a popular programming paradigm. I would 
describe OOP as an abstract system of representing *things* and the 
relationships between them. 

!!! note
    A programming language expresses a program. Paradigms like OOP are 
    not specified by a single implementation. As a result, programming 
    languages vary in the details of implementation. This leads to a 
    decent amount of contention among various programming communities
    as they argue whether or not a language supports OOP. 

Many programming languages, `python` included, provide syntax for expressing
OOP components. However, such syntax choices are efficiency and saftey tools.
You can always design object oriented (OO) programs with the tools available. 

In OOP a *thing* is represented by a `class`. The `class` contains information 
about the *thing*, called the state, and information about how the *thing*
behaves. A behaviour is like a procedure/function, it takes in information,
including the state of the class, and produces an output. A `class` is said 
enscapulate the behaviour and state. 

!!! example
    We want to represent a nut in a program. The state of the nut is a *very
    complex* thing. At the end of the day, to *correctly* simulate the nut 
    we would have to simulate its constituent atoms. This is not going to 
    be very efficient so instead we will make some simplifying assumptions. 
    Our first assumption is that the state of the nut can be completely specified
    by `is_cracked`, `is_germinated`, and `nutrients_remaining`. A nut 
    is *fairly* innanimate, so we will only model two behaviours `crack` and 
    `germinate`. 

    First, let's look at the `crack` behaviour. A cracked nut cannot un-crack
    itself. OOP let's us enforce this. When we instantiate the nut object,
    we can set `is_cracked` to `False`, then the `crack` method with change
    `is_cracked` to `True`. Without the OOP enscapulation is would be harder
    to enforce a contract on how information is changed/mutated by a program.
    A similar contract would be enforced on the `germinate` behaviour. 

    The strictness of these contracts is language specific. `java` allows 
    the programmer to restrict access to the information in a `class`, 
    leading to strict contracts. In `python` on the other hand, the state 
    of the `class` can be changed by any peice of code. As a result, `python`
    contracts need to be self enforced. 

`class`es can be related in various ways. It is at this point that language 
implementation begins to muddy the water. There are two common ways that 
`class`es can be related: by *inheritance*, or by *composition*. As a general
rule inheritance implies *specialisation*. Often this type of relationship 
is described using the phrase: `class A` **is a** `class B`. Composition on 
the other hand indicates that `class A` **has a** `class B`.

!!! example
    Revisiting the nut example, a hazelnut, *is a* nut. This implies that 
    we would program a hazelnut using inheritance. This is a good time to 
    rigorously define how something that *is* something else should behave.
    The state of a hazelnut must include all the attributes of a nut. We
    cannot remove attributes, but we can add them. Since hazelnuts are good
    eating, it makes sense to extend the state to include an attribute 
    `is_cooked`. Similarly, a hazelnut *is a* nut, so it must have all the 
    same behaviours of a nut. Once again, we can add new behaviours, for 
    `cook`. We can also change how the implementation details in inheritance 
    situations.

If `Hazelnut` inherits from `Nut`, then we call `Nut` the *parent* or *super* 
`class` and `Hazelnut` the *child* `class`. The implementation of a behaviour 
is called a method and a variable that describes the state of the class is 
called an attribute (opinions may differ but this is my convention). Composition
is achieved when a `class` attribute is an instance of another `class`. 

!!! example
    A tree might also be described by a `class`. An instance of a `Tree` can 
    have an instance of a `Nut`. This is composition.

## Abstract Classes, Interfaces and Multiple Inheritance
A benefit of OOP is that it reduces repetition. An inherited method does 
not always need to be re-implemented. Re-implementation of a method in the 
parent class is often called overriding. When a method is overriden (perhaps
multiple times) it starts to raise questions about name resolution. Moreover,
it can become difficult to track down functionality and bugs within the 
heirachy. 

An abstract class is a contract. Abstract classes are abstract in the sense
that they cannot be instantiated. You never find, just a nut. It is always 
a specific type of nut, under the abstract classification of a nut. Abstract
classes may specify what behaviours to expect, but often these behaviours are 
not implemented in the abstract class. For example, `crack` might be implemented
differently for `Macadamia` and `Hazelnut`. Methods *can be* implemented in 
the abstract class.

An interface is a pure contract. Like an abstract class it states a subset 
of the behaviours and state of any child classes. However, unlike an abstract
class it cannot provide any implementation. An interface may seem like it 
is an extra and unecessary level of complexity, so why were they created?
Interfaces were created for scenarios where something **is** multiple other 
things. For example, a `Hazelnut` *is a* nut, but it is also a `Food`. 

!!! example
    Interfaces are used by some programming languages to resolve multple 
    inheritance scenarios. For example, `java` does not allow one class 
    to inherit from more than one other `class`. Let's say that `Food` 
    and `Nut` both provide implementations of the method `scent`. Which,
    implementations should be invoked by `Hazelnut.scent()`? `python`
    unlike `java` allows multiple inheritance, and resolves methods according
    to some convention.

## Static Classes and Static Methods


## Object Oriented `python`
`python` provides the `class` keyword, to define a class:
```python 
class ClassName(ParentClass1, ParentClass2, ...):
```
`python` uses so called "dunder" methods to manage how a use defined class
interacts with the syntax of the language. For example, `__init__` manages
how a `class` is instantiated and `__eq__` manages equality checks using `==`,
etc. When defining a method the `def` keyword is used, within the indented 
scope of the `class`. In `python` you do not have to declare `class` attributes
and can dynamically allocate them. However, I recommend that you do declare
them as it makes the logic of the program much clearer.

!!! example
    Access to abstract classes is managed via the `abc` package in the standard 
    library. 
    ```python
    class Nut(abc.ABC):
        is_cracked: bool
        is_germinated: bool
        remaining_nutrients: float

        def __init__(self: object, weight: float) -> object:
            self.is_cracked: bool = False
            self.is_germinated: bool = False
            self.remaining_nutrients: float = self._nutrients_per_kg(weight)
    
        @abc.abstractmethod
        def _nutrients_per_kg(self: object, weight: float) -> float: 
            """
            """

        @abc.abstractmethod
        def crack(self: object) -> object:
            """
            """

        @abc.abstractmethod
        def germinate(self: object) -> object:
            """
            """
    ```
    You may notice that the first parameter to all of the methods is `self`.
    This refers to the instance of `Nut` and is required for non-static 
    methods.

`python` also provides the `dataclasses` package in the standard library.
This writing classes much faster, as given the attributes 
`dataclasses.dataclass` automatically generates `__init__`, `__eq__` and 
other dunder methods.
