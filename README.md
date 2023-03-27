# README

# Quickstart
Make sure opensees is on your path, make a symlink

`cp ./bin/opensees /usr/local/bin`

```
pip3 install -r requirements.txt
mypy --install-types
make
```

## Debugging
Set a `breakpoint()` anywhere

## TODO

## Design

- fix deselecting model.yml when clicking 'run'
- deploy to streamlit see how it works

- include slabs
- fix tests

include more tests for CDMX design
include tests for convergence of IDA
include tests for convergence of pushover

- fix pushover convergence
- fix ida convergence
- Compute Vy, uy correctly from pushover curve
- aim for lower period to account for more flexibility in real model


## COMMON PROBLEMS

```
pp/design.py:213: in force_design
    fem = seed_class.from_spec(self)
app/fem.py:125: in from_spec
    fem = cls(
app/fem.py:869: in __init__
    return super().__init__(*args, **kwargs)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <[AttributeError("'PlainFEM' object has no attribute 'periods'") raised in repr()] PlainFEM object at 0x28c0933d0>
AttributeError: can't set attribute
```

Happens when overriding @properties, check that the definition of dataclasses is not duplicating properties
