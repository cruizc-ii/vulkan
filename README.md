# README
SI engineering units.
force: kN
length: m
time: s

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
- deploy to streamlit see how it works
- [x] push with 1st mode, plot modes of vibration
- [ ] fix pushover stats, merge everything on a single call
- [ ] callibrate str-elems+slabs cost
- [ ] Compute Vy, uy correctly from pushover curve as 10% of original stiffness
- [ ] for collapse, when analysis doesn't converge at the last subdivision, create an empty file called collapse.csv with drifts, printA, etc, then process into results.yml with info about the collapse
- [ ] design with a smaller period, so the extra flexibility doesn't hurt as much
- [ ] change theta_y -> theta_y2 (Fardis)

### Design
- [x] fix deselecting model.yml when clicking 'run'
- [ ] introduce PDelta.
- [ ] include more tests for CDMX design
- [ ] aim for lower period in design procedure to account for more flexibility in real model
- [x] check mass and weight are correctly set
- [x] check gravity applying is correct.
- [x] check damping is correctly set
- [x] check scaling factors are correctly set

## Hypotheses
- [x] gravity loads are too big, our moments for design are too small, yes, this has been fixed
- [x] Some computation is incorrect so IMK numbers are wrong. models are unstable, yes models are unstable.
- [x] numerical rayleigh damping is incorrect ? not in bilin. check, rayleigh damping is correct.
- [x] crazy strategy of forcing convergence actually works, is there an explanation for this, yes.
- [x] Model is fundamentally wrong with negative inertias.
- [x] Kb, Ic computation to make it match is wrong., yes.
- [ ] Leave Ic as is, if it doesn't work, try out the n=10 method.
- [x] Opensees binary is wrong (unlikely), upgrade to a newer version to test out.
- [x] area too big, numerical instability (unlikely)


## Developing
### Adding new hazard curves
TODO: change this inconsistency and force users to provide them in SI units
provide them in (g)

### Adding new records
TODO: change this units inconsistency and force users to provide them in SI units
provide records in their base units, add a scale factor to `record-parameters.yml` such that spectra can be computed correctly in m/s/s


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
