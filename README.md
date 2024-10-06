# README

# Quickstart

Make sure opensees is on your path, make a symlink

`cp ./bin/opensees /usr/local/bin`

```
cd /usr/local
sudo cp <opensees location> ./bin/opensees.bin
cd ./bin
sudo mv opensees.bin opensees
chmod -R 755 opensees
```

Install pyenv

```
pyenv install 3.9.6
direnv allow
mypy --install-types
pip install -r requirements.txt
make
```

## Debugging

Set a `breakpoint()` anywhere in the code.

## Dockerize it

```
alias podman=docker
podman machine init --cpus 2 --memory 4096 --disk-size 20
podman machine start
podman machine list
podman build -t vulkan .
podman run -p 8080:80 -t vulkan
http://127.0.0.1:80
```

## Change project name

`fly.toml > app`

## Adding secrets

Add to `.env`
Add to `fly secrets set`

## Access fly container

`fly ssh console`

## Developing

SI engineering units: force: kN, length: m, time: s.

### Adding new hazard curves

TODO: change units inconsistency and force users to provide them in SI units
provide them in (g)

### Adding new records

TODO: change units inconsistency and force users to provide them in SI units
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

## No columns to parse from file

This error means pushover/ida failed.
This means the model or the analysis is wrong.
It's good that this is raised. Check results files.

## loss

```
KeyError: 'pfa'
File "/Users/carlo/vulkan/app/assets.py", line 310, in dollars
    strana_results_df = self.dollars_for_storey(
File "/Users/carlo/vulkan/app/assets.py", line 374, in dollars_for_storey
    xs = strana_results_df[summary_edp].apply(get_edp_by_floor).values
File "/Users/carlo/vulkan/.direnv/python-3.9.6/lib/python3.9/site-packages/pandas/core/frame.py", line 3505, in __getitem__
    indexer = self.columns.get_loc(key)
File "/Users/carlo/vulkan/.direnv/python-3.9.6/lib/python3.9/site-packages/pandas/core/indexes/base.py", line 3623, in get_loc
    raise KeyError(key) from err
```

I think pfa is missing from strana_results_df, usually the code is commented out because that part of the code is hard.
