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

## TODO

Source code has MIT licence.
Anyone can mount it on server.
Biblioteca unam has a .zip of the source code.

## TODO

- [x] push with 1st mode, plot modes of vibration
- [x] fix pushover stats, merge everything on a single call
- [x] callibrate str-elems+slabs cost
- [x] Compute Vy, uy correctly from pushover curve as 10% of original stiffness
- [x] change theta_y -> theta_y2 (Fardis)
- [x] for collapse, when analysis doesn't converge at the last subdivision, create an empty file called collapse.csv with drifts, printA, etc, then process into results.yml with info about the collapse

## TODO

- [ ] deploy to streamlit see how it works
- [x] aim for lower period in design procedure to account for more flexibility in real model, this doesnt work.
- [x] sustituir ya corrección de My/ke, cambiamos theta_y pero la actualizamos
- [x] design with a smaller period, so the extra flexibility doesn't hurt as much!
- [x] que fallo? debe ser alrededor de 15k/m2 ~0.6-1.2k USD/m2
- [x] hacer breakdown de los costos de Wasim en mi clasificacion.
- [x] incluir porcentaje de oficina transversal e.g. 2.7 veces cabe la distribucion de oficina que tiene dy=10m, el algoritmo de colocacion hace ese trabajo y multiplica asset.net_worth x 2.7
- [ ] incluir cimentacion como rugged asset always, 10% structural cost
- [ ] nonstructural glass windows drift sensitive
- [ ] libro de manuel alejandro con alcocer, hacer mejor los costos, desacoplar el acero longitudinal del transversal?
- [ ] ver publicaciones sobre IDAs con cortante, hablar con Vamvatsikos

cap6. columnas. comparar con cortante basal del metodo simplificado

refuerzo por integridad, para proteger del colapso progresivo por carga vertical

q=4 se revisa por capacidad
q=2 se revisa con cortante del analisis (mas laxo)

- [ ] echar ojo a la desagregación numeros negativos
- [ ] asegurar que el cambio preserva la inestabilidad dinamica (keep resolution but remove shear failure)

- [ ] rate of exceedance deaggregation

fix L=l0

count how many records exceeded without collapse
with collapse
and without collapse

then normalize

### Design

- [x] introduce PDelta.
- [x] fix deselecting model.yml when clicking 'run'
- [x] include more tests for CDMX design
- [x] check mass and weight are correctly set
- [x] check gravity applying is correct.
- [x] check damping is correctly set
- [x] check scaling factors are correctly set

## Loss Checks

- [x] AAL, 0.01 a bit low. 1e-4 2e-4 around 0.02%
- [x] EL v0 = AAL
- [x] v(L=0) = v0.

## Hypotheses

- [x] gravity loads are too big, our moments for design are too small, yes, this has been fixed
- [x] Some computation is incorrect so IMK numbers are wrong. models are unstable, yes models are unstable.
- [x] numerical rayleigh damping is incorrect ? not in bilin. check, rayleigh damping is correct.
- [x] crazy strategy of forcing convergence actually works, is there an explanation for this, yes.
- [x] Model is fundamentally wrong with negative inertias.
- [x] Kb, Ic computation to make it match is wrong., yes.
- [x] Leave Ic as is, if it doesn't work, try out the n=10 method.
- [x] Opensees binary is wrong (unlikely), upgrade to a newer version to test out.
- [x] area too big, numerical instability (unlikely)

## Developing

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

This error means pushover failed. Usually this means the model is wrong.
It's good that this is raised.

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
