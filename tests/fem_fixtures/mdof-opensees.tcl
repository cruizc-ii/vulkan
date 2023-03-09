#!/usr/local/bin/opensees
# ShearModelOpenSees, storeys None, bays None
wipe
model BasicBuilder -ndm 2 -ndf 3
geomTransf Linear 1
geomTransf PDelta 2
node 0 0.0 0.0
fix 0 1 1 1
node 2 0.0 1.0 -mass 1.0000 1e-9 1e-9
equalDOF 0 2 3
node 4 0.0 2.0 -mass 1.0000 1e-9 1e-9
equalDOF 0 4 3
node 6 0.0 3.0 -mass 1.0000 1e-9 1e-9
equalDOF 0 6 3
node 8 0.0 4.0 -mass 1.0000 1e-9 1e-9
equalDOF 0 8 3
node 10 0.0 5.0 -mass 1.0000 1e-9 1e-9
equalDOF 0 10 3
element elasticBeamColumn 0 0 2 1e+05 0.0416667 2 1
element elasticBeamColumn 3 2 4 1e+05 0.0416667 2 1
element elasticBeamColumn 6 4 6 1e+05 0.0416667 2 1
element elasticBeamColumn 9 6 8 1e+05 0.0416667 2 1
element elasticBeamColumn 12 8 10 1e+05 0.0416667 2 1
