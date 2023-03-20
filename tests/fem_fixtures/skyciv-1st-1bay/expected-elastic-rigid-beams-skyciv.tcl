#!/usr/local/bin/opensees
wipe
model BasicBuilder -ndm 2 -ndf 3
geomTransf Linear 1
geomTransf PDelta 2
node 0 0.0 0.0
fix 0 1 1 1
node 1 3.0 0.0
fix 1 1 1 1
node 2 0.0 2.5 -mass 30.5800 1e-9 1e-9
node 3 3.0 2.5
element elasticBeamColumn 0 0 2 1e+05 30000000.0 0.0007469 2
element elasticBeamColumn 1 1 3 1e+05 30000000.0 0.0007469 2
element elasticBeamColumn 2 2 3 1e+05 30000000.0 0.0003733 1
