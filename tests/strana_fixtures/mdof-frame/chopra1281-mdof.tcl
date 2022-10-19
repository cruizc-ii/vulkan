#!/usr/local/bin/opensees
wipe
model BasicBuilder -ndm 2 -ndf 3
geomTransf Linear 1
geomTransf Linear 2
node 0 0.0 0.0
fix 0 1 1 1
node 1 0.0 1.0 -mass 1.0 1e-9 1e-9
node 2 0.0 2.0 -mass 1.0 1e-9 1e-9
node 3 0.0 3.0 -mass 1.0 1e-9 1e-9
node 4 0.0 4.0 -mass 1.0 1e-9 1e-9
node 5 0.0 5.0 -mass 1.0 1e-9 1e-9
element elasticBeamColumn 0 0 1 1e6 2000000.0 1.0 2
element elasticBeamColumn 1 1 2 1e6 2000000.0 1.0 2
element elasticBeamColumn 2 2 3 1e6 2000000.0 1.0 2
element elasticBeamColumn 3 3 4 1e6 2000000.0 1.0 2
element elasticBeamColumn 4 4 5 1e6 2000000.0 1.0 2
