#!/usr/local/bin/opensees
wipe
model BasicBuilder -ndm 2 -ndf 3
geomTransf Linear 1
geomTransf Linear 2
node 0 0.0 0.0
fix 0 1 1 1
node 1 6.0 0.0
fix 1 1 1 1
node 2 12.0 0.0
fix 2 1 1 1
node 3 0.0 3.5 -mass 73.39449541284404 1e-9 1e-9
node 4 6.0 3.5
node 5 12.0 3.5
node 6 0.0 7.0 -mass 73.39449541284404 1e-9 1e-9
node 7 6.0 7.0
node 8 12.0 7.0
uniaxialMaterial Elastic 2 1000000000.0
uniaxialMaterial Steel01 3 404.0 61115.498147287806 0.01
section Aggregator 1 2 P 3 Mz
element forceBeamColumn 0 0 3 5 1 2
uniaxialMaterial Elastic 5 1000000000.0
uniaxialMaterial Steel01 6 404.0 61115.498147287806 0.01
section Aggregator 4 5 P 6 Mz
element forceBeamColumn 1 1 4 5 4 2
uniaxialMaterial Elastic 8 1000000000.0
uniaxialMaterial Steel01 9 120.0 977847.9703566049 0.01
section Aggregator 7 8 P 9 Mz
element forceBeamColumn 2 3 4 5 7 1
uniaxialMaterial Elastic 11 1000000000.0
uniaxialMaterial Steel01 12 404.0 61115.498147287806 0.01
section Aggregator 10 11 P 12 Mz
element forceBeamColumn 3 2 5 5 10 2
uniaxialMaterial Elastic 14 1000000000.0
uniaxialMaterial Steel01 15 120.0 977847.9703566049 0.01
section Aggregator 13 14 P 15 Mz
element forceBeamColumn 4 4 5 5 13 1
uniaxialMaterial Elastic 17 1000000000.0
uniaxialMaterial Steel01 18 96.0 15278.87453682195 0.01
section Aggregator 16 17 P 18 Mz
element forceBeamColumn 5 3 6 5 16 2
uniaxialMaterial Elastic 20 1000000000.0
uniaxialMaterial Steel01 21 96.0 15278.87453682195 0.01
section Aggregator 19 20 P 21 Mz
element forceBeamColumn 6 4 7 5 19 2
uniaxialMaterial Elastic 23 1000000000.0
uniaxialMaterial Steel01 24 45.0 244461.9925891512 0.01
section Aggregator 22 23 P 24 Mz
element forceBeamColumn 7 6 7 5 22 1
uniaxialMaterial Elastic 26 1000000000.0
uniaxialMaterial Steel01 27 96.0 15278.87453682195 0.01
section Aggregator 25 26 P 27 Mz
element forceBeamColumn 8 5 8 5 25 2
uniaxialMaterial Elastic 29 1000000000.0
uniaxialMaterial Steel01 30 45.0 244461.9925891512 0.01
section Aggregator 28 29 P 30 Mz
element forceBeamColumn 9 7 8 5 28 1
