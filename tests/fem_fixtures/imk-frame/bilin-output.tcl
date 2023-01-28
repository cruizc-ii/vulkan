#!/usr/local/bin/opensees
wipe
model BasicBuilder -ndm 2 -ndf 3
geomTransf Linear 1
geomTransf PDelta 2
node 0 0.0 0.0
fix 0 1 1 1
node 1 6.0 0.0
fix 1 1 1 1
node 2 12.0 0.0
fix 2 1 1 1
node 3 0.0 4.5 -mass 73.3945 1e-9 1e-9
node 4 6.0 4.5
node 5 12.0 4.5
node 6 0.0 7.7 -mass 73.3945 1e-9 1e-9
node 7 6.0 7.7
node 8 12.0 7.7
node 9 0.0 10.9 -mass 73.3945 1e-9 1e-9
node 10 6.0 10.9
node 11 12.0 10.9
uniaxialMaterial Elastic 2 1000000000.0
uniaxialMaterial Steel01 3 719.2258654122501 550039.4833255903 0.0055
section Aggregator 1 2 P 3 Mz
element forceBeamColumn 0 0 3 5 1 1
uniaxialMaterial Elastic 5 1000000000.0
uniaxialMaterial Steel01 6 719.2258654122501 550039.4833255903 0.0055
section Aggregator 4 5 P 6 Mz
element forceBeamColumn 1 1 4 5 4 1
uniaxialMaterial Elastic 8 1000000000.0
uniaxialMaterial Steel01 9 539.4193990591875 8800631.733209444 0.0055
section Aggregator 7 8 P 9 Mz
element forceBeamColumn 2 3 4 5 7 1
uniaxialMaterial Elastic 11 1000000000.0
uniaxialMaterial Steel01 12 719.2258654122501 550039.4833255903 0.0055
section Aggregator 10 11 P 12 Mz
element forceBeamColumn 3 2 5 5 10 1
uniaxialMaterial Elastic 14 1000000000.0
uniaxialMaterial Steel01 15 539.4193990591875 8800631.733209444 0.0055
section Aggregator 13 14 P 15 Mz
element forceBeamColumn 4 4 5 5 13 1
uniaxialMaterial Elastic 17 1000000000.0
uniaxialMaterial Steel01 18 183.36741050058487 244461.9925891512 0.0055
section Aggregator 16 17 P 18 Mz
element forceBeamColumn 5 3 6 5 16 1
uniaxialMaterial Elastic 20 1000000000.0
uniaxialMaterial Steel01 21 183.36741050058487 244461.9925891512 0.0055
section Aggregator 19 20 P 21 Mz
element forceBeamColumn 6 4 7 5 19 1
uniaxialMaterial Elastic 23 1000000000.0
uniaxialMaterial Steel01 24 137.52555787543866 3911391.881426419 0.0055
section Aggregator 22 23 P 24 Mz
element forceBeamColumn 7 6 7 5 22 1
uniaxialMaterial Elastic 26 1000000000.0
uniaxialMaterial Steel01 27 183.36741050058487 244461.9925891512 0.0055
section Aggregator 25 26 P 27 Mz
element forceBeamColumn 8 5 8 5 25 1
uniaxialMaterial Elastic 29 1000000000.0
uniaxialMaterial Steel01 30 137.52555787543866 3911391.881426419 0.0055
section Aggregator 28 29 P 30 Mz
element forceBeamColumn 9 7 8 5 28 1
uniaxialMaterial Elastic 32 1000000000.0
uniaxialMaterial Steel01 33 51.81729216303763 61115.498147287806 0.0055
section Aggregator 31 32 P 33 Mz
element forceBeamColumn 10 6 9 5 31 1
uniaxialMaterial Elastic 35 1000000000.0
uniaxialMaterial Steel01 36 51.81729216303763 61115.498147287806 0.0055
section Aggregator 34 35 P 36 Mz
element forceBeamColumn 11 7 10 5 34 1
uniaxialMaterial Elastic 38 1000000000.0
uniaxialMaterial Steel01 39 38.86296912227822 977847.9703566049 0.0055
section Aggregator 37 38 P 39 Mz
element forceBeamColumn 12 9 10 5 37 1
uniaxialMaterial Elastic 41 1000000000.0
uniaxialMaterial Steel01 42 51.81729216303763 61115.498147287806 0.0055
section Aggregator 40 41 P 42 Mz
element forceBeamColumn 13 8 11 5 40 1
uniaxialMaterial Elastic 44 1000000000.0
uniaxialMaterial Steel01 45 38.86296912227822 977847.9703566049 0.0055
section Aggregator 43 44 P 45 Mz
element forceBeamColumn 14 10 11 5 43 1

source modal.tcl