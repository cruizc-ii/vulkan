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
node 12 0.0 0.0
equalDOF 0 12 1 2
node 13 6.0 0.0
equalDOF 1 13 1 2
node 14 12.0 0.0
equalDOF 2 14 1 2
node 15 0.0 4.5
equalDOF 3 15 1 2
node 16 0.0 4.5
equalDOF 3 16 1 2
node 17 0.0 4.5
equalDOF 3 17 1 2
node 18 6.0 4.5
equalDOF 4 18 1 2
node 19 6.0 4.5
equalDOF 4 19 1 2
node 20 6.0 4.5
equalDOF 4 20 1 2
node 21 6.0 4.5
equalDOF 4 21 1 2
node 22 12.0 4.5
equalDOF 5 22 1 2
node 23 12.0 4.5
equalDOF 5 23 1 2
node 24 12.0 4.5
equalDOF 5 24 1 2
node 25 0.0 7.7
equalDOF 6 25 1 2
node 26 0.0 7.7
equalDOF 6 26 1 2
node 27 0.0 7.7
equalDOF 6 27 1 2
node 28 6.0 7.7
equalDOF 7 28 1 2
node 29 6.0 7.7
equalDOF 7 29 1 2
node 30 6.0 7.7
equalDOF 7 30 1 2
node 31 6.0 7.7
equalDOF 7 31 1 2
node 32 12.0 7.7
equalDOF 8 32 1 2
node 33 12.0 7.7
equalDOF 8 33 1 2
node 34 12.0 7.7
equalDOF 8 34 1 2
node 35 0.0 10.9
equalDOF 9 35 1 2
node 36 0.0 10.9
equalDOF 9 36 1 2
node 37 6.0 10.9
equalDOF 10 37 1 2
node 38 6.0 10.9
equalDOF 10 38 1 2
node 39 6.0 10.9
equalDOF 10 39 1 2
node 40 12.0 10.9
equalDOF 11 40 1 2
node 41 12.0 10.9
equalDOF 11 41 1 2
equalDOF 3 4 1
equalDOF 3 5 1
equalDOF 6 7 1
equalDOF 6 8 1
equalDOF 9 10 1
equalDOF 9 11 1
uniaxialMaterial Elastic 300001 1e9
uniaxialMaterial ModIMKPeakOriented 200001 203100.73 0.13 0.13 719.23 -719.23 16.259 16.259 16.259 16.259 1. 1. 1. 1. 0.01739455 0.01739455 0.01731115 0.01731115 1e-6 1e-6 0.03824692 0.03824692 1. 1.
section Aggregator 100001 300001 P 200001 Mz
element zeroLengthSection  1  0 12 100001
element elasticBeamColumn 2 12 15 1e+05 3e+07 -0.00702222 2
uniaxialMaterial Elastic 300003 1e9
uniaxialMaterial ModIMKPeakOriented 200003 203100.73 0.13 0.13 719.23 -719.23 16.259 16.259 16.259 16.259 1. 1. 1. 1. 0.01739455 0.01739455 0.01731115 0.01731115 1e-6 1e-6 0.03824692 0.03824692 1. 1.
section Aggregator 100003 300003 P 200003 Mz
element zeroLengthSection  3  15 3 100003
uniaxialMaterial Elastic 300004 1e9
uniaxialMaterial ModIMKPeakOriented 200004 203100.73 0.13 0.13 719.23 -719.23 16.259 16.259 16.259 16.259 1. 1. 1. 1. 0.01739455 0.01739455 0.01731115 0.01731115 1e-6 1e-6 0.03824692 0.03824692 1. 1.
section Aggregator 100004 300004 P 200004 Mz
element zeroLengthSection  4  1 13 100004
element elasticBeamColumn 5 13 18 1e+05 3e+07 -0.00702222 2
uniaxialMaterial Elastic 300006 1e9
uniaxialMaterial ModIMKPeakOriented 200006 203100.73 0.13 0.13 719.23 -719.23 16.259 16.259 16.259 16.259 1. 1. 1. 1. 0.01739455 0.01739455 0.01731115 0.01731115 1e-6 1e-6 0.03824692 0.03824692 1. 1.
section Aggregator 100006 300006 P 200006 Mz
element zeroLengthSection  6  18 4 100006
uniaxialMaterial Elastic 300007 1e9
uniaxialMaterial ModIMKPeakOriented 200007 152325.55 0.13 0.13 539.42 -539.42 16.259 16.259 16.259 16.259 1. 1. 1. 1. 0.01739455 0.01739455 0.01731115 0.01731115 1e-6 1e-6 0.03824692 0.03824692 1. 1.
section Aggregator 100007 300007 P 200007 Mz
element zeroLengthSection  7  3 16 100007
element elasticBeamColumn 8 16 21 1e+05 3e+07 1000 1
uniaxialMaterial Elastic 300009 1e9
uniaxialMaterial ModIMKPeakOriented 200009 152325.55 0.13 0.13 539.42 -539.42 16.259 16.259 16.259 16.259 1. 1. 1. 1. 0.01739455 0.01739455 0.01731115 0.01731115 1e-6 1e-6 0.03824692 0.03824692 1. 1.
section Aggregator 100009 300009 P 200009 Mz
element zeroLengthSection  9  21 4 100009
uniaxialMaterial Elastic 300010 1e9
uniaxialMaterial ModIMKPeakOriented 200010 203100.73 0.13 0.13 719.23 -719.23 16.259 16.259 16.259 16.259 1. 1. 1. 1. 0.01739455 0.01739455 0.01731115 0.01731115 1e-6 1e-6 0.03824692 0.03824692 1. 1.
section Aggregator 100010 300010 P 200010 Mz
element zeroLengthSection  10  2 14 100010
element elasticBeamColumn 11 14 22 1e+05 3e+07 -0.00702222 2
uniaxialMaterial Elastic 300012 1e9
uniaxialMaterial ModIMKPeakOriented 200012 203100.73 0.13 0.13 719.23 -719.23 16.259 16.259 16.259 16.259 1. 1. 1. 1. 0.01739455 0.01739455 0.01731115 0.01731115 1e-6 1e-6 0.03824692 0.03824692 1. 1.
section Aggregator 100012 300012 P 200012 Mz
element zeroLengthSection  12  22 5 100012
uniaxialMaterial Elastic 300013 1e9
uniaxialMaterial ModIMKPeakOriented 200013 152325.55 0.13 0.13 539.42 -539.42 16.259 16.259 16.259 16.259 1. 1. 1. 1. 0.01739455 0.01739455 0.01731115 0.01731115 1e-6 1e-6 0.03824692 0.03824692 1. 1.
section Aggregator 100013 300013 P 200013 Mz
element zeroLengthSection  13  4 19 100013
element elasticBeamColumn 14 19 24 1e+05 3e+07 1000 1
uniaxialMaterial Elastic 300015 1e9
uniaxialMaterial ModIMKPeakOriented 200015 152325.55 0.13 0.13 539.42 -539.42 16.259 16.259 16.259 16.259 1. 1. 1. 1. 0.01739455 0.01739455 0.01731115 0.01731115 1e-6 1e-6 0.03824692 0.03824692 1. 1.
section Aggregator 100015 300015 P 200015 Mz
element zeroLengthSection  15  24 5 100015
uniaxialMaterial Elastic 300016 1e9
uniaxialMaterial ModIMKPeakOriented 200016 51780.75 0.13 0.13 183.37 -183.37 16.258 16.258 16.258 16.258 1. 1. 1. 1. 0.01739455 0.01739455 0.01731115 0.01731115 1e-6 1e-6 0.03824692 0.03824692 1. 1.
section Aggregator 100016 300016 P 200016 Mz
element zeroLengthSection  16  3 17 100016
element elasticBeamColumn 17 17 25 1e+05 3e+07 -0.00103778 2
uniaxialMaterial Elastic 300018 1e9
uniaxialMaterial ModIMKPeakOriented 200018 51780.75 0.13 0.13 183.37 -183.37 16.258 16.258 16.258 16.258 1. 1. 1. 1. 0.01739455 0.01739455 0.01731115 0.01731115 1e-6 1e-6 0.03824692 0.03824692 1. 1.
section Aggregator 100018 300018 P 200018 Mz
element zeroLengthSection  18  25 6 100018
uniaxialMaterial Elastic 300019 1e9
uniaxialMaterial ModIMKPeakOriented 200019 51780.75 0.13 0.13 183.37 -183.37 16.258 16.258 16.258 16.258 1. 1. 1. 1. 0.01739455 0.01739455 0.01731115 0.01731115 1e-6 1e-6 0.03824692 0.03824692 1. 1.
section Aggregator 100019 300019 P 200019 Mz
element zeroLengthSection  19  4 20 100019
element elasticBeamColumn 20 20 28 1e+05 3e+07 -0.00103778 2
uniaxialMaterial Elastic 300021 1e9
uniaxialMaterial ModIMKPeakOriented 200021 51780.75 0.13 0.13 183.37 -183.37 16.258 16.258 16.258 16.258 1. 1. 1. 1. 0.01739455 0.01739455 0.01731115 0.01731115 1e-6 1e-6 0.03824692 0.03824692 1. 1.
section Aggregator 100021 300021 P 200021 Mz
element zeroLengthSection  21  28 7 100021
uniaxialMaterial Elastic 300022 1e9
uniaxialMaterial ModIMKPeakOriented 200022 38835.56 0.13 0.13 137.53 -137.53 16.258 16.258 16.258 16.258 1. 1. 1. 1. 0.01739455 0.01739455 0.01731115 0.01731115 1e-6 1e-6 0.03824692 0.03824692 1. 1.
section Aggregator 100022 300022 P 200022 Mz
element zeroLengthSection  22  6 26 100022
element elasticBeamColumn 23 26 31 1e+05 3e+07 1000 1
uniaxialMaterial Elastic 300024 1e9
uniaxialMaterial ModIMKPeakOriented 200024 38835.56 0.13 0.13 137.53 -137.53 16.258 16.258 16.258 16.258 1. 1. 1. 1. 0.01739455 0.01739455 0.01731115 0.01731115 1e-6 1e-6 0.03824692 0.03824692 1. 1.
section Aggregator 100024 300024 P 200024 Mz
element zeroLengthSection  24  31 7 100024
uniaxialMaterial Elastic 300025 1e9
uniaxialMaterial ModIMKPeakOriented 200025 51780.75 0.13 0.13 183.37 -183.37 16.258 16.258 16.258 16.258 1. 1. 1. 1. 0.01739455 0.01739455 0.01731115 0.01731115 1e-6 1e-6 0.03824692 0.03824692 1. 1.
section Aggregator 100025 300025 P 200025 Mz
element zeroLengthSection  25  5 23 100025
element elasticBeamColumn 26 23 32 1e+05 3e+07 -0.00103778 2
uniaxialMaterial Elastic 300027 1e9
uniaxialMaterial ModIMKPeakOriented 200027 51780.75 0.13 0.13 183.37 -183.37 16.258 16.258 16.258 16.258 1. 1. 1. 1. 0.01739455 0.01739455 0.01731115 0.01731115 1e-6 1e-6 0.03824692 0.03824692 1. 1.
section Aggregator 100027 300027 P 200027 Mz
element zeroLengthSection  27  32 8 100027
uniaxialMaterial Elastic 300028 1e9
uniaxialMaterial ModIMKPeakOriented 200028 38835.56 0.13 0.13 137.53 -137.53 16.258 16.258 16.258 16.258 1. 1. 1. 1. 0.01739455 0.01739455 0.01731115 0.01731115 1e-6 1e-6 0.03824692 0.03824692 1. 1.
section Aggregator 100028 300028 P 200028 Mz
element zeroLengthSection  28  7 29 100028
element elasticBeamColumn 29 29 34 1e+05 3e+07 1000 1
uniaxialMaterial Elastic 300030 1e9
uniaxialMaterial ModIMKPeakOriented 200030 38835.56 0.13 0.13 137.53 -137.53 16.258 16.258 16.258 16.258 1. 1. 1. 1. 0.01739455 0.01739455 0.01731115 0.01731115 1e-6 1e-6 0.03824692 0.03824692 1. 1.
section Aggregator 100030 300030 P 200030 Mz
element zeroLengthSection  30  34 8 100030
uniaxialMaterial Elastic 300031 1e9
uniaxialMaterial ModIMKPeakOriented 200031 16263.56 0.13 0.13 57.59 -57.59 16.258 16.258 16.258 16.258 1. 1. 1. 1. 0.01739455 0.01739455 0.01731115 0.01731115 1e-6 1e-6 0.03824692 0.03824692 1. 1.
section Aggregator 100031 300031 P 200031 Mz
element zeroLengthSection  31  6 27 100031
element elasticBeamColumn 32 27 35 1e+05 3e+07 -0.000336952 2
uniaxialMaterial Elastic 300033 1e9
uniaxialMaterial ModIMKPeakOriented 200033 16263.56 0.13 0.13 57.59 -57.59 16.258 16.258 16.258 16.258 1. 1. 1. 1. 0.01739455 0.01739455 0.01731115 0.01731115 1e-6 1e-6 0.03824692 0.03824692 1. 1.
section Aggregator 100033 300033 P 200033 Mz
element zeroLengthSection  33  35 9 100033
uniaxialMaterial Elastic 300034 1e9
uniaxialMaterial ModIMKPeakOriented 200034 16263.56 0.13 0.13 57.59 -57.59 16.258 16.258 16.258 16.258 1. 1. 1. 1. 0.01739455 0.01739455 0.01731115 0.01731115 1e-6 1e-6 0.03824692 0.03824692 1. 1.
section Aggregator 100034 300034 P 200034 Mz
element zeroLengthSection  34  7 30 100034
element elasticBeamColumn 35 30 37 1e+05 3e+07 -0.000336952 2
uniaxialMaterial Elastic 300036 1e9
uniaxialMaterial ModIMKPeakOriented 200036 16263.56 0.13 0.13 57.59 -57.59 16.258 16.258 16.258 16.258 1. 1. 1. 1. 0.01739455 0.01739455 0.01731115 0.01731115 1e-6 1e-6 0.03824692 0.03824692 1. 1.
section Aggregator 100036 300036 P 200036 Mz
element zeroLengthSection  36  37 10 100036
uniaxialMaterial Elastic 300037 1e9
uniaxialMaterial ModIMKPeakOriented 200037 16263.56 0.13 0.13 57.59 -57.59 16.258 16.258 16.258 16.258 1. 1. 1. 1. 0.01739455 0.01739455 0.01731115 0.01731115 1e-6 1e-6 0.03824692 0.03824692 1. 1.
section Aggregator 100037 300037 P 200037 Mz
element zeroLengthSection  37  9 36 100037
element elasticBeamColumn 38 36 39 1e+05 3e+07 1000 1
uniaxialMaterial Elastic 300039 1e9
uniaxialMaterial ModIMKPeakOriented 200039 16263.56 0.13 0.13 57.59 -57.59 16.258 16.258 16.258 16.258 1. 1. 1. 1. 0.01739455 0.01739455 0.01731115 0.01731115 1e-6 1e-6 0.03824692 0.03824692 1. 1.
section Aggregator 100039 300039 P 200039 Mz
element zeroLengthSection  39  39 10 100039
uniaxialMaterial Elastic 300040 1e9
uniaxialMaterial ModIMKPeakOriented 200040 16263.56 0.13 0.13 57.59 -57.59 16.258 16.258 16.258 16.258 1. 1. 1. 1. 0.01739455 0.01739455 0.01731115 0.01731115 1e-6 1e-6 0.03824692 0.03824692 1. 1.
section Aggregator 100040 300040 P 200040 Mz
element zeroLengthSection  40  8 33 100040
element elasticBeamColumn 41 33 40 1e+05 3e+07 -0.000336952 2
uniaxialMaterial Elastic 300042 1e9
uniaxialMaterial ModIMKPeakOriented 200042 16263.56 0.13 0.13 57.59 -57.59 16.258 16.258 16.258 16.258 1. 1. 1. 1. 0.01739455 0.01739455 0.01731115 0.01731115 1e-6 1e-6 0.03824692 0.03824692 1. 1.
section Aggregator 100042 300042 P 200042 Mz
element zeroLengthSection  42  40 11 100042
uniaxialMaterial Elastic 300043 1e9
uniaxialMaterial ModIMKPeakOriented 200043 16263.56 0.13 0.13 57.59 -57.59 16.258 16.258 16.258 16.258 1. 1. 1. 1. 0.01739455 0.01739455 0.01731115 0.01731115 1e-6 1e-6 0.03824692 0.03824692 1. 1.
section Aggregator 100043 300043 P 200043 Mz
element zeroLengthSection  43  10 38 100043
element elasticBeamColumn 44 38 41 1e+05 3e+07 1000 1
uniaxialMaterial Elastic 300045 1e9
uniaxialMaterial ModIMKPeakOriented 200045 16263.56 0.13 0.13 57.59 -57.59 16.258 16.258 16.258 16.258 1. 1. 1. 1. 0.01739455 0.01739455 0.01731115 0.01731115 1e-6 1e-6 0.03824692 0.03824692 1. 1.
section Aggregator 100045 300045 P 200045 Mz
element zeroLengthSection  45  41 11 100045

source modal.tcl
