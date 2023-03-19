model BasicBuilder -ndm 2 -ndf 3

node 1 0.0 0.0
node 2 3.0 0.0
fix 1 1 1 1
fix 2 1 1 1

node 3 0.0 12.0
node 4 3.0 12.0
equalDOF 3 4 1

node 5 0.0 24.0
node 6 3.0 24.0
equalDOF 5 6 1

node 7 0.0 36.0
node 8 3.0 36.0
equalDOF 7 8  1

node 9 0.0 48.0
node 10 3.0 48.0
equalDOF 9 10 1

node 11 0.0 60.0
node 12 3.0 60.0
equalDOF 11 12 1

mass 3 1.554 1e-9 1e-9
mass 4 1.554 1e-9 1e-9
mass 5 1.554 1e-9 1e-9
mass 6 1.554 1e-9 1e-9
mass 7 1.554 1e-9 1e-9
mass 8 1.554 1e-9 1e-9
mass 9 1.554 1e-9 1e-9
mass 10 1.554 1e-9 1e-9
mass 11 1.554 1e-9 1e-9
mass 12 1.554 1e-9 1e-9

geomTransf Linear 1
geomTransf Linear 2

# columns
element elasticBeamColumn 1 3 1 1e9 1.0 27250.5 2
element elasticBeamColumn 2 4 2 1e9 1.0 27250.5 2

element elasticBeamColumn 3 5 3 1e9 1.0 27250.5 2
element elasticBeamColumn 4 6 4 1e9 1.0 27250.5 2

element elasticBeamColumn 5 7 5 1e9 1.0 27250.5 2
element elasticBeamColumn 6 8 6 1e9 1.0 27250.5 2

element elasticBeamColumn 7 9 7 1e9 1.0 27250.5 2
element elasticBeamColumn 8 10 8 1e9 1.0 27250.5 2

element elasticBeamColumn 9 11 9 1e9 1.0 27250.5 2
element elasticBeamColumn 10 12 10 1e9 1.0 27250.5 2

# beams
element elasticBeamColumn 11 3 4  1e9 1.0 1e6 1
element elasticBeamColumn 12 5 6  1e9 1.0 1e6 1
element elasticBeamColumn 13 7 8  1e9 1.0 1e6 1
element elasticBeamColumn 14 9 10  1e9 1.0 1e6 1
element elasticBeamColumn 15 11 12 1e9 1.0 1e6 1


set eigenvalues [eigen 5]
set eigen_values_file [open /Users/carlo/vulkan/tests/strana_fixtures/chopra-1326/shear-frame/eigen-values.csv "w"]
puts $eigen_values_file $eigenvalues
close $eigen_values_file

recorder EnvelopeElement -file /Users/carlo/vulkan/tests/strana_fixtures/chopra-1326/shear-frame/base-columns-envelope.csv -ele 1 2 localForce
recorder NodeEnvelope -file /Users/carlo/vulkan/tests/strana_fixtures/chopra-1326/shear-frame/fixed-nodes.csv -node 1 2 -dof 1 3 reaction

set timeSeries "Series -dt 0.02 -filePath /Users/carlo/vulkan/records/elCentro.csv -factor -32.17"
pattern UniformExcitation 1 1 -accel $timeSeries
rayleigh 0.26862690983780524 0.00460700264785799 0 0
# rayleigh 0 0 0 0
constraints Transformation
numberer RCM
algorithm Linear
integrator Newmark 0.5 0.25
system BandGeneral
analysis Transient
analyze 4000 0.01
