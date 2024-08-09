#This integration scheme follows what has been canonically done in the IDA world, and it forces convergence to get a huge displacement
#more recent algorithms such as KrylovNewton can be faster and more precise and obtaining the collapse displacement, however when plotted they do not 'flatten', rather they spike up so it doesn't look that it is correct, altough it is.

set tol 1e-5
constraints Transformation
numberer RCM
integrator TRBDF2
# integrator HHT 0.6; #the smaller the alpha, the greater the numerical damping
system BandSPD
analysis Transient
test NormDispIncr $tol 1000 0; # switch to 2 for output
algorithm BFGS

# set algorithms {
#  "Newton"
#  "KrylovNewton"
#  "BFGS"
#  "Broyden"
# }

# set num_algorithms [llength $algorithms]
# set num_subdivisions 10
# set max_subdivisions 1 ;# (exp) divide original timestep into num_sub^exp steps

set results_file $results_dir/results.csv
set collapse_file $results_dir/collapse.csv
set fp [open $results_file w+]

set time [getTime]
set ok 0
set analysis_dt [expr {$record_dt/10}]
while {$time <= $duration && $ok == 0} {
    set ok [analyze 1 $record_dt]
    set time [getTime]
    if {$ok!=0} {
        set ok [analyze 1 $analysis_dt]
        set time [getTime]
    }
}
if {$ok != 0} {
    set cf [open $collapse_file w+]
    puts $fp "FAILURE, couldn't converge with any algorithm or reduced step"
    puts $cf "FAILURE, couldn't converge with any algorithm or reduced step"
    puts $cf "time $time"
    puts $fp "max drift $time"
}

close $fp
# close $cf
