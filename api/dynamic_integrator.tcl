#This integration scheme follows what has been canonically done in the IDA world, and it forces convergence to get a huge displacement
#more recent algorithms such as KrylovNewton can be faster and more precise and obtaining the collapse displacement, however when plotted they do not 'flatten', rather they spike up so it doesn't look that it is correct, altough it is.

set tol 1e-6
constraints Transformation
numberer RCM
# integrator TRBDF2
integrator Newmark 0.5 0.25
system BandGeneral
analysis Transient
test NormDispIncr $tol 10 0; # switch to 2 for output

set algorithms {
 "ModifiedNewton"
 "BFGS"
}


set num_algorithms [llength $algorithms]
set num_subdivisions 10
set max_subdivisions 2 ;# (exp) divide original timestep into num_sub^exp steps

set results_file $results_dir/results.csv
set collapse_file $results_dir/collapse.csv
# set results_file results.csv
set fp [open $results_file w+]

set aix 0
set break_outer 0
set converged 0
set time [getTime]
while {$time <= $duration && !$break_outer} {
    # reset to dt=record_dt, algorithm_0 = Newton
    set time [getTime]
    set analysis_dt $record_dt
    set algorithm [lindex $algorithms $aix]
    # puts "Trying $algorithm..."
    algorithm $algorithm
    set converged [analyze 1 $analysis_dt]
    set subdivision_retries 0
    # puts $converged
    while {$converged != 0} {
        # subdivide timestep
        incr subdivision_retries
        if {$subdivision_retries >= $max_subdivisions} {
            set time [getTime]
            puts $fp "Algorithm $algorithm did not converge at time $time $converged $subdivision_retries"
            incr aix
            if {$aix >= $num_algorithms} {
                set cf [open $collapse_file w+]
                puts $fp "FAILURE, couldn't converge with any algorithm or reduced step"
                puts $fp "FAILURE, couldn't converge with any algorithm or reduced step"
                puts $cf "FAILURE, couldn't converge with any algorithm or reduced step"
                puts $cf "time $time"
                # puts $fp "max drift $time"
                set break_outer 1
                close $cf
            }
            break
        }
        set n [expr {$num_subdivisions ** $subdivision_retries}]
        set analysis_dt [expr {$record_dt/$n}]
        set time [getTime]
        set converged [analyze $n $analysis_dt]
        puts $converged
        if {$converged == 0} {set aix 0}; # if we converged then reset to algorithm_0
    }
}

close $fp