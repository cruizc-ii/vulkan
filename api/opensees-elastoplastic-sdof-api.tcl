#!/usr/local/bin/opensees
source [file join [file dirname [info script]] "utils.tcl"]
# argc argv argv0
# All Tcl scripts have access to three predefined variables.
# $argc - number items of arguments passed to a script.
# $argv - list of the arguments.
# $argv0 - name of the script.
puts " -- Bilinear SDOF system analysis tool -- "
puts "============================"
puts "Provide name=<desired name> for your results folder to be created"
puts ""
puts "set outputdir to a relative or absolute filepath for the results, defaults to ../results"
puts "Variables (named arguments) required"
puts "1) Pushover. set pushover=1"
puts " du: maximum displacement (default 5*dy)"
puts "2) Time-history. set record variable"
puts " record: path+name of the record file with accelerations 'A' at each dt."
puts " period: (default 1)"
puts " damping: percentage of critical damping  (default 0.00)."
puts " dt: time intervals for each acceleration value (row) provided in 'record' (default 0.01)"
puts " scale: scales 'A's in record. (default 1.)"
puts "3) Spectra, set spectra=true variable, if pseudoacceleration Sa is needed pass (Sa=true), also
allows Sv=t"
puts "  - periods, a whitespace-separated list of values (defaults to 0.1 0.2 .. 5.0 with dT=0.1)"
puts " -----------"
puts "arguments: "
puts $argv
puts ""

foreach arg $argv {
    set kv [split $arg =]
    set [lindex $kv 0] [lindex $kv 1]
}

if {![info exists name]} {
    puts "Provider a name arg for your results folder to be created"
    return
}

if {![info exists fy]} {
    if {![info exists Ry]} {
        set fy 1e9
        set Ry 1e9
    }
}
if {![info exists outputdir]} {
    set results "../results"
} else {
    set results $outputdir
}

if {![info exists colname]} {
    set colname "globalForces"
}

puts $results

set results_dir ${results}/$name
# file mkdir $results_dir

if {![info exists period]} {
    set period 1.0
}

# mass will be 1. so fy=Ry. since Ry is the resistance per unit mass.
# stiffness is EA/L, but L=1 so k=EA. -> k/m = w^2 -> k = 4pi^2/T^2
set pi 3.14159
set k [expr {4*$pi**2 / $period**2}]
set dy [expr {$fy/$k}]

model BasicBuilder -ndm 2 -ndf 3

node 1 0.0 0.0
node 2 1. 0.0
mass 2 1. 1e-9 0
fix 1 1 1 1

uniaxialMaterial Steel01 99 $fy $k 1e-6
uniaxialMaterial Elastic 66 1
section Aggregator 10 99 P 66 Mz
geomTransf Linear 13
element nonlinearBeamColumn 17  1 2  5 10  13

set omega [expr sqrt([eigen 1])]
# puts "Computed period: "
# puts [expr {2*3.1416/$omega}]

if {[info exists pushover]} {
    file mkdir ${results_dir}/pushover
    puts "Pushover starting..."

    if {![info exists du]} {set du [expr {5.*$dy}]}

    recorder Node -file ${results_dir}/pushover/disp_1_2_3.csv -time -node 2 -dof 1 2 3 disp
    recorder Node -file ${results_dir}/pushover/reaction_1_2_3.csv -time -node 1 -dof 1 2 3 reaction;		# support reaction
    recorder Element -file ${results_dir}/pushover/localForce.csv -time -ele 17 localForce
    recorder Element -file ${results_dir}/pushover/globalForce.csv -time -ele 17 globalForce;						# element forces -- column
    # recorder Element -file ${results_dir}/pushover/sectionForce.csv -time -ele 17 section 10 force

    set dx 1.0e-5;
    set pushover_steps [expr int($du/$dx)]

    pattern Plain 200 Linear {
        load 2 -1.0 0 0
    }

    constraints Plain
    numberer RCM
    system BandGeneral
    test EnergyIncr 1.0e-9 10 0
    algorithm Newton
    integrator DisplacementControl 2 1 $dx
    analysis Static
    analyze $pushover_steps
    puts "Pushover done."

    wipeAnalysis
    remove recorders
    reset
}

if {[info exists record] } {
    if {![info exists dt]} {set dt 0.01}
    if {![info exists damping]} {set damping 0.}
    if {![info exists scale]} {set scale 1.}
    if {![info exists spectra]} {

        puts "------------------------------"
        puts "Transient analysis starting..."

        file mkdir ${results_dir}/timehistory

        recorder Node -file ${results_dir}/timehistory/disp_1_2_3.csv -time -node 2 -dof 1 2 3 disp
        recorder Node -file ${results_dir}/timehistory/vel_1_2_3.csv -time -node 2 -dof 1 2 3 vel
        recorder Node -file ${results_dir}/timehistory/accel_1_2_3.csv -time -node 2 -dof 1 2 3 accel
        recorder Node -file ${results_dir}/timehistory/reaction_1_2_3.csv -time -node 1 -dof 1 2 3 reaction; # support reaction

    # recorder Element -file ${results_dir}/timehistory/localForce.csv -time -ele 17 localForce
        recorder Element -file ${results_dir}/timehistory/${colname}.csv -time -ele 17 globalForce;						# element forces -- column
    # recorder Element -file ${results_dir}/timehistory/sectionForce.csv -time -ele 17 section 10 force

        recorder NodeEnvelope -file ${results_dir}/timehistory/disp_envelope.csv -time -node 2 -dof 1 disp
        recorder NodeEnvelope -file ${results_dir}/timehistory/vel_envelope.csv -time -node 2 -dof 1 vel
        recorder NodeEnvelope -file ${results_dir}/timehistory/accel_envelope.csv -time -node 2 -dof 1 accel

        recorder Drift -file ${results_dir}/timehistory/drifts_1.csv -time -iNode 1 -jNode 2 -perpDirn 1

        constraints Transformation
        numberer RCM
        system SparseGeneral
        test EnergyIncr 1.0e-4 10 0
        algorithm ModifiedNewton
        integrator Newmark 0.5 0.25
        analysis Transient
        rayleigh [expr 2 * $damping * $omega] 0 0 0

        set converged 0
        set time 0

        set Accel "Series -dt $dt -filePath $record -factor $scale"
        pattern UniformExcitation 1 1 -accel $Accel
        set duration [expr {$dt * [linecount $record]}]


    while {$converged == 0 && $time <= $duration} {
        set converged [analyze 1 $dt]

        if {$converged != 0} {
            puts "nonlinear behavior encountered, switching to ModifiedNewton..."
            test NormDispIncr 1e-9 100 0
            algorithm ModifiedNewton -initial
            set converged [analyze 1 $dt]

            if {$converged == 0} {
                puts "switching back to Newton..."
                test NormDispIncr 1e-6 10 0
                algorithm Newton
            }
        }

        set time [getTime]
       # puts $time
    }

    puts "Transient analysis complete."
    puts "---------------------------"
    } else {
        puts "Spectra starting..."

        # file mkdir ${results_dir}/spectra
        # set outfile ${results_dir}-${damping}.csv
        set outfile ${results_dir}
        set spectrafile [open $outfile w+]

        if {![info exists periods]} {
            set T0 0.01
            set Tn 5.0
            set dT 0.01
            set periods [xrange $T0 $Tn $dT]
        }

        foreach period $periods {

            set envelope ${results}/disp_envelope_${period}.csv
            set th ${results}/disp_th_${period}.csv

            set k [expr {4*$pi**2 / $period**2}]
            set omega [expr {2*$pi/ $period}]
            wipe
            model BasicBuilder -ndm 2 -ndf 3

            node 1 0.0 0.0
            node 2 1. 0.0
            mass 2 1. 1e-9 0
            fix 1 1 1 1

            uniaxialMaterial Steel01 99 $fy $k 1e-6
            uniaxialMaterial Elastic 66 1
            section Aggregator 10 99 P 66 Mz
            geomTransf Linear 13
            element nonlinearBeamColumn 17  1 2  5 10  13

            recorder EnvelopeNode -file $envelope -time -node 2 -dof 1 disp
            # recorder Node -file $th -time -node 2 -dof 1 disp

            set Accel "Series -dt $dt -filePath $record -factor $scale"
            pattern UniformExcitation 1 1 -accel $Accel
            set duration [expr {$dt * [linecount $record]}]

            constraints Transformation
            numberer RCM
            system SparseGeneral
            test EnergyIncr 1.0e-4 10 0
            algorithm ModifiedNewton
            integrator Newmark 0.5 0.25
            analysis Transient
            rayleigh [expr 2 * $damping * $omega] 0 0 0

            set converged 0
            set time 0

            while {$converged == 0 && $time <= $duration} {
                set converged [analyze 1 $dt]

                if {$converged != 0} {
                    puts "nonlinear behavior encountered, switching to ModifiedNewton..."
                    test NormDispIncr 1e-9 100 0
                    algorithm ModifiedNewton -initial
                    set converged [analyze 1 $dt]

                    if {$converged == 0} {
                        puts "switching back to Newton..."
                        test NormDispIncr 1e-6 10 0
                        algorithm Newton
                    }
                }

                set time [getTime]
            }
            remove recorders
            remove loadPattern 1
            wipeAnalysis
            reset

            set max_disp [get_envelope_files $envelope 1]
            file delete $envelope

            if {[info exists Sa]} {
                set max_disp [expr {$max_disp * $k}]
            } 

            if {[info exists Sv]} {
                set max_disp [expr {$max_disp * $omega}]
            }

            puts "$period $max_disp"
            lappend disps $max_disp
            puts $spectrafile "$period $max_disp"

        }

        close $spectrafile
    }
}


