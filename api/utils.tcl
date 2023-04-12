# ---
proc roundto {value decimals} {
  return [expr {double(round(10**$decimals * $value)/10.0**$decimals)}]
}

proc cum_sum_list {a_list} {
    set cum_sum 0
    foreach elem $a_list {
    set cum_sum [expr $cum_sum + $elem]
    lappend summed_list $cum_sum
}
    unset cum_sum
    return $summed_list
}

proc dot {l1 l2} {
    set len1 [llength $l1]
    set len2 [llength $l2]
    if {$len1 != $len2} {
        break
    }
    for {set i 0} {$i < $len1} {incr i} {
        lappend inner_product [expr [lindex $l1 $i]*[lindex $l2 $i]]
    }
    return [ladd $inner_product]
}

proc list_product {l1 l2 l3} {
    set len1 [llength $l1]
    set len2 [llength $l2]
    set len3 [llength $l3]
    if {$len1 != $len2 && $len2 != $len3} {
        break
    }

    for {set i 0} {$i < $len1} {incr i} {
        lappend product [expr [lindex $l1 $i] * [lindex $l2 $i] * [lindex $l3 $i]]
    }
    # puts $product
    return [ladd $product]
}

proc inertia {l} {
    return [expr [lindex $l 0]*pow([lindex $l 1], 3)/12]
}

proc area {l} {
    return [expr [lindex $l 0]*[lindex $l 1]]
}

proc ladd {l} {::tcl::mathop::+ {*}$l}

proc precision {number {digits 3}} {
    return [format "%.${digits}f" $number]
}

proc get_maximum_files {files} {
    foreach fil $files {
        set current_file [open $fil]
        while {[gets $current_file line] >= 0} {
            foreach string $line {
                lappend numbers $string
            }
        }
        close $current_file
    }
set maximum_number [tcl::mathfunc::max {*}$numbers]
return $maximum_number
}


proc get_envelope_files {files {skip_cols 0}} {
    set envelope 0

    foreach fil $files {
        set current_file [open $fil r]

        while {[gets $current_file line] >= 0} {
            set num 1

            foreach string $line {

                if {$skip_cols >= $num} {
                    incr num
                    continue
                } else {

                    if {[string is double -strict $string]} {

                        if {[expr {abs($string)}] > $envelope} {
                            set envelope [expr {abs($string)}]
                        }
                    }
                }
                incr num
            }
        }
        close $current_file
    }
return $envelope
}

set LAST_TAG 2041701730;# the idea is not to collide with any other tag

proc newtag {tagname} {
    set unique_tag [expr {$::LAST_TAG - 1}]
    set ::LAST_TAG $unique_tag
    upvar $tagname $tagname
    eval "set $tagname $unique_tag"
    return $unique_tag
}


proc linecount {file} {
    set i 0
    set fid [open $file r]
    while {[gets $fid line] > -1} {incr i}
    close $fid
    return $i
}

proc xrange {{from 0} {to 1} {step 0.1}} {
    # returns equivalent to xrange(from, to, step)

    for {set ix 0} true {incr ix} {
        set n [expr {$ix*$step + $from}]
        lappend range [format "%.2f" $n]
        if {$n >= $to} break
    }

    return $range
}
