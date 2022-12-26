set results_dir .
set eigenvalues [eigen -fullGenLapack 3]
set eigen_values_file [open $results_dir/eigen_values.csv "w"]
set omega {}
set T {}
set pi 3.141593

foreach value $eigenvalues {
        puts $value
        lappend omega [expr sqrt($value)]
        lappend T [expr (2*$pi)/sqrt($value)]
}
puts $eigen_values_file $T
puts "periods $T"
puts $eigen_values_file $omega 
close $eigen_values_file 

set eigen_vectors_file [open $results_dir/eigen_vectors.csv "w"]
set storeys {1 2 3}
set massNodes {3 6 9}
foreach mode $storeys {
  foreach m $massNodes {
      lappend eigenvector($mode) [lindex [nodeEigenvector $m $mode 1] 0]
  }
  puts $eigen_vectors_file $eigenvector($mode)
}
