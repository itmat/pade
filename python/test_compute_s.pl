#!/usr/bin/env perl

use strict;
use warnings;

sub ComputeS {
    my ($vector1_ref, $vector2_ref, $missing_value_designator, $min_presence1, $min_presence2) = @_;
    my @vector1 = @{$vector1_ref};
    my @vector2 = @{$vector2_ref};
    my $length_vector1=@vector1;
    my $length_vector2=@vector2;
    my $i;
    my $j;
    my $m;
    my $mean1;
    my $mean2;
    my @values1;
    my @values2;
    my $length_values1;
    my $length_values2;
    my $S;
    my $t;
    my $sd1;
    my $sd2;
    my $min_flag=0;
    $j=0;
    $m=0;

    # Make s be the sum of all non-missing values in vector 1
    for($i=0;$i<$length_vector1;$i++) {
	if(defined $vector1[$i] && $vector1[$i] ne $missing_value_designator) {
	    $values1[$j] = $vector1[$i];
	    $m=$m+$vector1[$i];
	    $j++;
	}
    }

    $length_values1 = $j;
    if($length_values1 < $min_presence1) {
	$min_flag=1;
    }
    else {
        print "Length values 1 is $length_values1\n";
	$mean1 = $m/$length_values1;
        print "Mean 1 is $mean1\n";
    }

    $j=0;
    $m=0;
    for($i=0;$i<$length_vector2;$i++) {
	if(defined $vector2[$i] && $vector2[$i] ne $missing_value_designator) {
	    $values2[$j] = $vector2[$i];
	    $m=$m+$vector2[$i];
	    $j++;
	}
    }
    $length_values2 = $j;
    if($length_values2 < $min_presence2) {
	$min_flag=1;
    }
    else {
	$mean2 = $m/$length_values2;
    }
    if($min_flag == 1) {
	$S = "NA";
    }
    else {
	$sd1 = 0;
	for($i=0; $i<$length_values1; $i++) {
	    $sd1 = $sd1 + ($values1[$i]-$mean1)**2;
	}
     
	$sd1 = sqrt($sd1/($length_values1-1));
	$sd2 = 0;
	for($i=0; $i<$length_values2; $i++) {
	    $sd2 = $sd2 + ($values2[$i]-$mean2)**2;
	}
	$sd2 = sqrt($sd2/($length_values2-1));
	$S = sqrt(($sd1**2*($length_values1-1) + $sd2**2*($length_values2-1))/($length_values1+$length_values2-2));
    }
    print "SDs are $sd1, $sd2\n";
    return $S;
}


print ComputeS([1, '', 5, 3, '', 9, 6, 3, 6, 8],
               [7, 4, 9, 6, 2, 4, 7, '', 2, 1],
               '',
               3, 3);
