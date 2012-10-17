#!/usr/bin/perl

#  ---------------------------------------------------------------
# | Copyright (c) 2004                                            |
# | The Computational Biology and Informatics Laboratory (CBIL)   |
# | The University of Pennsylvania.  All rights reserved.         |
# | By Gregory R Grant                                            |
#  ---------------------------------------------------------------

use strict;
use IO::File;
use Getopt::Long;
use Data::Dumper;
use autodie;

sub shape {
    my $data = shift;
    
    my @dims;

  I: for my $i (0 .. @{ $data } - 1) {
        if ($i > $dims[0]) {
            $dims[0] = $i;
        }
        next I if ! ref $data->[$i];

      J: for my $j ( 0 .. @{ $data->[$i] } -1 ) {
            if ($j > $dims[1]) {
                $dims[1] = $j;
            }
            next J if ! ref $data->[$i][$j];

            for my $k ( 0 .. @{ $data->[$i][$j] } -1 ) {
                if ($k > $dims[2]) {
                    $dims[2] = $k;
                }
            }
        }
    }

    return [ map { $_ + 1 } @dims ];
}

sub declaration {
    my ($name, $shape) = @_;
    return "$name = np.zeros((" . join(', ', @{ $shape } ) . "))\n";
}

sub np_array {
    my ($name, $items) = @_;
    my @items = map { $_ || 0 } @{ $items };
    return "$name = np.array([" . join(', ', @items) . "])\n";
}

sub assign_array {
    my ($name, $data, @indexes) = @_;
    my $res = '';
    if (ref($data)) {
        for my $i (0 .. @{ $data } - 1) {
            if ($data->[$i]) {
                $res .= assign_array($name, $data->[$i], @indexes, $i);
            }
        }
    }
    else {
        $res = $name . '[' . join(', ', @indexes) . '] = ' . $data . "\n";
    }
    return $res;
}

sub declare_and_assign {
    my ($name, $data) = @_;
    return declaration($name, shape($data)) . assign_array($name, $data);
}

$| = 1;

my ($help, $infile, $id2info, $id2url, $outfile, $min_presence, $min_presence_list, $shift, $level_confidence, $level_confidence_list, $aux_page_size, $usage, $silent_mode, $keep_outliers, $outlier_filter_strictness, $data_is_logged, $note, $medians, $median, $means, $mean, $tstat, $pvalstat, $num_perms, $only_report_outliers, $num_channels, $paired, $avalues, $design, $noavalues, $unpaired, $data_not_logged, $missing_value_designator, $id_filter_file, $no_background_filter, $background_filter_strictness, $use_logged_data, $use_unlogged_data, $num_bins, $pool, $output_gene_conf_list, $output_gene_conf_list_combined, $output_text, $tstat_tuning_param, $tstat_tuning_parameter);

GetOptions("help!"=>\$help, "usage!"=>\$usage, "infile=s"=>\$infile, "id2info:s"=>\$id2info, "id2url:s"=>\$id2url, "outfile=s"=>\$outfile, "min_presence:i"=>\$min_presence, "min_presence_list=s"=>\$min_presence_list, "shift:f"=>\$shift, "level_confidence:s"=>\$level_confidence, "level_confidence_list=s"=>\$level_confidence_list, "aux_page_size:i"=>\$aux_page_size, "silent_mode!"=>\$silent_mode, "no_outlier_filter!"=>\$keep_outliers, "outlier_filter_strictness:f"=>\$outlier_filter_strictness, "data_is_logged!"=>\$data_is_logged, "note=s"=>\$note, "medians!"=>\$medians, "median!"=>\$median, "means!"=>\$means, "mean!"=>\$mean, "tstat!"=>\$tstat, "pvalstat!"=>\$pvalstat,   "num_permutations:i"=>\$num_perms, "num_perms:i"=>\$num_perms, "only_report_outliers!"=>\$only_report_outliers, "num_channels:i"=>\$num_channels, "paired!"=>\$paired, "avalues!"=>\$avalues, "design=s"=>\$design,  "unpaired!"=>\$unpaired,  "noavalues!"=>\$noavalues,  "data_not_logged!"=>\$data_not_logged, "missing_value=s"=>\$missing_value_designator, "id_filter_file=s"=>\$id_filter_file, "no_background_filter!"=>\$no_background_filter, "background_filter_strictness:f"=>\$background_filter_strictness, "use_logged_data!"=>\$use_logged_data, "use_unlogged_data!"=>\$use_unlogged_data, "num_bins:i"=>\$num_bins, "pool!"=>\$pool, "output_gene_confidence_list!"=>\$output_gene_conf_list, "output_gene_confidence_list_combined!"=>\$output_gene_conf_list_combined, "output_text!"=>\$output_text, "tstat_tuning_param:f"=>\$tstat_tuning_param, "tstat_tuning_parameter:f"=>\$tstat_tuning_parameter);

if(!($level_confidence =~ /^(l|L)$/)) {
    $level_confidence = $level_confidence + 0;
}

print "\n-------------------------------------------------------------------------------";
print "\n|                            Welcome to PaGE 5.1.6                             |";
print "\n|                          microarray analysis tool                            |";
print "\n-------------------------------------------------------------------------------";

print "\n\n ****  PLEASE SET YOUR TERMINAL WINDOW TO AT LEAST 80 COLUMNS  ****\n";

if($help != 1) {
    if(!($id2info =~ /\S/)) {
	print "\n\nInclude a tab delimited file mapping ID's to descipritons with\n      --id2info <filename>\n";
    }
    if(!($id2url =~ /\S/)) {
	print "\nInclude a tab delimited file mapping ID's to URL's with\n      --id2url <filename>\n";
    }
}

if(!($help =~ /\S/)) {
    print "\nFor help use PaGE_5.1.pl --help\n\n";
}

if($mean == 1) {
    $means = 1;
}
if($median == 1) {
    $medians = 1;
}
if($tstat_tuning_parameter =~ /\S/) {
    $tstat_tuning_param = $tstat_tuning_parameter;
}

if($help || $usage) {
    &WriteHelp();
}

my $start_time = time();

if(!defined $aux_page_size) {
    $aux_page_size = 500;
}
if($aux_page_size < 10) {
    $aux_page_size = 10;
}

if(!defined $silent_mode) {
    $silent_mode=0;
}

if($use_unlogged_data == 1) {
    $use_logged_data = 0;
}

my $num_perms_default = 200;
if(!($num_perms =~ /\S/)) {
    $num_perms = $num_perms_default;
}
else {
    if($num_perms =~ /[^\d+]/) {
	print "\nNote: the num_perms you requested is not an integer.\nIt is being set to the default of $num_perms_default.\n\n";
	$num_perms = $num_perms_default;
    }
    if($num_perms < 2) {
	print "\nNote: the num_perms must be at least two.\nIt is being set to the default of $num_perms_default.\n\n";
	$num_perms = $num_perms_default;
    }
}

my $num_bins_default = 1000;
if(!($num_bins =~ /\S/)) {
    $num_bins = $num_bins_default;
}
else {
    if($num_bins =~ /[^\d+]/) {
	print "\nNote: the num_bins you requested is not an integer.\nIt is being set to the default of $num_bins_default.\n\n";
	$num_bins = $num_bins_default;
    }
    if($num_bins < 10) {
	print "\nNote: the num_bins must be at least ten.\nIt is being set to the default of $num_bins_default.\n\n";
	$num_bins = $num_bins_default;
    }
}

# Deal with the user input parameters, parsing, and initializing

my ($num_channels, $paired, $avalues, $design, $data_is_logged, $missing_value_designator, $no_background_filter, $background_filter_strictness, $level_confidence_array_ref, $min_presence_array_ref, $infile, $cond_ref, $rep_ref, $cond_A_ref, $rep_A_ref, $num_conds, $num_reps_ref, $num_cols, $tstat, $means, $medians, $use_logged_data, $use_unlogged_data, $pool, $outfile, $id2info_ref, $id2url_ref, $level_confidence) = GetUserInput($num_channels, $paired, $avalues, $design, $data_is_logged, $noavalues, $unpaired, $data_not_logged, $missing_value_designator, $no_background_filter, $background_filter_strictness, $level_confidence_list, $level_confidence, $min_presence, $min_presence_list, $infile, $medians, $means, $tstat, $use_logged_data, $use_unlogged_data, $pool, $outfile, $id2info, $id2url);



my $start;
if(!($design eq "D")) {
    $start = 1;
}
else {
    $start = 0;
}

my @alpha;
my @alpha_up;
my @alpha_down;
my @alpha_default;

if($tstat_tuning_param =~ /\S/) {
    for(my $i=$start; $i<$num_conds; $i++) {
	$alpha[$i] = $tstat_tuning_param;
    }
}
my %id2info;
my %id2url;
%id2info = %{$id2info_ref};
%id2url = %{$id2url_ref};
my @num_reps = @{$num_reps_ref};
my @min_presence_array = @{$min_presence_array_ref};
my @level_confidence_array = @{$level_confidence_array_ref};

$min_presence_list = $min_presence_array[0];
for(my $i=1; $i<$num_conds; $i++) {
    $min_presence_list = $min_presence_list . ",$min_presence_array[$i]";
}

if(!($pool =~ /1/)) {
    $pool = 0;
}

my $stat;
if($tstat == 1) {
    $stat = 0;
}
if($means == 1) {
    $stat = 1;
}
if($medians == 1) {
    $stat = 2;
}


# DEBUG
# print "num_channels=$num_channels\n";
# print "paired=$paired\n";
# print "avalues=$avalues\n";
# print "design=$design\n";
# print "data_is_logged=$data_is_logged\n";

# Read the infile

my ($data_ref, $data_A_ref, $ids_ref, $ids_hash_ref, $min_intensity, $max_intensity, $NEGS) = ReadInfile($num_channels, $paired, $avalues, $design, $data_is_logged, $use_logged_data, $shift, $infile, $noavalues, $unpaired, $data_not_logged, $missing_value_designator, $infile, $num_cols, $cond_ref, $rep_ref, $cond_A_ref, $rep_A_ref, $num_conds, $num_reps_ref, $num_cols);

my @data = @{$data_ref};
my @data_A = @{$data_A_ref};
my @ids = @{$ids_ref};
my %ids_hash = %{$ids_hash_ref};
my $num_ids = @ids;

if($NEGS == 1) {
    $shift = CheckForNegsSpecialCase($shift, $num_channels, $NEGS, $means, $medians, $min_intensity, $max_intensity, $use_logged_data);
}
else {
    if(!($shift =~ /^\d+$/)) {
	$shift = 0;
    }
}

my $unpermuted_means_ref = FindMeansUnpermuted(\@data, $num_conds, "missing");
my @unpermuted_means = @{$unpermuted_means_ref};

#print "num_conds = $num_conds\n";
#print "min_intensity = $min_intensity\n";
#print "max_intensity = $max_intensity\n";
#print "NEGS = $NEGS\n";

#PrintData(\@data, \@data_A, \@ids);

my $perms_ref;

$perms_ref = InitializePermuationArray($num_conds, \@num_reps, $num_perms, $paired, $design);

my @permutations = @{$perms_ref};

#PrintPermutationMatrix($num_conds, $perms_ref, $design);

my ($adjusted_data_ref) = AdjustData(\@data, $data_is_logged, $use_logged_data, $shift, "missing");
my @adjusted_data = @{$adjusted_data_ref};

#PrintData(\@adjusted_data, \@adjusted_data, \@ids);
#PrintData(\@data, \@data_A, \@ids);

my $alpha_default_ref;
if($tstat == 1) {
$alpha_default_ref = FindDefaultAlpha(\@adjusted_data, $stat, $num_conds, "missing", \@min_presence_array, $design, \@num_reps, $data_is_logged, $use_logged_data, $paired);
    @alpha_default = @{$alpha_default_ref};
}

# DEBUG
#for(my $i=$start; $i<$num_conds; $i++) {
#    print STDERR "alpha[$i] = $alpha[$i]\n";
#}
# DEBUG

my ($confidence_levels_hash_ref, $num_down_ref, $num_up_ref, $cutratios_down_ref, $cutratios_up_ref, $max_ref, $min_ref, $unpermuted_stat_ref, $gene_confidences_up_ref, $gene_confidences_down_ref, $level_confidence_array_ref, $alpha_up_ref, $alpha_down_ref, $breakdown) = DoConfidencesByCutoff($adjusted_data_ref, \@permutations, \@num_reps, $num_bins, $data_is_logged, $use_logged_data, $shift, $num_conds, $paired, "missing", \@min_presence_array, $stat, $pool, $design, \@ids, \@level_confidence_array, \@alpha, $level_confidence, \@alpha_default, $tstat_tuning_param);
@alpha_up = @{$alpha_up_ref};
@alpha_down = @{$alpha_down_ref};
my %confidence_levels_hash = %{$confidence_levels_hash_ref};
my %gene_confidences_up = %{$gene_confidences_up_ref};
my %gene_confidences_down = %{$gene_confidences_down_ref};

@level_confidence_array=@{$level_confidence_array_ref};
my @num_down=@{$num_down_ref};
my @num_up=@{$num_up_ref};
my @cutratios_down=@{$cutratios_down_ref};
my @cutratios_up=@{$cutratios_up_ref};
my @max = @{$max_ref};
my @min = @{$min_ref};
my @unpermuted_stat = @{$unpermuted_stat_ref};

# DEBUG
# foreach my $id (keys %confidence_levels_hash) {
#    for(my $cond=$start; $cond<$num_conds; $cond++) {
#	my $x = $confidence_levels_hash{$id}[$cond];
#	print "confidence_levels_hash{$id}[$cond] = $x\n";
#    }
# }
# DEBUG

# DEBUG
#for(my $i=$start; $i<$num_conds; $i++) {
#	print STDERR "cutratios_down[$i] = $cutratios_down[$i]\n";
#}
# DEBUG

my @cutoffs = &ComputeLevelCutoffs(\@cutratios_up, \@max, \@cutratios_down, \@min, \@unpermuted_stat, $design, $num_ids, $num_conds);

my ($patterns_ref, $num_levels_ref, $num_neg_levels_ref) = &CreatePatterns(\@cutoffs, \@unpermuted_stat, $num_conds, $design, $num_ids, \@ids);
my @num_levels = @{$num_levels_ref};
my @num_neg_levels = @{$num_neg_levels_ref};
my %patterns = %{$patterns_ref};
my $clusters_ref = &ClusterPatterns(\%patterns, $num_conds, $design);
my %clusters = %{$clusters_ref};
my $pattern_list_ref = &CreateSortedPatternList(\%clusters);
my @pattern_list = @{$pattern_list_ref};
$pattern_list_ref = &ReorderPatterns(\@pattern_list);
my @pattern_list = @{$pattern_list_ref};
# following is a placeholder until outliers routine defined...
my %outliers;

my $headerstuff = WriteHtmlOutput(\@data, \%confidence_levels_hash, \@num_reps, $num_conds, $silent_mode, $note, $shift, \@min, \@max, $design,\@cutratios_up, \@cutratios_down, \@cutoffs, \@num_neg_levels, \@num_levels, \@pattern_list, \@num_up, \@num_down, \%clusters, $aux_page_size, $outfile, $id2info, $id2url, \%ids_hash, \@unpermuted_means, \@unpermuted_stat, \%outliers, \@level_confidence_array, $output_text, \@alpha_up, \@alpha_down, $paired, $data_is_logged, $use_logged_data, $tstat, $means, $medians, $num_perms, \@min_presence_array, $breakdown);
# If requested, print the gene confidences for every gene, to a text file

if($output_gene_conf_list == 1) {
    $outfile =~ s/.txt$//i;
    $outfile =~ s/.html$//i;
    $outfile =~ s/.htm$//i;
    $outfile =~ s/.text$//i;
    my $gene_conf_outfile = $outfile . "-gene_conf_list.txt";
    open(GENECONFLISTOUTFILE, ">$gene_conf_outfile");
    print GENECONFLISTOUTFILE "$headerstuff\n";
    for(my $i=$start; $i<$num_conds; $i++) {
	print GENECONFLISTOUTFILE "Condition $i Upregulated Genes\n-----------\n";
	foreach my $id (sort {$gene_confidences_up{$b}[$i]<=>$gene_confidences_up{$a}[$i]} keys %gene_confidences_up) {
	    my $xx = $gene_confidences_up{$id}[$i];
	    print GENECONFLISTOUTFILE "$id\t$xx\n";
	}
	print GENECONFLISTOUTFILE "\nCondition $i Downregulated Genes\n-----------\n";
	foreach my $id (sort {$gene_confidences_down{$b}[$i]<=>$gene_confidences_down{$a}[$i]} keys %gene_confidences_down) {
	    my $xx = $gene_confidences_down{$id}[$i];
	    print GENECONFLISTOUTFILE "$id\t$xx\n";
	}
    }
    close(GENECONFLISTOUTFILE);
}
if($output_gene_conf_list_combined == 1) {
    my %gene_confidences_combined;
    for(my $i=$start; $i<$num_conds; $i++) {
	foreach my $id (sort {$gene_confidences_up{$b}[$i]<=>$gene_confidences_up{$a}[$i]} keys %gene_confidences_up) {
	    my $xx = $gene_confidences_up{$id}[$i];
	    $gene_confidences_combined{$id}[$i] = $xx;
	}
	print GENECONFLISTOUTFILE "\nCondition $i Downregulated Genes\n-----------\n";
	foreach my $id (sort {$gene_confidences_down{$b}[$i]<=>$gene_confidences_down{$a}[$i]} keys %gene_confidences_down) {
	    my $xx = $gene_confidences_down{$id}[$i];
	    my $yy = $gene_confidences_up{$id}[$i];
	    if($xx > $yy) {
		$gene_confidences_combined{$id}[$i] = $xx;
	    }
	    else {
		$gene_confidences_combined{$id}[$i] = $yy;
	    }
	}
    }
    $outfile =~ s/.txt$//i;
    $outfile =~ s/.html$//i;
    $outfile =~ s/.htm$//i;
    $outfile =~ s/.text$//i;
    my $gene_conf_outfile = $outfile . "-gene_conf_list_combined.txt";
    open(GENECONFLISTOUTFILE, ">$gene_conf_outfile");
    print GENECONFLISTOUTFILE "$headerstuff\n";
    for(my $i=$start; $i<$num_conds; $i++) {
	print GENECONFLISTOUTFILE "Condition $i\n-----------\n";
	foreach my $id (sort {$gene_confidences_combined{$b}[$i]<=>$gene_confidences_combined{$a}[$i]} keys %gene_confidences_combined) {
	    my $xx = $gene_confidences_combined{$id}[$i];
	    print GENECONFLISTOUTFILE "$id\t$xx\n";
	}
    }
    close(GENECONFLISTOUTFILE);
}


my $duration = time()-$start_time;
my $time;
my $minutes;
if($duration > 59) {
    $minutes=int($duration/60);
    my $remainder=$duration % 60;
    $time = "$minutes minutes $remainder seconds";
}
else {
    $time="$duration seconds";
}
if($minutes>59) {
    my $hours=int($minutes/60);
    my $remainder=$minutes % 60;
    $time = "$hours hours $remainder minutes";
}
print "\nRun time: $time.\n\n";


# SUBROUTINES START HERE

sub ReorderPatterns {
    my ($pattern_list_ref, $num_conds) = @_;
    my @pattern_list = @{$pattern_list_ref};
    my $n = @pattern_list;
    my $aa;
    my $bb;
    if($num_conds==2) {
	my $c=0;
	my @pos;
	my @negs;
	while($pattern_list[$c]<0) {
	    $negs[$c]=$pattern_list[$c];
	    $c++;
	}
	for(my $i=$c;$i<$n;$i++) {
	    $pos[$i-$c ]=$pattern_list[$i];
	}
	$aa=@negs;
	$bb=@pos;
	for(my $i=0;$i<$bb;$i++) {
	    $pattern_list[$i]=$pos[$bb-1-$i];
	}
	for(my $i=$bb; $i<$bb+$aa; $i++) {
	    $pattern_list[$i]=$negs[$i-$bb];
	}
    }
    return \@pattern_list;
}


sub CreateSortedPatternList {
    my ($clusters_ref) = @_;
    my %clusters = %{$clusters_ref};
    my @pattern_list;

    my $i = 0;
    foreach my $tag (keys %clusters) {
	$pattern_list[$i] = $tag;
	$i++;
    }

    my $flag = 0;
    my $n = @pattern_list-1;
    while ($flag==0) {
	$flag =1;
	for($i=0; $i<$n; $i++) {
	    my @pattern1 = split(/,/, $pattern_list[$i]);
	    my @pattern2 = split(/,/, $pattern_list[$i+1]);
	    my $j = 0;
	    while ($pattern1[$j]==$pattern2[$j] && $j<@pattern1-1) {
		$j++;
	    }
	    if ($pattern1[$j]<$pattern2[$j]) {
		$flag = 0;
		my $temp = $pattern_list[$i];
		$pattern_list[$i] = $pattern_list[$i+1];
		$pattern_list[$i+1] = $temp;
	    }
	}
    }

    return \@pattern_list;
}


sub ClusterPatterns {
    my ($patterns_ref, $num_conds, $design) = @_;
    my %patterns = %{$patterns_ref};
    my %cluster;

    if($design ne "D") {
	$start = 1;
    }
    else {
	$start = 0;
    }
    foreach my $tag (keys %patterns) {
	my $string = "";
	for (my $cond=$start; $cond< $num_conds-1; $cond++) {
	    if($patterns{$tag}[$cond] =~ /\S/) {
		$string .= $patterns{$tag}[$cond].",";
	    }
	    else {
		$string .= "0,";
	    }
	}
	if($patterns{$tag}[$num_conds-1] =~ /\S/) {
	    $string .= $patterns{$tag}[$num_conds-1];
	}
	else {
		$string .= "0";
	}
	my $n = (defined $cluster{$string}) ? @{$cluster{$string}} : 0;
	if(($string=~/[1-9]/)) {
	    $cluster{$string}[$n] = $tag;
	}
    }
    return \%cluster;
}


sub CreatePatterns {
    my ($cutoffs_ref, $unpermuted_stat_ref, $num_conds, $design, $num_ids, $ids_ref) = @_;
    my @cutoffs = @{$cutoffs_ref};
    my @unpermuted_stat = @{$unpermuted_stat_ref};
    my @num_levels;
    my @num_neg_levels;
    my @ids = @{$ids_ref};
    my %patterns;
    my $center;
    if($design ne "D") {
	$start = 1;
    }
    else {
	$start = 0;
    }
    if($stat == 0) {
	$center = 0;
    }
    else {
	$center = 1;
    }

    for (my $cond=$start; $cond<$num_conds; $cond++) {
	$num_levels[$cond] = @{$cutoffs[$cond]}+1;
	my $j=0;
	my $temp = @{$cutoffs[$cond]};
	while ($cutoffs[$cond][$j]<$center && $j<$temp) {
	    $j++;
	}
	$num_neg_levels[$cond] = $j;
    }

    for(my $id=0; $id<$num_ids; $id++) {
	for (my $cond=$start; $cond<$num_conds; $cond++) {
	    my $j=0;
	    while ($unpermuted_stat[$id][$cond]>$cutoffs[$cond][$j] && $j<@{$cutoffs[$cond]}) {
		$j++;
	    }

	    my $temp = $j-$num_neg_levels[$cond];
	    $patterns{$ids[$id]}[$cond] = $temp;
	}
    }

    return (\%patterns, \@num_levels, \@num_neg_levels);
}


sub ComputeLevelCutoffs {
    my ($cutratios_up_ref, $max_ref, $cutratios_down_ref, $min_ref, $unpermuted_stat_ref, $design, $num_ids, $num_conds)=@_;
    @cutratios_up=@{$cutratios_up_ref};
    @max=@{$max_ref};
    @cutratios_down=@{$cutratios_down_ref};
    @min=@{$min_ref};
    @unpermuted_stat = @{$unpermuted_stat_ref};
    my @output;
    my $start;
    if($design ne "D") {
	$start = 1;
    }
    else {
	$start = 0;
    }

    for (my $cond=$start; $cond<$num_conds; $cond++) {
	my @cutofflist;
	my $currentlevel;
	if($stat == 0) {
	    $currentlevel = 0;
	    until ($cutratios_up[$cond]+$currentlevel>=$max[$cond]) {
		$currentlevel += $cutratios_up[$cond];
		push (@cutofflist, $currentlevel);
	    }
	}
	else {
	    $currentlevel = 1;
	    until ($cutratios_up[$cond]*$currentlevel>=$max[$cond]) {
		$currentlevel *= $cutratios_up[$cond];
		push (@cutofflist, $currentlevel);
	    }
	}
	if($stat==0) {
	    $currentlevel = 0;
	    until ($currentlevel + $cutratios_down[$cond]<=$min[$cond]) {
		$currentlevel += $cutratios_down[$cond];
		unshift (@cutofflist, $currentlevel);
	    }
	}
	else {
	    $currentlevel = 1;
	    my $left_count = 1;
	    # must keep into consideration the fact that $min[$cond] could be 0
	    until ($cutratios_down[$cond]*$currentlevel<=$min[$cond] || $left_count == 0) {
		$currentlevel *= $cutratios_down[$cond];
		unshift (@cutofflist, $currentlevel);
		$left_count = 0;
		for(my $id=0; $id<$num_ids; $id++) {
		    if ($unpermuted_stat[$id][$cond]<$currentlevel && $unpermuted_stat[$id][$cond]>0) {
			$left_count++;
		    }
		}
	    }
	}
	$output[$cond] = \@cutofflist;
    }

    return @output;
}


sub AdjustData {
    my ($data_ref, $data_is_logged, $use_logged_data, $shift, $missing_value_designator) = @_;
    my @data = @{$data_ref};
    my @data_adjusted;
    my $num_ids = @data;
    my $num_conds = @{$data[1]};
    for(my $k=0; $k<$num_ids; $k++) {
	for(my $kk=0; $kk<$num_conds; $kk++) {
	    my $num_reps = @{$data[$k][$kk]};
	    for(my $kkk=0; $kkk<$num_reps; $kkk++) {
		my $val = $data[$k][$kk][$kkk];
		if($val ne $missing_value_designator && defined $data[$k][$kk][$kkk]) {
		    if($use_logged_data == 1 && $data_is_logged == 0) {
			$data_adjusted[$k][$kk][$kkk] = log($val+$shift);
		    }
		    else {
			if($use_logged_data == 0 && $data_is_logged == 1) {
			    $data_adjusted[$k][$kk][$kkk] = exp($val+$shift);
			}
			else {
			    $data_adjusted[$k][$kk][$kkk] = $val + $shift;
			}
		    }
		}
		else {
		    $data_adjusted[$k][$kk][$kkk] = $data[$k][$kk][$kkk]
		}
	    }
	}
    }
    return \@data_adjusted;
}

sub CheckForNegsSpecialCase {
    my ($shift, $num_channels, $NEGS, $means, $medians, $min_intensity, $max_intensity, $use_logged_data) = @_;
# check for NEGS special cases here

    if(($shift =~ /\S/) && (!($shift =~ /^\d*.?\d*/))) {
	die("\nError: The shift you specified is not a valid number\n\n");
    }

    if(($num_channels == 1) && ($NEGS == 1 && (($means eq "1" || $medians eq "1") || $use_logged_data == 1))) {
	if(!(($shift =~ /\S/) && ($shift > $min_intensity))) {
	    my $answer;
	    if($means eq "1") {
		print "You have requested the mean statistic, but you have negative intensities.\n";
	    }
	    if($medians eq "1") {
		print "You have requested the median statistic, but you have negative intensities.\n";
	    }
	    if($use_logged_data == 1) {
		print "You have requested to log transform the data, but you have negative intensities.\n";
	    }
	    print "You can proceed only if you shift the data to be all positive.\n";
	    my $suggested_shift = ($max_intensity - 5 * $min_intensity)/4;
	    print "A shift of $suggested_shift is suggested.\n";
	    print "Do you want to use this shift\nEnter Y to use $suggested_shift, or N to enter a different shift: ";
	    $answer = <STDIN>;
	    chomp($answer);
	    while(!($answer eq "Y" || $answer eq "y") && !($answer eq "N" || $answer eq "n")) {
		print "\nPlease enter Y or N: ";
		$answer = <STDIN>;
		chomp($answer);
	    }
	    if($answer eq "Y" || $answer eq "y") {
		$shift = $suggested_shift;
	    }
	    else {
		print "\nEnter the desired shift: ";
		$answer = <STDIN>;
		chomp($answer);
		while(!($answer =~ /^\d+.?\d*/) || (!($answer =~ /^\d*.?\d+/) || $answer + $min_intensity < 0)) {
		    my $temp = $min_intensity * -1;
		    print "\nThat shift is not valid, it must be a number greater than $temp\n";
		    print "\nEnter the desired shift: ";
		    $answer = <STDIN>;
		    chomp($answer);
		}
		$shift = $answer;
	    }
	}
    }

    if($shift =~ /\S/) {
	print "\nIntensities will be shifted by $shift\n";
    }
    else {
	$shift = 0;
    }
    return $shift;
}

sub DoConfidencesByCutoff {

    my ($data_ref, $permutations_ref, $num_reps_ref, $num_bins, $data_is_logged, $use_logged_data, $shift, $num_conds, $paired, $missing_value_designator, $min_presence_array_ref, $stat, $pool, $design, $ids_ref, $level_confidence_ref, $alpha_ref, $level_confidence, $alpha_default_ref, $tstat_tuning_param) = @_;
    open my $mean_perm_up_fh, '>', 'mean_perm_up.py';
    print $mean_perm_up_fh "import numpy as np\n";
    print $mean_perm_up_fh declare_and_assign('data', $data_ref);
    print $mean_perm_up_fh declare_and_assign('default_alphas', $alpha_default_ref);

    my @alpha_default = @{$alpha_default_ref};
    my @ids = @{$ids_ref};
    my @data = @{$data_ref};
    my %confidence_levels_hash;
    my %gene_confidences_up;
    my %gene_confidences_down;
    my @gene_confidences_up_vect;
    my @gene_confidences_down_vect;
    my @level_confidence_array = @{$level_confidence_ref};
    my @cutratios_up;
    my @cutratios_down;
    my @permutations = @{$permutations_ref};
    my @min_presence_array = @{$min_presence_array_ref};
    my @num_reps = @{$num_reps_ref};
    my $num_ids = @data;
    my @num_null_down;
    my @num_null_up;
    my @num_null_down_vect;
    my @num_null_up_vect;
    my @CONF_bins_down;
    my @CONF_bins_up;
    my @CONF_bins_down_vect;
    my @CONF_bins_up_vect;
    my @alpha = @{$alpha_ref};
    my $center;
    my $start;
    my @num_up;
    my @num_down;
    my @D;
    my $num_range_values = 10;
    my $min_ref;
    my $max_ref;
    my @min;
    my @max;
    my @min_vect;
    my @max_vect;
    my $min_vect_ref;
    my $max_vect_ref;
    my @num_unpooled_up_vect;
    my @num_unpooled_down_vect;
    my @dist_pooled_up_vect;
    my @dist_pooled_down_vect;
    my @mean_perm_up_vect;
    my @mean_perm_down_vect;
    my $R;
    my $V;
    my @R_vect;
    my @V_vect;
    my @max_conf_down_vect;
    my @max_conf_up_vect;
    my @max_number_up;
    my @max_number_down;
    my @max_up_index;
    my @max_down_index;
    my @num_up_by_conf_vect;
    my @num_down_by_conf_vect;
    my @alpha_up;
    my @alpha_down;
    my $breakdown;

# hard coded in several places
    my @tuning_param_range_values;
    $tuning_param_range_values[0] = .0001;
    $tuning_param_range_values[1] = .01;
    $tuning_param_range_values[2] = .1;
    $tuning_param_range_values[3] = .3;
    $tuning_param_range_values[4] = .5;
    $tuning_param_range_values[5] = 1;
    $tuning_param_range_values[6] = 1.5;
    $tuning_param_range_values[7] = 2;
    $tuning_param_range_values[8] = 3;
    $tuning_param_range_values[9] = 10;

    if(!($design eq "D")) {
	$start = 1;
    }
    else {
	$start = 0;
    }
    if($stat==0) {
	$center = 0;
    }
    else {
	$center = 1;
    }
    if($stat == 0 && !($tstat_tuning_param =~ /\S/)) {
	($min_vect_ref, $max_vect_ref) = FindMaxMinValueOfStatVect(\@data, $stat, $num_conds, $missing_value_designator, $design, $min_presence_array_ref, \@alpha, $data_is_logged, $use_logged_data, \@alpha_default, $paired);
	@min_vect = @{$min_vect_ref};
	@max_vect = @{$max_vect_ref};
    }
    else {
	($min_ref, $max_ref) = FindMaxMinValueOfStat(\@data, $stat, $num_conds, $missing_value_designator, $design, $min_presence_array_ref, \@alpha, $data_is_logged, $use_logged_data, $paired);
	@min = @{$min_ref};
	@max = @{$max_ref};
    }


# DEBUG
#    for(my $i=$start; $i<$num_conds; $i++) {
#	print STDERR "min[$i]=$min[$i]\n";
#	print STDERR "max[$i]=$max[$i]\n";
#    }
# DEBUG
    my $val;
    my @val_vect;
    my $val_vect_ref;
    my $min_presence1 = $min_presence_array[0];
    my @dist_pooled_up;
    my @dist_pooled_down;
    my @num_unpooled_up;
    my @num_unpooled_down;
    my @mean_perm_up;
    my @mean_perm_down;
    for(my $cond=$start; $cond<$num_conds; $cond++) {
	my $temp_num = $num_conds - $start;
	my $temp_cond = $cond + 1 - $start;
	if($silent_mode == 0) {
	    print "\n* working on condition $temp_cond of $temp_num\n\n";
	}
	my $min_presence2 = $min_presence_array[$cond];
	my $num_perms = @{$permutations[$cond]};
	for(my $perm=0; $perm<$num_perms; $perm++) {
	    my $perm_plus = $perm + 1;
	    if($silent_mode == 0) {
		print "permutation $perm_plus of $num_perms\n";
	    }
	    my @temp_up;
	    my @temp_down;
	    my @temp_up_vect;
	    my @temp_down_vect;
	    for(my $id=0; $id<$num_ids; $id++) {
		my @vector1;
		my @vector2;
		my $v1=0;
		my $v2=0;
		if(!($design eq "D")) {
		    if($paired == 0) {
			for(my $j=1; $j<$num_reps[0]+1; $j++) {
			    if($permutations[$cond][$perm][$j-1] == 1) {
				$vector1[$v1] = $data[$id][0][$j];
				$v1++;
			    }
			    else {
				$vector2[$v2] = $data[$id][0][$j];
				$v2++;
			    }
			}
			for(my $j=$num_reps[0]+1; $j<$num_reps[0] + $num_reps[$cond] + 1; $j++) {
			    if($permutations[$cond][$perm][$j-1] == 1) {
				$vector1[$v1] = $data[$id][$cond][$j-$num_reps[0]];
				$v1++;
			    }
			    else {
				$vector2[$v2] = $data[$id][$cond][$j-$num_reps[0]];
				$v2++;
			    }
			}
		    }
		    else {
			for(my $j=1; $j<$num_reps[0]+1; $j++) {
			    if($permutations[$cond][$perm][$j-1] == 1) {
				$vector1[$v1] = $data[$id][0][$j];
				$vector2[$v1] = $data[$id][$cond][$j];
			    }
			    else {
				$vector1[$v1] = $data[$id][$cond][$j];
				$vector2[$v1] = $data[$id][0][$j];
			    }
			    $v1++;
			}
		    }
		}
		else {
		    for(my $j=1; $j<$num_reps[$cond]+1; $j++) {
			if($permutations[$cond][$perm][$j-1] == 0) {
			    $vector1[$v1] = $data[$id][$cond][$j];
			    $v1++;
			}
			else {
			    if($stat == 0) {
				$vector1[$v1] = -$data[$id][$cond][$j];
			    }
			    else {
				$vector1[$v1] = 1/$data[$id][$cond][$j];
			    }
			    $v1++;
			}
		    }
		}
# DEBUG
#		print "--------------------------\nid=$id, cond=$cond\n";
#	    my $l = @{$permutations[$cond][$perm]};
#	    for(my $k=0; $k<$l; $k++) {
#		print "permutations[$cond][$perm][$k]=$permutations[$cond][$perm][$k]\n";
#	    }
#	    my $l1 = @vector1;
#	    my $l2 = @vector2;
#	    for(my $k=0; $k<$l1; $k++) {
#		print "vector1[$k]=$vector1[$k]\n";
#	    }
#	    for(my $k=0; $k<$l2; $k++) {
#		print "vector2[$k]=$vector2[$k]\n";
#	    }
# DEBUG

		if(!($design eq "D")) {
		    if($stat==0) {
			if($paired==0) {
			    if($tstat_tuning_param =~ /\S/) {
				$val = ComputeTstat(\@vector2,\@vector1, $missing_value_designator, $min_presence2, $min_presence1, $alpha[$cond]);
			    }
			    else {
				my $val_vect_ref = ComputeTstatVector(\@vector2,\@vector1, $missing_value_designator, $min_presence2, $min_presence1, $alpha_default[$cond]);
				@val_vect = @{$val_vect_ref};
			    }
			}
			else {
			    if($tstat_tuning_param =~ /\S/) {
				$val = ComputePairedTstat(\@vector2,\@vector1, $missing_value_designator, $min_presence2, $min_presence1, $alpha[$cond], $data_is_logged, $use_logged_data);
			    }
			    else {
				my $val_vect_ref = ComputePairedTstatVector(\@vector2,\@vector1, $missing_value_designator, $min_presence2, $min_presence1, $alpha_default[$cond], $data_is_logged, $use_logged_data);
				@val_vect = @{$val_vect_ref};
			    }
			}
		    }
 		    if($stat==1) {
			if($paired==0) {
			    my $val1 = ComputeMean(\@vector2, $missing_value_designator, $min_presence2);
			    $val = ComputeMean(\@vector1, $missing_value_designator, $min_presence1);
			    if($val ne "NA" && $val1 ne "NA") {
				$val = $val1/$val;
			    }
			    else {
				$val = "NA";
			    }
#			    print "RATIO = $val\n";
			}
			else {
			    $val = ComputePairedMean(\@vector2,\@vector1, $missing_value_designator, $min_presence2, $min_presence1);
			}
		    }
		    if($stat==2) {
			my $val1 = ComputeMedian(\@vector2, $missing_value_designator, $min_presence2);
			$val = ComputeMedian(\@vector1, $missing_value_designator, $min_presence1);
			if($val ne "NA" && $val1 ne "NA") {
			    $val = $val1/$val;
			}
			else {
			    $val = "NA";
			}
		    }
		}
		else {
		    if($stat == 0) {
			if($tstat_tuning_param =~ /\S/) {
			    $val = ComputeOneSampleTstat(\@vector1, $missing_value_designator, $min_presence1, $alpha[$cond], $use_logged_data);
			}
			else {
			    my $val_vect_ref = ComputeOneSampleTstatVector(\@vector1, $missing_value_designator, $min_presence1, $alpha_default[$cond], $use_logged_data);
			    @val_vect = @{$val_vect_ref};
			}
		    }
		    else {
			$val = ComputeGeometricMean(\@vector1, $missing_value_designator, $min_presence1);
		    }
		}
		my $bin;
		if($stat == 0 && !($tstat_tuning_param =~ /\S/)) {
		    for(my $i=0; $i<$num_range_values; $i++) {
			$val = $val_vect[$i];
			if($val ne "NA") {
			    if($val>=$center) {
				if($val>$max_vect[$i][$cond]) {
				    $temp_up_vect[$i][$num_bins]++;
				}
				else {
				    $bin=int($num_bins * ($val - $center) / ($max_vect[$i][$cond] - $center));
				    $temp_up_vect[$i][$bin]++;
				}
				$temp_down_vect[$i][0]++;
			    }
			    if($val<=$center) {
				if($val<$min_vect[$i][$cond]) {
				    $temp_down_vect[$i][$num_bins]++;
				}
				else {
				    $bin=int($num_bins * (-$val - $center) / (-$min_vect[$i][$cond] - $center));
				    $temp_down_vect[$i][$bin]++;
				}
				$temp_up_vect[$i][0]++;
			    }
			}
		    }
		}
		else {
		    if($val ne "NA") {
			if($pool == 1) {
			    if($val>=$center) {
				if($val>$max[$cond]) {
				    $dist_pooled_up[$cond][$num_bins]++;
				}
				else {
				    $bin=int($num_bins * ($val - $center) / ($max[$cond] - $center));
				    $dist_pooled_up[$cond][$bin]++;
				}
				$dist_pooled_down[$cond][0]++;
			    }
			    if($val<=$center) {
				if($val<$min[$cond]) {
				    $dist_pooled_down[$cond][$num_bins]++;
				}
				else {
				    if($stat == 0) {
					$bin=int($num_bins * (-$val - $center) / (-$min[$cond] - $center));
				    }
				    else {
					$bin=int($num_bins * (1/$val - $center) / (1/$min[$cond] - $center));
				    }
				    $dist_pooled_down[$cond][$bin]++;
				}
				$dist_pooled_up[$cond][0]++;
			    }
			}
			else {
			    if($val>=$center) {
				if($val>$max[$cond]) {
				    $temp_up[$num_bins]++;
				}
				else {
				    $bin=int($num_bins * ($val - $center) / ($max[$cond] - $center));
				    $temp_up[$bin]++;
				}
				$temp_down[0]++;
			    }
			    if($val<=$center) {
				if($val<$min[$cond]) {
				    $temp_down[$num_bins]++;
				}
				else {
				    if($stat == 0) {
					$bin=int($num_bins * (-$val - $center) / (-$min[$cond] - $center));
				    }
				    else {
					$bin=int($num_bins * (1/$val - $center) / (1/$min[$cond] - $center));
				    }
				    $temp_down[$bin]++;
				}
				$temp_up[0]++;
			    }
			}
		    }
		}
	    }
# DEBUG
#	    print "perm = $perm\n";
#	    for(my $j=0; $j<$num_range_values; $j++) {
#		for(my $i=0; $i<$num_bins; $i++) {
#		    print "temp_up_vect[$j][$i] = $temp_up_vect[$j][$i]\n";
#		}
#	    }
# DEBUG

	    if($pool == 0) {
# after this loop temp_up[$i] holds number permuted stats greater or equal to that bin
		if($stat == 0 && !($tstat_tuning_param =~ /\S/)) {
		    for(my $j=0; $j<$num_range_values; $j++) {
			for(my $i=$num_bins-1; $i>=0; $i--) {
			    $temp_up_vect[$j][$i] = $temp_up_vect[$j][$i] + $temp_up_vect[$j][$i+1];
			    $temp_down_vect[$j][$i] = $temp_down_vect[$j][$i] + $temp_down_vect[$j][$i+1];
			}
			for(my $i=0; $i<$num_bins+1; $i++) {
# ultimately will divide num_unpooled_up[$i] by num_perms to get mean value of temp_up[$i]
# over all perms
			    $num_unpooled_up_vect[$j][$cond][$i] = $num_unpooled_up_vect[$j][$cond][$i] + $temp_up_vect[$j][$i];
			    $num_unpooled_down_vect[$j][$cond][$i] = $num_unpooled_down_vect[$j][$cond][$i] + $temp_down_vect[$j][$i];
			}
		    }
		}
		else {
		    for(my $i=$num_bins-1; $i>=0; $i--) {
			$temp_up[$i] = $temp_up[$i] + $temp_up[$i+1];
			$temp_down[$i] = $temp_down[$i] + $temp_down[$i+1];
		    }
		    for(my $i=0; $i<$num_bins+1; $i++) {
# ultimately will divide num_unpooled_up[$i] by num_perms to get mean value of temp_up[$i]
# over all perms
			$num_unpooled_up[$cond][$i] = $num_unpooled_up[$cond][$i] + $temp_up[$i];
			$num_unpooled_down[$cond][$i] = $num_unpooled_down[$cond][$i] + $temp_down[$i];
#			$D[$i][$temp_up[$i]]++;
		    }
		}
	    }
	}
# DEBUG
#	print "******************************************\n";
#		for(my $i=0; $i<$num_bins+1; $i++) {
#		    print "bin $i\n";
#		    for(my $j=0; $j<1001; $j++) {
#			$D[$i][$j] = $D[$i][$j]+0;
#			if($D[$i][$j] > 0) {
#			    print "D[$i][$j]=$D[$i][$j]\t";
#			}
#		    }
#		    print "\n";
#		}
# DEBUG

# DEBUG
#	print "******************************************\n";
#	for(my $j=0; $j<$num_range_values; $j++) {
#	    for(my $i=0; $i<$num_bins+1; $i++) {
#		print "num_unpooled_up_vect[$j][$cond][$i]=$num_unpooled_up_vect[$j][$cond][$i]\n";
#	    }
#	    for(my $i=0; $i<$num_bins+1; $i++) {
#		print "num_unpooled_down_vect[$j][$cond][$i]=$num_unpooled_down_vect[$j][$cond][$i]\n";
#	    }
#	}
#	print "******************************************\n";
# DEBUG


# the following computes the $mean_perm_up and $mean_perm_down arrays, which give the mean
# number up and down for each bin over all permutations

	if($stat == 0 && !($tstat_tuning_param =~ /\S/)) {
	    for(my $j=0; $j<$num_range_values; $j++) {
		for(my $i=$num_bins-1; $i>=0; $i--) {
		    $dist_pooled_up_vect[$j][$cond][$i] = $dist_pooled_up_vect[$j][$cond][$i] + $dist_pooled_up_vect[$j][$cond][$i+1];
		    $dist_pooled_down_vect[$j][$cond][$i] = $dist_pooled_down_vect[$j][$cond][$i] + $dist_pooled_down_vect[$j][$cond][$i+1];
		}
		for(my $i=0; $i<$num_bins+1; $i++) {
		    if($pool == 0) {
			$mean_perm_up_vect[$j][$cond][$i] = $num_unpooled_up_vect[$j][$cond][$i]/$num_perms;
			$mean_perm_down_vect[$j][$cond][$i] = $num_unpooled_down_vect[$j][$cond][$i]/$num_perms;
		    }
		    else {
			$mean_perm_up_vect[$j][$cond][$i] = $dist_pooled_up_vect[$j][$cond][$i]/$num_perms;
			$mean_perm_down_vect[$j][$cond][$i] = $dist_pooled_down_vect[$j][$cond][$i]/$num_perms;
		    }
		}
	    }
	}
	else {
	    for(my $i=$num_bins-1; $i>=0; $i--) {
		$dist_pooled_up[$cond][$i] = $dist_pooled_up[$cond][$i] + $dist_pooled_up[$cond][$i+1];
		$dist_pooled_down[$cond][$i] = $dist_pooled_down[$cond][$i] + $dist_pooled_down[$cond][$i+1];
	    }
	    for(my $i=0; $i<$num_bins+1; $i++) {
		if($pool == 0) {
		    $mean_perm_up[$cond][$i] = $num_unpooled_up[$cond][$i]/$num_perms;
		    $mean_perm_down[$cond][$i] = $num_unpooled_down[$cond][$i]/$num_perms;
		}
		else {
		    $mean_perm_up[$cond][$i] = $dist_pooled_up[$cond][$i]/$num_perms;
		    $mean_perm_down[$cond][$i] = $dist_pooled_down[$cond][$i]/$num_perms;
		}
	    }
	}

# DEBUG
#	print "******************************************\n";
#	    for(my $j=0; $j<$num_range_values; $j++) {
#		for(my $i=0; $i<$num_bins+1; $i++) {
#		    print "mean_perm_up_vect[$j][$cond][$i]=$mean_perm_up_vect[$j][$cond][$i]\n";
#		}
#		for(my $i=0; $i<$num_bins+1; $i++) {
#		    print "mean_perm_down_vect[$j][$cond][$i]=$mean_perm_down_vect[$j][$cond][$i]\n";
#		}
#	    }
#	print "******************************************\n";
# DEBUG

# DEBUG
#	print "******************************************\n";
#		for(my $i=0; $i<$num_bins+1; $i++) {
#		    print "dist_pooled_up[$cond][$i]=$dist_pooled_up[$cond][$i]\n";
#		}
#	        for(my $i=0; $i<$num_bins+1; $i++) {
#		    print "dist_pooled_down[$cond][$i]=$dist_pooled_down[$cond][$i]\n";
#		}
#	print "******************************************\n";
# DEBUG

    }
    print $mean_perm_up_fh declare_and_assign('mean_perm_up', \@mean_perm_up_vect);

  
    my $num_unpermuted_up_ref;
    my $num_unpermuted_down_ref;
    my @num_unpermuted_up;
    my @num_unpermuted_down;
    my $num_unpermuted_up_vect_ref;
    my $num_unpermuted_down_vect_ref;
    my @num_unpermuted_up_vect;
    my @num_unpermuted_down_vect;
    my @unpermuted_stat_vect;
    my $unpermuted_stat_vect_ref;
    my @unpermuted_stat_vect;
    my $unpermuted_stat_ref;
    my @unpermuted_stat;

# The folowing computes statistics for the unpermuted data

    if($stat == 0 && !($tstat_tuning_param =~ /\S/)) {
      warn "Got into first block\n";
      ($num_unpermuted_up_vect_ref, $num_unpermuted_down_vect_ref, $unpermuted_stat_vect_ref) = FindDistUnpermutedStatVect(\@data, $stat, $num_conds, $missing_value_designator, \@max_vect, \@min_vect, \@min_presence_array, $design, \@alpha, $data_is_logged, $use_logged_data, $alpha_default_ref, $paired);

      open my $up_fh,  '>', 'perl_unperm_up';
        open my $down_fh,  '>', 'perl_unperm_down';
        open my $stats_fh,  '>', 'perl_unperm_stats';

        print $up_fh    Dumper($num_unpermuted_up_vect_ref);
        print $down_fh  Dumper($num_unpermuted_down_vect_ref);
        print $stats_fh Dumper($unpermuted_stat_vect_ref);

	@num_unpermuted_up_vect = @{$num_unpermuted_up_vect_ref};
	@num_unpermuted_down_vect = @{$num_unpermuted_down_vect_ref};
	@unpermuted_stat_vect = @{$unpermuted_stat_vect_ref};
# DEBUG
#	for(my $j=0; $j<$num_range_values; $j++) {
#	    for(my $bin=0; $bin<$num_bins+1; $bin++) {
#		print "num_unpermuted_up_vect[$j][1][$bin]=$num_unpermuted_up_vect[$j][1][$bin]\n";
#		print "num_unpermuted_down_vect[$j][1][$bin]=$num_unpermuted_down_vect[$j][1][$bin]\n";
#	    }
#	}
#	for(my $j=0; $j<$num_range_values; $j++) {
#	    for(my $id=0; $id<1000; $id++) {
#		print "unpermuted_stat_vect[$j][$id][1]=$unpermuted_stat_vect[$j][$id][1]\n";
#	    }
#	}
# DEBUG
    }
    else {
      warn "Got into second block";
	($num_unpermuted_up_ref, $num_unpermuted_down_ref, $unpermuted_stat_ref) = FindDistUnpermutedStat(\@data, $stat, $num_conds, $missing_value_designator, \@max, \@min, \@min_presence_array, $design, \@alpha, $data_is_logged, $use_logged_data, $paired);
	@num_unpermuted_up = @{$num_unpermuted_up_ref};
	@num_unpermuted_down = @{$num_unpermuted_down_ref};
	@unpermuted_stat = @{$unpermuted_stat_ref};
    }
    warn "Got past there";

    for(my $cond=$start; $cond<$num_conds; $cond++) {
	if($stat == 0 && !($tstat_tuning_param =~ /\S/)) {
            for(my $j=0; $j<$num_range_values; $j++) {
		for(my $bin=$num_bins; $bin>=0; $bin--) {
		    $num_unpermuted_up_vect[$j][$cond][$bin] = $num_unpermuted_up_vect[$j][$cond][$bin] + $num_unpermuted_up_vect[$j][$cond][$bin+1];
		    $num_unpermuted_down_vect[$j][$cond][$bin] = $num_unpermuted_down_vect[$j][$cond][$bin] + $num_unpermuted_down_vect[$j][$cond][$bin+1];
		}

		for(my $bin=0; $bin<$num_bins+1; $bin++) {
		    $V_vect[$j] = $mean_perm_up_vect[$j][$cond][$bin];
		    $R_vect[$j] = $num_unpermuted_up_vect[$j][$cond][$bin];
		    $num_null_up_vect[$j][$cond][$bin] = AdjustNumDiff($V_vect[$j],$R_vect[$j],$num_ids);
		    $V_vect[$j] = $mean_perm_down_vect[$j][$cond][$bin];
		    $R_vect[$j] = $num_unpermuted_down_vect[$j][$cond][$bin];
		    $num_null_down_vect[$j][$cond][$bin] = AdjustNumDiff($V_vect[$j],$R_vect[$j],$num_ids);
		}
	      }
	  }

	else {
	    for(my $bin=$num_bins; $bin>=0; $bin--) {
		$num_unpermuted_up[$cond][$bin] = $num_unpermuted_up[$cond][$bin] + $num_unpermuted_up[$cond][$bin+1];
		$num_unpermuted_down[$cond][$bin] = $num_unpermuted_down[$cond][$bin] + $num_unpermuted_down[$cond][$bin+1];
	    }

	    for(my $bin=0; $bin<$num_bins+1; $bin++) {
		$V = $mean_perm_up[$cond][$bin];
		$R = $num_unpermuted_up[$cond][$bin];
		$num_null_up[$cond][$bin] = AdjustNumDiff($V,$R,$num_ids);
		$V = $mean_perm_down[$cond][$bin];
		$R = $num_unpermuted_down[$cond][$bin];
		$num_null_down[$cond][$bin] = AdjustNumDiff($V,$R,$num_ids);
	    }
	}

# DEBUG
#	for(my $j=0; $j<$num_range_values; $j++) {
#	    for(my $bin=0; $bin<$num_bins+1; $bin++) {
#		print "num_null_down_vect[$j][$cond][$bin] = $num_null_down_vect[$j][$cond][$bin]\n";
#	    }
#	}
# DEBUG
      
	if($stat == 0 && !($tstat_tuning_param =~ /\S/)) {
            for(my $j=0; $j<$num_range_values; $j++) {
		for(my $bin=0; $bin<$num_bins+1; $bin++) {
		    if($num_unpermuted_up_vect[$j][$cond][$bin] > 0) {
			$CONF_bins_up_vect[$j][$cond][$bin] = ($num_unpermuted_up_vect[$j][$cond][$bin] - $num_null_up_vect[$j][$cond][$bin]) / $num_unpermuted_up_vect[$j][$cond][$bin];
		    }
		    else {
			$CONF_bins_up_vect[$j][$cond][$bin] = 0;
		    }
		    if($num_unpermuted_down_vect[$j][$cond][$bin] > 0) {
			$CONF_bins_down_vect[$j][$cond][$bin] = ($num_unpermuted_down_vect[$j][$cond][$bin] - $num_null_down_vect[$j][$cond][$bin]) / $num_unpermuted_down_vect[$j][$cond][$bin];
		    }
		    else {
			$CONF_bins_down_vect[$j][$cond][$bin] = 0;
		    }
		    if($CONF_bins_up_vect[$j][$cond][$bin] < 0) {
			$CONF_bins_up_vect[$j][$cond][$bin] = 0;
		    }
		    if($CONF_bins_down_vect[$j][$cond][$bin] < 0) {
			$CONF_bins_down_vect[$j][$cond][$bin] = 0;
		    }
		}
	      }

# DEBUG
#	for(my $j=0; $j<$num_range_values; $j++) {
#	    for(my $bin=0; $bin<$num_bins+1; $bin++) {
#		print "CONF_bins_up_vect[$j][$cond][$bin] = $CONF_bins_up_vect[$j][$cond][$bin]\n";
#	    }
#	}
#	for(my $j=0; $j<$num_range_values; $j++) {
#	    for(my $bin=0; $bin<$num_bins+1; $bin++) {
#		print "CONF_bins_down_vect[$j][$cond][$bin] = $CONF_bins_down_vect[$j][$cond][$bin]\n";
#	    }
#	}
# DEBUG

	}

	else {
	    for(my $bin=0; $bin<$num_bins+1; $bin++) {
		if($num_unpermuted_up[$cond][$bin] > 0) {
		    $CONF_bins_up[$cond][$bin] = ($num_unpermuted_up[$cond][$bin] - $num_null_up[$cond][$bin]) / $num_unpermuted_up[$cond][$bin];
		}
		else {
		    $CONF_bins_up[$cond][$bin] = 0;
		}
		if($num_unpermuted_down[$cond][$bin] > 0) {
		    $CONF_bins_down[$cond][$bin] = ($num_unpermuted_down[$cond][$bin] - $num_null_down[$cond][$bin]) / $num_unpermuted_down[$cond][$bin];
		}
		else {
		    $CONF_bins_down[$cond][$bin] = 0;
		}
		if($CONF_bins_up[$cond][$bin] < 0) {
		    $CONF_bins_up[$cond][$bin] = 0;
		}
		if($CONF_bins_down[$cond][$bin] < 0) {
		    $CONF_bins_down[$cond][$bin] = 0;
		}
	    }
	}

	if($stat == 0 && !($tstat_tuning_param =~ /\S/)) {
            for(my $j=0; $j<$num_range_values; $j++) {
		my $running = $CONF_bins_up_vect[$j][$cond][0];
		$max_conf_up_vect[$j] = 0;
		for(my $bin=1; $bin<$num_bins+1; $bin++) {
		    if($CONF_bins_up_vect[$j][$cond][$bin] > $max_conf_up_vect[$j] && $CONF_bins_up_vect[$j][$cond][$bin] < 1) {
			$max_conf_up_vect[$j] = $CONF_bins_up_vect[$j][$cond][$bin];
		    }
		    if($CONF_bins_up_vect[$j][$cond][$bin] < $running) {
			$CONF_bins_up_vect[$j][$cond][$bin] = $running;
		    }
		    else {
			$running = $CONF_bins_up_vect[$j][$cond][$bin];
		    }
		}

		$running = $CONF_bins_down_vect[$j][$cond][0];
		$max_conf_down_vect[$j] = 0;
		for(my $bin=1; $bin<$num_bins+1; $bin++) {
		    if($CONF_bins_down_vect[$j][$cond][$bin] > $max_conf_down_vect[$j] && $CONF_bins_down_vect[$j][$cond][$bin] < 1) {
			$max_conf_down_vect[$j] = $CONF_bins_down_vect[$j][$cond][$bin];
		}
		    if($CONF_bins_down_vect[$j][$cond][$bin] < $running) {
			$CONF_bins_down_vect[$j][$cond][$bin] = $running;
		    }
		    else {
			$running = $CONF_bins_down_vect[$j][$cond][$bin];
		    }
		}
	    }
# DEBUG
#	for(my $j=0; $j<$num_range_values; $j++) {
#	    for(my $bin=0; $bin<$num_bins+1; $bin++) {
#		print "CONF_bins_up_vect[$j][$cond][$bin] = $CONF_bins_up_vect[$j][$cond][$bin]\n";
#	    }
#	}
#	for(my $j=0; $j<$num_range_values; $j++) {
#	    for(my $bin=0; $bin<$num_bins+1; $bin++) {
#		print "CONF_bins_down_vect[$j][$cond][$bin] = $CONF_bins_down_vect[$j][$cond][$bin]\n";
#	    }
#	}
# DEBUG

	}
	else {
	    my $running = $CONF_bins_up[$cond][0];
	    my $max_conf_up = 0;
	    for(my $bin=1; $bin<$num_bins+1; $bin++) {
		if($CONF_bins_up[$cond][$bin] > $max_conf_up && $CONF_bins_up[$cond][$bin] < 1) {
		    $max_conf_up = $CONF_bins_up[$cond][$bin];
		}
		if($CONF_bins_up[$cond][$bin] < $running) {
		    $CONF_bins_up[$cond][$bin] = $running;
		}
		else {
		    $running = $CONF_bins_up[$cond][$bin];
		}
	    }

	    my $running = $CONF_bins_down[$cond][0];
	    my $max_conf_down = 0;
	    for(my $bin=1; $bin<$num_bins+1; $bin++) {
		if($CONF_bins_down[$cond][$bin] > $max_conf_down && $CONF_bins_down[$cond][$bin] < 1) {
		    $max_conf_down = $CONF_bins_down[$cond][$bin];
		}
		if($CONF_bins_down[$cond][$bin] < $running) {
		    $CONF_bins_down[$cond][$bin] = $running;
		}
		else {
		    $running = $CONF_bins_down[$cond][$bin];
		}
	    }
	}
    }
    open my $conf_bins_up_down_fh, '>', 'conf_bins_up_down.py';
    print $conf_bins_up_down_fh "import numpy as np\n";
    print $conf_bins_up_down_fh declare_and_assign('conf_up', \@CONF_bins_up_vect);
    print $conf_bins_up_down_fh declare_and_assign('conf_down', \@CONF_bins_down_vect);
    close $conf_bins_up_down_fh;

    print "Here I am: stat is $stat, tuning param is $tstat_tuning_param!!!\n";
    open my $null_up_down, '>', 'null_up_down.py';
    print $null_up_down declare_and_assign('num_null_up', \@num_null_up_vect);


    if($stat == 0 && !($tstat_tuning_param =~ /\S/)) {
	for(my $cond=$start; $cond<$num_conds; $cond++) {
	    for(my $j=0; $j<$num_range_values; $j++) {
		for(my $id=0; $id<$num_ids; $id++) {
		    if($unpermuted_stat_vect[$j][$id][$cond]>=$center) {
			my $bin;
			$bin=int($num_bins*($unpermuted_stat_vect[$j][$id][$cond]-$center)/($max_vect[$j][$cond]-$center));
			$gene_confidences_up_vect[$j]{$ids[$id]}[$cond]=int(1000 * $CONF_bins_up_vect[$j][$cond][$bin])/1000;
			$gene_confidences_down_vect[$j]{$ids[$id]}[$cond]=0;
		    }
		    else {
			my $bin;
			if($stat == 0) {
			    $bin=int($num_bins*(-$unpermuted_stat_vect[$j][$id][$cond]-$center)/(-$min_vect[$j][$cond]-$center));
			}
			else {
			    $bin=int($num_bins*(1/$unpermuted_stat_vect[$j][$id][$cond]-$center)/(1/$min_vect[$j][$cond]-$center));
			}
			$gene_confidences_down_vect[$j]{$ids[$id]}[$cond]=int(1000 * $CONF_bins_down_vect[$j][$cond][$bin])/1000;
			$gene_confidences_up_vect[$j]{$ids[$id]}[$cond]=0;
		    }
		}
	    }
	}

	my $fewio = $alpha_default[1];

	for(my $i=0; $i<$num_range_values;$i++) {
	    for(my $cond=$start; $cond<$num_conds; $cond++) {
		my $sdfjjweo = $alpha_default[$cond] * $tuning_param_range_values[$i];
		for(my $j=0; $j<10; $j++) {
		    $num_up_by_conf_vect[$i][$cond][$j]=0;
		}
		foreach my $id (sort {$gene_confidences_up_vect[$i]{$b}[$cond]<=>$gene_confidences_up_vect[$i]{$a}[$cond]} keys %{$gene_confidences_up_vect[$i]}) {
		    my $xx = $gene_confidences_up_vect[$i]{$id}[$cond];
		    for(my $j=0; $j<10; $j++) {
			my $c;
			$c = $j * .05 + .5;
			if($xx >= $c) {
			    $num_up_by_conf_vect[$i][$cond][$j]++;
			}
		    }
		}
		for(my $j=0; $j<10; $j++) {
		    $num_down_by_conf_vect[$i][$cond][$j]=0;
		}
		foreach my $id (sort {$gene_confidences_down_vect[$i]{$b}[$cond]<=>$gene_confidences_down_vect[$i]{$a}[$cond]} keys %{$gene_confidences_down_vect[$i]}) {
		    my $xx = $gene_confidences_down_vect[$i]{$id}[$cond];
		    for(my $j=0; $j<10; $j++) {
			my $c;
			$c = $j * .05 + .5;
			if($xx >= $c) {
			    $num_down_by_conf_vect[$i][$cond][$j]++;
			}
		    }
		}
	    }
	}
	for(my $cond=$start; $cond<$num_conds; $cond++) {
	    for(my $j=0; $j<10; $j++) {
		$max_number_up[$cond][$j] = -1;
		for(my $i=0; $i<$num_range_values;$i++) {
		    if($num_up_by_conf_vect[$i][$cond][$j] > $max_number_up[$cond][$j]) {
			$max_number_up[$cond][$j] = $num_up_by_conf_vect[$i][$cond][$j];
			$max_up_index[$cond][$j] = $i;
		    }
		}
	    }
	}
	for(my $cond=$start; $cond<$num_conds; $cond++) {
	    for(my $j=0; $j<10; $j++) {
		$max_number_down[$cond][$j] = -1;
		for(my $i=0; $i<$num_range_values;$i++) {
		    if($num_down_by_conf_vect[$i][$cond][$j] > $max_number_down[$cond][$j]) {
			$max_number_down[$cond][$j] = $num_down_by_conf_vect[$i][$cond][$j];
			$max_down_index[$cond][$j] = $i;
		    }
		}
	    }
	}
#for(my $j=0; $j<10; $j++) {
#	print STDERR "max_up_index = $max_up_index[1][$j]\n";
#}
#for(my $j=0; $j<10; $j++) {
#	print STDERR "max_down_index = $max_down_index[1][$j]\n";
#}

    }
    else {
	for(my $cond=$start; $cond<$num_conds; $cond++) {
	    for(my $id=0; $id<$num_ids; $id++) {
		if($unpermuted_stat[$id][$cond]>=$center) {
		    my $bin;
		    $bin=int($num_bins*($unpermuted_stat[$id][$cond]-$center)/($max[$cond]-$center));
		    $gene_confidences_up{$ids[$id]}[$cond]=int(1000 * $CONF_bins_up[$cond][$bin])/1000;
		    $gene_confidences_down{$ids[$id]}[$cond]=0;
		}
		else {
		    my $bin;
		    if($stat == 0) {
			$bin=int($num_bins*(-$unpermuted_stat[$id][$cond]-$center)/(-$min[$cond]-$center));
		    }
		    else {
			$bin=int($num_bins*(1/$unpermuted_stat[$id][$cond]-$center)/(1/$min[$cond]-$center));
		    }
		    $gene_confidences_down{$ids[$id]}[$cond]=int(1000 * $CONF_bins_down[$cond][$bin])/1000;
		    $gene_confidences_up{$ids[$id]}[$cond]=0;
		}
	    }
	}
    }
    if($stat == 0 && !($tstat_tuning_param =~ /\S/)) {
	$breakdown = "";
	for(my $i=$start; $i<$num_conds; $i++) {
	    $breakdown = $breakdown . "------------------------\n";
	    $breakdown = $breakdown . "condition $i\n";
	    $breakdown = $breakdown . "conf.\tnum.up\tnum.down\n";
	    $breakdown = $breakdown . "------------------------\n";
	    for(my $j=0; $j<10; $j++) {
		my $c;
		$c = $j * .05 + .5;
		$breakdown = $breakdown . "$c\t$num_up_by_conf_vect[$max_up_index[$i][$j]][$i][$j]\t$num_down_by_conf_vect[$max_down_index[$i][$j]][$i][$j]\n";
	    }
	}
	print "\n$breakdown";
    }
    else {
	my @num_up_by_conf;
	my @num_down_by_conf;
	for(my $i=$start; $i<$num_conds; $i++) {
	    for(my $j=0; $j<10; $j++) {
		$num_up_by_conf[$i][$j]=0;
	    }
	    foreach my $id (sort {$gene_confidences_up{$b}[$i]<=>$gene_confidences_up{$a}[$i]} keys %gene_confidences_up) {
		my $xx = $gene_confidences_up{$id}[$i];
		for(my $j=0; $j<10; $j++) {
		    my $c;
		    $c = $j * .05 + .5;
		    if($xx >= $c) {
			$num_up_by_conf[$i][$j]++;
		    }
		}
	    }
	    for(my $j=0; $j<10; $j++) {
		$num_down_by_conf[$i][$j]=0;
	    }
	    foreach my $id (sort {$gene_confidences_down{$b}[$i]<=>$gene_confidences_down{$a}[$i]} keys %gene_confidences_down) {
		my $xx = $gene_confidences_down{$id}[$i];
		for(my $j=0; $j<10; $j++) {
		    my $c;
		    $c = $j * .05 + .5;
		    if($xx >= $c) {
			$num_down_by_conf[$i][$j]++;
		    }
		}
	    }
	}
	$breakdown = "";
	for(my $i=$start; $i<$num_conds; $i++) {
	    $breakdown = $breakdown . "------------------------\n";
	    $breakdown = $breakdown . "condition $i\n";
	    $breakdown = $breakdown . "conf.\tnum.up\tnum.down\n";
	    $breakdown = $breakdown . "------------------------\n";
	    for(my $j=0; $j<10; $j++) {
		my $c;
		$c = $j * .05 + .5;
		$breakdown = $breakdown . "$c\t$num_up_by_conf[$i][$j]\t$num_down_by_conf[$i][$j]\n";
	    }
	}
	print "\n$breakdown";
    }

# if level confidence originally requested to be set "later"

    if($level_confidence =~ /(l|L)/) {
	if($num_conds>1+$start) {
	    print "\n\nNOTE: The above summary is to help you choose the level confidence(s).\n";
	    print "NOTE: You can use any number(s) between 0 and 1 as the level confidence(s).\n";
	}
	else {
	    print "\n\nNOTE: The above summary is to help you choose the level confidence.\n";
	    print "NOTE: You can use any number between 0 and 1 as the level confidence.\n";
	}
	if($num_conds>1+$start) {
	    print "\nPlease enter the level confidence (a number between 0 and 1)\n(or enter S to specify a separate confidence for each group): ";
	}
	else {
	   print "\nPlease enter the level confidence (a number between 0 and 1): ";
	}
	$level_confidence = <STDIN>;
	chomp($level_confidence);
	while((!(0 < $level_confidence && $level_confidence < 1) || !($level_confidence =~ /^0?\.\d*$/)) && ((($level_confidence ne "S" && $level_confidence ne "s")) || $num_conds == 1+$start)) {
	    if($num_conds>1+$start) {
		print "\n\nThe level confidence must be a number strictly beween 0 and 1.\n(or enter S to specify a separate confidence for each group)\nPlease re-enter it: ";
	    }
	    else {
		print "\n\nThe level confidence must be a number strictly beween 0 and 1\nPlease re-enter it: ";
	    }
	    $level_confidence = <STDIN>;
	    chomp($level_confidence);
	}
	if($level_confidence eq "S" || $level_confidence eq "s") {
	    for(my $i=$start; $i<$num_conds; $i++) {
		print "\nEnter the level confidence for group $i: ";
		$level_confidence_array[$i] = <STDIN>;
		chomp($level_confidence_array[$i]);
		while(!(0 < $level_confidence_array[$i] && $level_confidence_array[$i] < 1) || !($level_confidence_array[$i] =~ /^0?\.\d*$/)) {
		    print "\nThe level confidence must be a number strictly beween 0 and 1.\nPlease re-enter it: ";
		    $level_confidence_array[$i] = <STDIN>;
		    chomp($level_confidence_array[$i]);
		}
	    }
	}
	else {
	    for(my $cond=$start; $cond<$num_conds; $cond++) {
		$level_confidence_array[$cond] = $level_confidence;
	    }
	}
    }
# DEBUG
#    for(my $i=$start; $i<$num_conds; $i++) {
#	print "level_confidence_array[$i] = $level_confidence_array[$i]\n";
#    }
# DEBUG

    if($stat == 0 && !($tstat_tuning_param =~ /\S/)) {
	my @num_up_by_conf_vect;
	my @num_down_by_conf_vect;
	my @max_number_up;
        my @max_number_down;
	my @max_index;
        for(my $cond=$start; $cond<$num_conds; $cond++) {
            for(my $i=0; $i<$num_range_values;$i++) {
                $num_up_by_conf_vect[$i][$cond]=0;
                foreach my $id (keys %{$gene_confidences_up_vect[$i]}) {
                    my $xx = $gene_confidences_up_vect[$i]{$id}[$cond];
                    if($xx >= $level_confidence_array[$cond]) {
                        $num_up_by_conf_vect[$i][$cond]++;
                    }
                }
                $num_down_by_conf_vect[$i][$cond]=0;
                foreach my $id (keys %{$gene_confidences_down_vect[$i]}) {
                    my $xx = $gene_confidences_down_vect[$i]{$id}[$cond];
                    if($xx >= $level_confidence_array[$cond]) {
                        $num_down_by_conf_vect[$i][$cond]++;
                    }
                }
            }
        }
        for(my $cond=$start; $cond<$num_conds; $cond++) {
	    $max_number_up[$cond] = -1;
	    for(my $i=0; $i<$num_range_values;$i++) {
		if($num_up_by_conf_vect[$i][$cond] > $max_number_up[$cond]) {
		    $max_number_up[$cond] = $num_up_by_conf_vect[$i][$cond];
		    $max_up_index[$cond] = $i;
		}
	    }
	    $max_number_down[$cond] = -1;
	    for(my $i=0; $i<$num_range_values;$i++) {
		if($num_down_by_conf_vect[$i][$cond] > $max_number_down[$cond]) {
		    $max_number_down[$cond] = $num_down_by_conf_vect[$i][$cond];
		    $max_down_index[$cond] = $i;
		}
	    }

	    @{$CONF_bins_up[$cond]} = @{$CONF_bins_up_vect[$max_up_index[$cond]][$cond]};
	    @{$CONF_bins_down[$cond]} = @{$CONF_bins_down_vect[$max_down_index[$cond]][$cond]};
	    $max[$cond] = $max_vect[$max_up_index[$cond]][$cond];
	    $min[$cond] = $min_vect[$max_down_index[$cond]][$cond];
	    for(my $id=0; $id<$num_ids; $id++) {
		if($unpermuted_stat_vect[$max_up_index[$cond]][$id][$cond] >= 0 ) {
			$unpermuted_stat[$id][$cond] = $unpermuted_stat_vect[$max_up_index[$cond]][$id][$cond];
		}
		else {
			$unpermuted_stat[$id][$cond] = $unpermuted_stat_vect[$max_down_index[$cond]][$id][$cond];
		}
	    }
	    for(my $bin=0; $bin<$num_bins; $bin++) {
		$num_unpermuted_up[$cond][$bin] = $num_unpermuted_up_vect[$max_up_index[$cond]][$cond][$bin];
		$num_unpermuted_down[$cond][$bin] = $num_unpermuted_down_vect[$max_down_index[$cond]][$cond][$bin];
	    }
	    $alpha_up[$cond] = $tuning_param_range_values[$max_up_index[$cond]] * $alpha_default[$cond];
	    $alpha_down[$cond] = $tuning_param_range_values[$max_down_index[$cond]] * $alpha_default[$cond];
	}
    }

# DEBUG
#	for(my $cond=0; $cond<$num_conds; $cond++) {
#	    for(my $bin=0; $bin<$num_bins+1; $bin++) {
#		print "CONF_bins_up[$cond][$bin] = $CONF_bins_up[$cond][$bin]\n";
#	    }
#	}
#	for(my $j=0; $j<$num_range_values; $j++) {
#	    for(my $bin=0; $bin<$num_bins+1; $bin++) {
#		print "CONF_bins_down_vect[$j][$cond][$bin] = $CONF_bins_down_vect[$j][$cond][$bin]\n";
#	    }
#	}
# DEBUG


    for(my $cond=$start; $cond<$num_conds; $cond++) {
        my $flag=0;
	$num_up[$cond]=0;
	my $index_cutoff_up=$num_bins+1;
	my $index_cutoff_down=$num_bins+1;
	for(my $i=0; $i<$num_bins+1;$i++) {
	    if(($CONF_bins_up[$cond][$i]>=$level_confidence_array[$cond]) && ($flag==0)) {
		$index_cutoff_up=$i;
		$flag=1;
	    }
	    if($flag==1) {
		$num_up[$cond]=$num_up[$cond]+$num_unpermuted_up[$cond][$i];
		$flag=2;
	    }
	}
	$flag=0;
	$num_down[$cond]=0;
	for(my $i=0; $i<$num_bins+1;$i++) {
	    if(($CONF_bins_down[$cond][$i]>=$level_confidence_array[$cond]) && ($flag==0)) {
		$index_cutoff_down=$i;
		$flag=1;
	    }
	    if($flag==1) {
		$num_down[$cond]=$num_down[$cond]+$num_unpermuted_down[$cond][$i];
		$flag=2;
	    }
	}
	$cutratios_up[$cond]=$index_cutoff_up*($max[$cond]-$center)/$num_bins+$center;
	if($stat == 0) {
	    $cutratios_down[$cond]=-($index_cutoff_down*(-$min[$cond]-$center)/$num_bins+$center);
	}
	else {
	    $cutratios_down[$cond]=1/($index_cutoff_down*(1/$min[$cond]-$center)/$num_bins+$center);
	}
	for(my $id=0; $id<$num_ids; $id++) {
	    if($unpermuted_stat[$id][$cond]>=$center) {
		my $bin;
		$bin=int($num_bins*($unpermuted_stat[$id][$cond]-$center)/($max[$cond]-$center));
		if($CONF_bins_up[$cond][$bin]>=$level_confidence_array[$cond]) {
		    $confidence_levels_hash{$ids[$id]}[$cond]=int(1000 * $CONF_bins_up[$cond][$bin])/1000;
		}
		$gene_confidences_up{$ids[$id]}[$cond]=int(1000 * $CONF_bins_up[$cond][$bin])/1000;
		$gene_confidences_down{$ids[$id]}[$cond]=0;
	    }
	    else {
		my $bin;
		if($stat == 0) {
		    $bin=int($num_bins*(-$unpermuted_stat[$id][$cond]-$center)/(-$min[$cond]-$center));
		}
		else {
		    $bin=int($num_bins*(1/$unpermuted_stat[$id][$cond]-$center)/(1/$min[$cond]-$center));
		}
		if($CONF_bins_down[$cond][$bin]>=$level_confidence_array[$cond]) {
		    $confidence_levels_hash{$ids[$id]}[$cond]=int(1000 * $CONF_bins_down[$cond][$bin]) / 1000;
		}
		$gene_confidences_down{$ids[$id]}[$cond]=int(1000 * $CONF_bins_down[$cond][$bin])/1000;
		$gene_confidences_up{$ids[$id]}[$cond]=0;
	    }
	}
    }

    return (\%confidence_levels_hash, \@num_down, \@num_up, \@cutratios_down, \@cutratios_up, \@max, \@min, \@unpermuted_stat, \%gene_confidences_up, \%gene_confidences_down, \@level_confidence_array, \@alpha_up, \@alpha_down, $breakdown);
}

sub AdjustNumDiff {
    my ($V, $R, $num_ids) = @_;
    my @V;
    $V[0] = $V;
    for(my $i=1; $i<6; $i++) {
	$V[$i] = $V[0]-$V[0]/$num_ids*($R-$V[$i-1]);
    }
    return $V[5];

}

sub FindDistUnpermutedStat {

# This finds the distribution into bins of the statistic to be used, over the
# unpermuted data.

    my ($data_ref, $stat, $num_conds, $missing_value_designator, $max_ref, $min_ref, $min_presence_array_ref, $design, $alpha_ref, $data_is_logged, $use_logged_data, $paired) = @_;
    my @data = @{$data_ref};
    my @min_presence_array = @{$min_presence_array_ref};
    my $num_ids = @data;
    my @max = @{$max_ref};
    my @min= @{$min_ref};
    my @dist_up;
    my @dist_down;
    my @alpha = @{$alpha_ref};
    my $value;
    my $center;
    my $start;
    my @unpermuted_stat;
    if($stat == 0) {
	$center = 0;
    }
    else {
	$center = 1;
    }
    if(!($design eq "D")) {
	$start = 1;
    }
    else {
	$start = 0;
    }

    for(my $cond=$start; $cond<$num_conds; $cond++) {
	for(my $id=0; $id<$num_ids; $id++) {
	    if(!($design eq "D")) {
		my @vector1 = @{$data[$id][0]};
		my @vector2 = @{$data[$id][$cond]};
		if($stat == 0) {
		    if($paired == 0) {
			$value = ComputeTstat(\@vector2, \@vector1, $missing_value_designator, $min_presence_array[$cond], $min_presence_array[0], $alpha[$cond]);
		    }
		    else {
			$value = ComputePairedTstat(\@vector2, \@vector1, $missing_value_designator, $min_presence_array[$cond], $min_presence_array[0], $alpha[$cond], $data_is_logged, $use_logged_data);
		    }
		}
		if($stat == 1) {
		    if($paired == 0) {
			$value = ComputeMean(\@vector2, $missing_value_designator, $min_presence_array[$cond])/ComputeMean(\@vector1, $missing_value_designator, $min_presence_array[0]);
		    }
		    else {
			$value = ComputePairedMean(\@vector2, \@vector1, $missing_value_designator, $min_presence_array[$cond], $min_presence_array[0]);
		    }
		}
		if($stat == 2) {
		    $value = ComputeMedian(\@vector2, $missing_value_designator, $min_presence_array[$cond])/ComputeMedian(\@vector1, $missing_value_designator, $min_presence_array[0]);
		}
		if($value ne "NA") {
		    if($value>=$center) {
			my $bin=int($num_bins * ($value - $center) / ($max[$cond] - $center));
			$dist_up[$cond][$bin]++;
			$dist_down[$cond][0]++;
		    }
		    if($value<=$center) {
			my $bin;
			if($stat == 0) {
			    $bin=int($num_bins * (-$value - $center) / (-$min[$cond] - $center));
			}
			else {
			    $bin=int($num_bins * (1/$value - $center) / (1/$min[$cond] - $center));
			}
			$dist_down[$cond][$bin]++;
			$dist_up[$cond][0]++;
		    }
		}
	    }
	    else {
		my @vector = @{$data[$id][$cond]};
		if($stat == 0) {
		    $value = ComputeOneSampleTstat(\@vector, $missing_value_designator, $min_presence_array[$cond], $alpha[$cond], $use_logged_data);
		}
		else {
		    $value = ComputeGeometricMean(\@vector, $missing_value_designator, $min_presence_array[$cond]);
		}
		if($value ne "NA") {
		    if($value>=$center) {
			my $bin=int($num_bins * ($value - $center) / ($max[$cond] - $center));
			$dist_up[$cond][$bin]++;
			$dist_down[$cond][0]++;
		    }
		    if($value<=$center) {
			my $bin;
			if($stat == 0) {
			    $bin=int($num_bins * (-$value - $center) / (-$min[$cond] - $center));
			}
			else {
			    $bin=int($num_bins * (1/$value - $center) / (1/$min[$cond] - $center));
			}
			$dist_down[$cond][$bin]++;
			$dist_up[$cond][0]++;
		    }
		}
	    }
	    $unpermuted_stat[$id][$cond] = $value;
	}
    }
    return (\@dist_up, \@dist_down, \@unpermuted_stat);
}

sub FindDistUnpermutedStatVect {

    my @args = @_;

# This finds the distribution into bins of the statistic to be used, over the
# unpermuted data.

    my ($data_ref, $stat, $num_conds, $missing_value_designator, $max_vect_ref, $min_vect_ref, $min_presence_array_ref, $design, $alpha_ref, $data_is_logged, $use_logged_data, $alpha_default_ref, $paired) = @args;

    open my $out_fh, '>', 'unpermuted_stats.py';

    print $out_fh "import numpy as np\n";
    print $out_fh declaration('data', [scalar @{ $data_ref }, 16 ]);

    my @pydata ;

    for my $i (0 .. @{ $data_ref } - 1) {
        my @row;
        for my $j (0 .. 3) {
            for my $k (1 .. 4) {
                push @row, $data_ref->[$i][$j][$k];
            }
        }
        push @pydata, \@row;
    }
    
    print $out_fh assign_array('data', \@pydata);

    print $out_fh declaration('maxes', shape($max_vect_ref));
    print $out_fh assign_array('maxes', $max_vect_ref);

    print $out_fh declaration('mins', shape($min_vect_ref));
    print $out_fh assign_array('mins', $min_vect_ref);

    print $out_fh np_array('alphas', $alpha_ref);
    print $out_fh np_array('alpha_default', $alpha_default_ref);

    my @alpha_default = @{$alpha_default_ref};
    my @data = @{$data_ref};
    my @min_presence_array = @{$min_presence_array_ref};
    my $num_ids = @data;
    my @max;
    my @min;
    my @max_vect = @{$max_vect_ref};
    my @min_vect= @{$min_vect_ref};
    my @dist_up;
    my @dist_down;
    my @dist_up_vect;
    my @dist_down_vect;
    my @alpha = @{$alpha_ref};
    my $value;
    my @value;
    my $value_ref;
    my $center;
    my $start;
    my @unpermuted_stat;
    my @unpermuted_stat_vect;
    my $num_range_values = 10;

    if($stat == 0) {
	$center = 0;
    }
    else {
	$center = 1;
    }
    if(!($design eq "D")) {
	$start = 1;
    }
    else {
	$start = 0;
    }

    for(my $cond=$start; $cond<$num_conds; $cond++) {
	for(my $id=0; $id<$num_ids; $id++) {
	    if(!($design eq "D")) {
		my @vector1 = @{$data[$id][0]};
		my @vector2 = @{$data[$id][$cond]};
		if($stat == 0) {
		    if($paired == 0) {
			$value_ref = ComputeTstatVector(\@vector2, \@vector1, $missing_value_designator, $min_presence_array[$cond], $min_presence_array[0], $alpha_default[$cond]);
			@value = @{$value_ref};
		    }
		    else {
			$value_ref = ComputePairedTstatVector(\@vector2, \@vector1, $missing_value_designator, $min_presence_array[$cond], $min_presence_array[0], $alpha_default[$cond], $data_is_logged, $use_logged_data);
			@value = @{$value_ref};
		    }
		}
		for(my $i=0; $i<$num_range_values; $i++) {
		    $value = $value[$i];
		    $unpermuted_stat_vect[$i][$id][$cond] = $value;
		    if($value ne "NA") {
			if($value>=$center) {
			    my $bin=int($num_bins * ($value - $center) / ($max_vect[$i][$cond] - $center));
                            if ($i == 0 && $cond == 1 && $bin == 1) {
                                my $boundary = $max_vect[$i][$cond] / $num_bins;
                                
                                print "up[$i][$cond][$bin] for $value under $boundary, num bins = $num_bins, center = $center, max is $max_vect[$i][$cond]\n";
                            }
			    $dist_up_vect[$i][$cond][$bin]++;
			    $dist_down_vect[$i][$cond][0]++;
			}
			if($value<=$center) {
			    my $bin;
			    $bin=int($num_bins * (-$value - $center) / (-$min_vect[$i][$cond] - $center));
			    $dist_down_vect[$i][$cond][$bin]++;
			    $dist_up_vect[$i][$cond][0]++;
			}
		    }
		}
	    }
	    else {
		my @vector = @{$data[$id][$cond]};
		$value_ref = ComputeOneSampleTstatVector(\@vector, $missing_value_designator, $min_presence_array[$cond], $alpha_default[$cond], $use_logged_data);
		@value = @{$value_ref};
		for(my $i=0; $i<$num_range_values; $i++) {
		    $value = $value[$i];
		    $unpermuted_stat_vect[$i][$id][$cond] = $value;
		    if($value ne "NA") {
			if($value>=$center) {
			    my $bin=int($num_bins * ($value - $center) / ($max_vect[$i][$cond] - $center));
			    $dist_up_vect[$i][$cond][$bin]++;
			    $dist_down_vect[$i][$cond][0]++;
			}
			if($value<=$center) {
			    my $bin;
			    $bin=int($num_bins * (-$value - $center) / (-$min_vect[$i][$cond] - $center));
			    $dist_down_vect[$i][$cond][$bin]++;
			    $dist_up_vect[$i][$cond][0]++;
			}
		    }
		}
	    }
	}
    }
    
    print $out_fh declare_and_assign('dist_up', \@dist_up_vect);
    print $out_fh declare_and_assign('dist_down', \@dist_down_vect);
    print $out_fh declare_and_assign('stats', \@unpermuted_stat_vect);
    return (\@dist_up_vect, \@dist_down_vect, \@unpermuted_stat_vect);
}


sub FindDefaultAlpha {

# This finds the distribution into bins of the statistic to be used, over the
# unpermuted data.

    my ($data_ref, $stat, $num_conds, $missing_value_designator, $min_presence_array_ref, $design, $num_reps_ref, $data_is_logged, $use_logged_data, $paired) = @_;
    my @data = @{$data_ref};
    my @min_presence_array = @{$min_presence_array_ref};
    my @num_reps = @{$num_reps_ref};
    my $num_ids = @data;
    my @alpha;
    my $value;
    my $start;
    my $num=0;
    my $mean=0;
    if(!($design eq "D")) {
	$start = 1;
    }
    else {
	$start = 0;
    }
    for(my $cond=$start; $cond<$num_conds; $cond++) {
	$num=0;
	$mean=0;
	for(my $id=0; $id<$num_ids; $id++) {
	    if(!($design eq "D")) {
		my @vector1 = @{$data[$id][0]};
		my @vector2 = @{$data[$id][$cond]};
		if($paired == 0) {
		    $value = ComputeS(\@vector2, \@vector1, $missing_value_designator, $min_presence_array[$cond], $min_presence_array[0]);
		}
		else {
		    $value = ComputePairedS(\@vector2, \@vector1, $missing_value_designator, $min_presence_array[$cond], $min_presence_array[0], $data_is_logged, $use_logged_data);
		}
		if($value ne "NA") {
		    $mean = $mean + $value;
		    $num++;
		}
	    }
	    else {
		my @vector = @{$data[$id][$cond]};
		$value = ComputeOneSampleS(\@vector, $missing_value_designator, $min_presence_array[$cond]);
		if($value ne "NA") {
		    $mean = $mean + $value;
		    $num++;
		}
	    }
	}
	$mean = $mean / $num;
	my $std_dev = 0;
	$num=0;
	for(my $id=0; $id<$num_ids; $id++) {
	    if(!($design eq "D")) {
		my @vector1 = @{$data[$id][0]};
		my @vector2 = @{$data[$id][$cond]};
		if($paired == 0) {
		    $value = ComputeS(\@vector2, \@vector1, $missing_value_designator, $min_presence_array[$cond], $min_presence_array[0]);
		}
		else {
		    $value = ComputePairedS(\@vector2, \@vector1, $missing_value_designator, $min_presence_array[$cond], $min_presence_array[0], $data_is_logged, $use_logged_data);
		}
		if($value ne "NA" && $value < $mean) {
		    $std_dev = $std_dev + ($value - $mean) * ($value - $mean);
		    $num++;
		}
	    }
	    else {
		my @vector = @{$data[$id][$cond]};
		$value = ComputeOneSampleS(\@vector, $missing_value_designator, $min_presence_array[$cond]);
		if($value ne "NA" && $value < $mean) {
		    $std_dev = $std_dev + ($value - $mean) * ($value - $mean);
		    $num++;
		}
	    }
	}
	$std_dev = sqrt($std_dev / ($num - 1));
#	print "TUNING PARAM: mean = $mean, std_dev = $std_dev\n";
	$alpha[$cond] = $mean * 2/ sqrt($num_reps[$cond]+$num_reps[0]);
#	print "alpha = $alpha[$cond]\n";
    }
    return \@alpha;
}

sub FindMeansUnpermuted {

    my ($data_ref, $num_conds, $missing_value_designator) = @_;
    my @data = @{$data_ref};
    my $num_ids = @data;
    my $value;
    my @unpermuted_means;
    for(my $cond=0; $cond<$num_conds; $cond++) {
	for(my $id=0; $id<$num_ids; $id++) {
	    my @vector2 = @{$data[$id][$cond]};
	    $value = ComputeMean(\@vector2, $missing_value_designator, 1);
	    $unpermuted_means[$id][$cond] = $value;
	}
    }
    return (\@unpermuted_means);
}

sub FindMaxMinValueOfStat {

# This finds the max and min value of the statistic to be used, over the
# unpermuted data.

    my ($data_ref, $stat, $num_conds, $missing_value_designator, $design, $min_presence_array_ref, $alpha_ref, $data_is_logged, $use_logged_data, $paired) = @_;
    my @data = @{$data_ref};
    my $num_ids = @data;
    my @min_presence_array = @{$min_presence_array_ref};
    my @alpha = @{$alpha_ref};
    my $min;
    my $max;
    my $value;
    my @min;
    my @max;
    my $start;
    if(!($design eq "D")) {
	$start = 1;
    }
    else {
	$start = 0;
    }
    for(my $cond=$start; $cond<$num_conds; $cond++) {
	$min = 10000000;
	$max = -10000000;
	if(!($design eq "D")) {
	    for(my $id=0; $id<$num_ids; $id++) {
		my @vector1 = @{$data[$id][0]};
		my @vector2 = @{$data[$id][$cond]};
		if($stat == 0) {
		    if($paired == 0) {
			$value = ComputeTstat(\@vector2, \@vector1, $missing_value_designator, $min_presence_array[$cond], $min_presence_array[0], $alpha[$cond]);
			if($value ne "NA" && $max < $value) {
			    $max = $value;
			}
			if($value ne "NA" && $min > $value) {
			    $min = $value;
			}
		    }
		    else {
			$value = ComputePairedTstat(\@vector2, \@vector1, $missing_value_designator, $min_presence_array[$cond], $min_presence_array[0], $alpha[$cond], $data_is_logged, $use_logged_data);
			if($value ne "NA" && $max < $value) {
			    $max = $value;
			}
			if($value ne "NA" && $min > $value) {
			    $min = $value;
			}
		    }
		}
		if($stat == 1) {
		    if($paired == 0) {
			my $value1 = ComputeMean(\@vector2, $missing_value_designator, $min_presence_array[$cond]);
			$value = ComputeMean(\@vector1, $missing_value_designator, $min_presence_array[0]);
			if($value ne "NA" && $value1 ne "NA") {
			    $value = $value1/$value;
			}
			else {
			    $value = "NA";
			}
			if($value ne "NA" && $max < $value) {
			    $max = $value;
			}
			if($value ne "NA" && $min > $value) {
			    $min = $value;
			}
		    }
		    else {
			$value = ComputePairedMean(\@vector2, \@vector1, $missing_value_designator, $min_presence_array[$cond], $min_presence_array[0]);
			if($value ne "NA" && $max < $value) {
			    $max = $value;
			}
			if($value ne "NA" && $min > $value) {
			    $min = $value;
			}
		    }
		}
		if($stat == 2) {
		    $value = ComputeMedian(\@vector2, $missing_value_designator, $min_presence_array[$cond])/ComputeMean(\@vector1, $missing_value_designator, $min_presence_array[0]);
		    if($value ne "NA" && $max < $value) {
			$max = $value;
		    }
		    if($value ne "NA" && $min > $value) {
			$min = $value;
		    }
		}
	    }
	}
	else {
	    for(my $id=0; $id<$num_ids; $id++) {
		my @vector = @{$data[$id][$cond]};
		if($stat == 0) {
#		    print "$vector[0]\t$vector[1]\t$vector[2]\t$vector[3]\t$vector[4]\t$vector[5]\n";
		    $value = ComputeOneSampleTstat(\@vector, $missing_value_designator, $min_presence_array[$cond], $alpha[$cond], $use_logged_data);
#		    print "value=$value\n";
		}
		else {
		    $value = ComputeGeometricMean(\@vector, $missing_value_designator, $min_presence_array[$cond]);
		}
		if($value ne "NA" && $max < $value) {
		    $max = $value;
		}
		if($value ne "NA" && $min > $value) {
		    $min = $value;
		}
	    }
	}

	if($stat == 0) {
	    if($min < 0) {
		$min[$cond]=$min;
	    }
	    else {
		$min[$cond]=0;
	    }
	    if($max > 0) {
		$max[$cond]=$max;
	    }
	    else {
		$max[$cond]=0;
	    }
	}
	else {
	    if($min < 1) {
		$min[$cond]=$min;
	    }
	    else {
		$min[$cond]=1;
	    }
	    if($max > 1) {
		$max[$cond]=$max;
	    }
	    else {
		$max[$cond]=1;
	    }
	}

# this bit of code was intended to avoid the case where the min or max were too close to zero,
# or to 1 in the case of ratios.  But it caused problems with generating the patterns, rather
# than fix that, am commenting out this, but might need to do something if indeed, the min or
# max is 0...
#	if($stat == 0) {
#	    if($min < -1) {
#		$min[$cond]=$min;
#	    }
#	    else {
#		$min[$cond]=-1;
#	    }
#	    if($max > 1) {
#		$max[$cond]=$max;
#	    }
#	    else {
#		$max[$cond]=1;
#	    }
#	}
#	else {
#	    if($min < 1/2) {
#		$min[$cond]=$min;
#	    }
#	    else {
#		$min[$cond]=1/2;
#	    }
#	    if($max > 2) {
#		$max[$cond]=$max;
#	    }
#	    else {
#		$max[$cond]=2;
#	    }
#	}
    }
    return (\@min, \@max);
}
sub FindMaxMinValueOfStatVect {

# This finds the max and min value of the statistic to be used, over the
# unpermuted data.

    my ($data_ref, $stat, $num_conds, $missing_value_designator, $design, $min_presence_array_ref, $alpha_ref, $data_is_logged, $use_logged_data, $alpha_default_ref, $paired) = @_;
    my @data = @{$data_ref};
    my $num_ids = @data;
    my @min_presence_array = @{$min_presence_array_ref};
    my @alpha = @{$alpha_ref};
    my $min;
    my $max;
    my $value;
    my @value;
    my $value_ref;
    my @min;
    my @max;
    my @min_vect;
    my @max_vect;
    my $start;
    my @alpha_default = @{$alpha_default_ref};
    my $num_range_values = 10;

    if(!($design eq "D")) {
	$start = 1;
    }
    else {
	$start = 0;
    }
    for(my $cond=$start; $cond<$num_conds; $cond++) {
	for(my $i=0; $i<$num_range_values; $i++) {
	    $min[$i] = 10000000;
	    $max[$i] = -10000000;
	}
	if(!($design eq "D")) {
	    for(my $id=0; $id<$num_ids; $id++) {
		my @vector1 = @{$data[$id][0]};
		my @vector2 = @{$data[$id][$cond]};
		if($stat == 0) {
		    if($paired == 0) {
			$value_ref = ComputeTstatVector(\@vector2, \@vector1, $missing_value_designator, $min_presence_array[$cond], $min_presence_array[0], $alpha_default[$cond]);
			@value = @{$value_ref};
			for(my $i=0; $i<$num_range_values; $i++) {
			    $value = @value[$i];
			    if($value ne "NA" && $max[$i] < $value) {
				$max[$i] = $value;
			    }
			    if($value ne "NA" && $min[$i] > $value) {
				$min[$i] = $value;
			    }
			}
		    }
		    else {
			$value_ref = ComputePairedTstatVector(\@vector2, \@vector1, $missing_value_designator, $min_presence_array[$cond], $min_presence_array[0], $alpha_default[$cond], $data_is_logged, $use_logged_data);
			@value = @{$value_ref};
			for(my $i=0; $i<$num_range_values; $i++) {
			    $value = @value[$i];
			    if($value ne "NA" && $max[$i] < $value) {
				$max[$i] = $value;
			    }
			    if($value ne "NA" && $min[$i] > $value) {
				$min[$i] = $value[$i];
			    }
			}
		    }
		}
	    }
	}
	else {
	    for(my $id=0; $id<$num_ids; $id++) {
		my @vector = @{$data[$id][$cond]};
		$value_ref = ComputeOneSampleTstatVector(\@vector, $missing_value_designator, $min_presence_array[$cond], $alpha_default[$cond], $use_logged_data);
		@value = @{$value_ref};
		for(my $i=0; $i<$num_range_values; $i++) {
		    $value = @value[$i];
		    if($value ne "NA" && $max[$i] < $value) {
			$max[$i] = $value;
		    }
		    if($value ne "NA" && $min[$i] > $value) {
			$min[$i] = $value;
		    }
		}
	    }
	}

	for(my $i=0; $i<$num_range_values; $i++) {
	    if($min[$i] < 0) {
		$min_vect[$i][$cond]=$min[$i];
	    }
	    else {
		$min_vect[$i][$cond]=0;
	    }
	    if($max[$i] > 0) {
		$max_vect[$i][$cond]=$max[$i];
	    }
	    else {
		$max_vect[$i][$cond]=0;
	    }
	}
    }
# DEBUG
#    for(my $cond=$start; $cond<$num_conds; $cond++) {
#	for(my $i=0; $i<$num_range_values; $i++) {
#	    print "max_vect[$i][$cond]=$max_vect[$i][$cond]\n";
#	}
#    }
# DEBUG

    return (\@min_vect, \@max_vect);
}

sub ComputeMedian {
    my ($vector_ref, $missing_value_designator, $min_presence) = @_;
    my @vector = @{$vector_ref};

    my @t;
    my @s;
    my $len = @vector;
    my $ind=0;
    my $m;
    for(my $i=0; $i<$len; $i++) {
        if(defined $vector[$i] && $vector[$i] ne $missing_value_designator) {
            $t[$ind] = $vector[$i];
            $ind++;
        }
    }
    if($ind>$min_presence-1) {
	my $c = @t;

	@s = sort {$a <=> $b} @t;
	$c = @s;

	if(($c % 2) == 1) {
	    $m = $s[($c-1)/2];
	}
	else {
	    $m = ($s[($c-2)/2]+$s[($c)/2])/2;
	}
    }
    else {
	$m = "NA";
    }

    return $m;
}

sub ComputeMean {
    my ($vector_ref, $missing_value_designator, $min_presence) = @_;
    my @vector = @{$vector_ref};
    my $len = @vector;
    my $m=0;
    my $n=0;
    for(my $i=0; $i<$len; $i++) {
        if(defined $vector[$i] && $vector[$i] ne $missing_value_designator) {
            $m=$m+$vector[$i];
            $n++;
        }
    }
    if($n>$min_presence-1) {
	$m=$m/$n;
    }
    else {
	$m = "NA";
    }

    return $m;
}

sub ComputeTstat {
    my ($vector1_ref, $vector2_ref, $missing_value_designator, $min_presence1, $min_presence2, $tstat_tuning_param) = @_;
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
	$mean1 = $m/$length_values1;
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
	$t = "NA";
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
	$t = ($mean1 - $mean2)*sqrt($length_values1*$length_values2)/(($tstat_tuning_param+$S)*sqrt($length_values1+$length_values2));
    }
    return $t;
}

sub ComputeTstatVector {
    my ($vector1_ref, $vector2_ref, $missing_value_designator, $min_presence1, $min_presence2, $tstat_tuning_param_default) = @_;
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
    my @t;
    my $sd1;
    my $sd2;
    my $min_flag=0;
    $j=0;
    $m=0;

# hard coded in several places
    my @tuning_param_range_values;
    $tuning_param_range_values[0] = .0001;
    $tuning_param_range_values[1] = .01;
    $tuning_param_range_values[2] = .1;
    $tuning_param_range_values[3] = .3;
    $tuning_param_range_values[4] = .5;
    $tuning_param_range_values[5] = 1;
    $tuning_param_range_values[6] = 1.5;
    $tuning_param_range_values[7] = 2;
    $tuning_param_range_values[8] = 3;
    $tuning_param_range_values[9] = 10;
    my $num_range_values = @tuning_param_range_values;

    for($i=0;$i<$length_vector1;$i++) {
#	print "vector1[$i]=$vector1[$i]\n";
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
	$mean1 = $m/$length_values1;
    }
    $j=0;
    $m=0;
    for($i=0;$i<$length_vector2;$i++) {
#	print "vector2[$i]=$vector2[$i]\n";
	if(defined $vector2[$i] && $vector2[$i] ne $missing_value_designator) {
	    $values2[$j] = $vector2[$i];
	    $m=$m+$vector2[$i];
	    $j++;
	}
    }
    $length_values2 = $j;
#    print "length_values1=$length_values1, length_values2=$length_values2\n";
    if($length_values2 < $min_presence2) {
	$min_flag=1;
    }
    else {
	$mean2 = $m/$length_values2;
    }
    if($min_flag == 1) {
	for(my $i=0; $i<$num_range_values; $i++) {
	    $t[$i] = "NA";
	}
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
	for(my $i=0; $i<$num_range_values; $i++) {
	    $t[$i] = ($mean1 - $mean2)*sqrt($length_values1*$length_values2)/(($tuning_param_range_values[$i] * $tstat_tuning_param_default+$S)*sqrt($length_values1+$length_values2));

	}
    }
    return \@t;
}

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
	$mean1 = $m/$length_values1;
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
    return $S;
}

sub ComputeOneSampleTstat {
    my ($vector_ref, $missing_value_designator, $min_presence, $tstat_tuning_param, $use_logged_data) = @_;
    my @vector=@{$vector_ref};
    my $length_vector=@vector;
    my $i;
    my $j;
    my $m;
    my $mean;
    my @values;
    my $length_values;
    my $S;
    my $t;
    my $sd;

    $j=0;
    $m=0;
    for($i=0;$i<$length_vector;$i++) {
        if(defined $vector[$i] && $vector[$i] ne $missing_value_designator) {
            $values[$j] = $vector[$i];
            $m=$m+$vector[$i];
            $j++;
        }
    }
    if($j<$min_presence) {
	$t="NA";
    }
    else {
	$length_values = $j;

	$mean = $m/$length_values;
	$sd = 0;
	for($i=0; $i<$length_values; $i++) {
	    $sd = $sd + ($values[$i]-$mean)**2;
	}
	$sd = sqrt($sd/($length_values-1));
	$t = $mean*sqrt($length_values)/($tstat_tuning_param*sqrt(2) + $sd);
    }
    return $t;
}

sub ComputeOneSampleTstatVector {
    my ($vector_ref, $missing_value_designator, $min_presence, $tstat_tuning_param_default, $use_logged_data) = @_;
    my @vector=@{$vector_ref};
    my $length_vector=@vector;
    my $i;
    my $j;
    my $m;
    my $mean;
    my @values;
    my $length_values;
    my $S;
    my @t;
    my $sd;
# hard coded in several places
    my @tuning_param_range_values;
    $tuning_param_range_values[0] = .0001;
    $tuning_param_range_values[1] = .01;
    $tuning_param_range_values[2] = .1;
    $tuning_param_range_values[3] = .3;
    $tuning_param_range_values[4] = .5;
    $tuning_param_range_values[5] = 1;
    $tuning_param_range_values[6] = 1.5;
    $tuning_param_range_values[7] = 2;
    $tuning_param_range_values[8] = 3;
    $tuning_param_range_values[9] = 10;
    my $num_range_values = @tuning_param_range_values;

    $j=0;
    $m=0;
    for($i=0;$i<$length_vector;$i++) {
        if(defined $vector[$i] && $vector[$i] ne $missing_value_designator) {
            $values[$j] = $vector[$i];
            $m=$m+$vector[$i];
            $j++;
        }
    }
    if($j<$min_presence) {
	for(my $i=0; $i<$num_range_values; $i++) {
	    $t[$i]="NA";
	}
    }
    else {
	$length_values = $j;

	$mean = $m/$length_values;
	$sd = 0;
	for($i=0; $i<$length_values; $i++) {
	    $sd = $sd + ($values[$i]-$mean)**2;
	}
	$sd = sqrt($sd/($length_values-1));
	for(my $i=0; $i<$num_range_values; $i++) {
	    $t[$i] = $mean*sqrt($length_values)/(($tuning_param_range_values[$i] * $tstat_tuning_param_default)*sqrt(2) + $sd);
	}
    }
    return \@t;
}


sub ComputeOneSampleS {
    my ($vector_ref, $missing_value_designator, $min_presence) = @_;
    my @vector=@{$vector_ref};
    my $length_vector=@vector;
    my $i;
    my $j;
    my $m;
    my $mean;
    my @values;
    my $length_values;
    my $S;
    my $t;
    my $sd;

    $j=0;
    $m=0;
    for($i=0;$i<$length_vector;$i++) {
        if(defined $vector[$i] && $vector[$i] ne $missing_value_designator) {
            $values[$j] = $vector[$i];
            $m=$m+$vector[$i];
            $j++;
        }
    }
    if($j<$min_presence) {
	$sd="NA";
    }
    else {
	$length_values = $j;
	$mean = $m/$length_values;
	$sd = 0;
	for($i=0; $i<$length_values; $i++) {
	    $sd = $sd + ($values[$i]-$mean)**2;
	}
	$sd = sqrt($sd/($length_values-1));
    }
    $sd = $sd / sqrt(2);
    return $sd;
}

sub ComputePairedTstat {

# this is going to compute the paired t-test, by taking differences
# and then calling the one-sample t-test routine.  If the two groups
# are ratios to a common reference, then probably should take logs and
# input only the M values

    my ($vector1_ref, $vector2_ref, $missing_value_designator, $min_presence1, $min_presence2, $tstat_tuning_param, $data_is_logged, $use_logged_data) = @_;
    my @vector1 = @{$vector1_ref};
    my @vector2 = @{$vector2_ref};
    my $length_vector1=@vector1;
    my $length_vector2=@vector2;
    my $i;
    my $j;
    my $m;
    my $mean1;
    my $mean2;
    my @values;
    my $length_values;
    my $S;
    my $t;
    my $sd1;
    my $sd2;

    $j=0;
    $m=0;
    for($i=0;$i<$length_vector1;$i++) {
	if((defined $vector1[$i] && defined $vector2[$i]) && ($vector1[$i] ne $missing_value_designator && $vector2[$i] ne $missing_value_designator)) {
	    $values[$j] = $vector1[$i] - $vector2[$i];
	    $j++;
	}
    }
    if($j<$min_presence1 || $j<$min_presence2) {
	$t="NA";
    }
    else {
	my $length = @values;
	$t = ComputeOneSampleTstat(\@values, $missing_value_designator, $length, $tstat_tuning_param, $use_logged_data);
    }
    return $t;
}

sub ComputePairedTstatVector {

# this is going to compute the paired t-stat by taking differences
# and then calling the one-sample t-stat routine.  If the two groups
# are ratios to a common reference, then probably should take logs and
# input only the M values

    my ($vector1_ref, $vector2_ref, $missing_value_designator, $min_presence1, $min_presence2, $tstat_tuning_param_default, $data_is_logged, $use_logged_data) = @_;
    my @vector1 = @{$vector1_ref};
    my @vector2 = @{$vector2_ref};
    my $length_vector1=@vector1;
    my $length_vector2=@vector2;
    my $i;
    my $j;
    my $m;
    my $mean1;
    my $mean2;
    my @values;
    my $length_values;
    my $S;
    my @t;
    my $t_vect_ref;
    my $sd1;
    my $sd2;
    my $num_range_values = 10;

    $j=0;
    $m=0;
    for($i=0;$i<$length_vector1;$i++) {
	if((defined $vector1[$i] && defined $vector2[$i]) && ($vector1[$i] ne $missing_value_designator && $vector2[$i] ne $missing_value_designator)) {
	    $values[$j] = $vector1[$i] - $vector2[$i];
	    $j++;
	}
    }
    if($j<$min_presence1 || $j<$min_presence2) {
	for(my $i=0; $i<$num_range_values; $i++) {
	    $t[$i]="NA";
	}
    }
    else {
	my $length = @values;
	$t_vect_ref = ComputeOneSampleTstatVector(\@values, $missing_value_designator, $length, $tstat_tuning_param_default, $use_logged_data);
	@t = @{$t_vect_ref};
    }
    return \@t;
}

sub ComputePairedS {

# this is going to compute the paired t-stat, by taking differences
# and then calling the one-sample t-stat routine.  If the two groups
# are ratios to a common reference, then probably should take logs and
# input only the M values

    my ($vector1_ref, $vector2_ref, $missing_value_designator, $min_presence1, $min_presence2, $data_is_logged, $use_logged_data) = @_;
    my @vector1 = @{$vector1_ref};
    my @vector2 = @{$vector2_ref};
    my $length_vector1=@vector1;
    my $length_vector2=@vector2;
    my $i;
    my $j;
    my $m;
    my $mean1;
    my $mean2;
    my @values;
    my $length_values;
    my $S;
    my $t;
    my $sd1;
    my $sd2;

    $j=0;
    $m=0;
    for($i=0;$i<$length_vector1;$i++) {
	if((defined $vector1[$i] && defined $vector2[$i]) && ($vector1[$i] ne $missing_value_designator && $vector2[$i] ne $missing_value_designator)) {
	    $values[$j] = $vector1[$i] - $vector2[$i];
	    $j++;
	}
    }
    if($j<$min_presence1 || $j<$min_presence2) {
	$S="NA";
    }
    else {
	my $length = @values;
	$S = ComputeOneSampleS(\@values, $missing_value_designator, $length);
    }
    return $S;
}


sub ComputePairedMean {
    my ($vector1_ref, $vector2_ref, $missing_value_designator, $min_presence1, $min_presence2) = @_;
    my @vector1 = @{$vector1_ref};
    my @vector2 = @{$vector2_ref};
    my $len = @vector1;
    my $n=0;
    my $r=1;
    for(my $i=0; $i<$len; $i++) {
	if((defined $vector1[$i] && defined $vector2[$i]) && ($vector1[$i] ne $missing_value_designator && $vector2[$i] ne $missing_value_designator)) {
	    $a = $vector1[$i];
	    $b = $vector2[$i];
	    $r = $r * $a / $b;
	    $n++;
        }
    }
    if($n<$min_presence1 || $n<$min_presence2) {
	$r="NA";
    }
    else {
	$r = $r ** (1/$n);
    }
    return $r;
}

sub ComputeGeometricMean {
    my ($vector_ref, $missing_value_designator, $min_presence) = @_;
    my @vector = @{$vector_ref};
    my $len = @vector;
    my $n=0;
    my $r=1;
    for(my $i=0; $i<$len; $i++) {
	if(defined $vector[$i] && $vector[$i] ne $missing_value_designator) {
	    $a = $vector[$i];
	    $r = $r * $a;
	    $n++;
        }
    }
    if($n<$min_presence) {
	$r="NA";
    }
    else {
	$r = $r ** (1/$n);
    }
    return $r;
}


sub PrintPermutationMatrix {

# this subroutine prints the matrix of permutations for DEBUG

    my ($num_conds, $perm_matrix_ref, $design) = @_;
    my @permutations = @{$perm_matrix_ref};

    if(!($design eq "D")) {
	for(my $i=1; $i<$num_conds; $i++) {
	    print "---- condition $i ----\n";
	    my $n = @{$permutations[$i]};
	    for(my $j=0; $j<$n; $j++) {
		my $m = @{$permutations[$i][$j]};
		print "$permutations[$i][$j][0]";
		for(my $k=1; $k<$m; $k++) {
		    print ",$permutations[$i][$j][$k]";
		}
		print "\n";
	    }
	}
    }
    else {
	for(my $i=0; $i<$num_conds; $i++) {
	    print "---- condition $i ----\n";
	    my $n = @{$permutations[$i]};
	    for(my $j=0; $j<$n; $j++) {
		my $m = @{$permutations[$i][$j]};
		print "$permutations[$i][$j][0]";
		for(my $k=1; $k<$m; $k++) {
		    print ",$permutations[$i][$j][$k]";
		}
		print "\n";
	    }
	}
    }
}


sub InitializePermuationArray {

    my ($num_conds, $num_reps_ref, $num_perms, $paired, $design) = @_;
    my @num_reps = @{$num_reps_ref};

    my @perm_array;
    my $perm_array_ref;


    if(!($design eq "D")) {
	for(my $i=1; $i<$num_conds; $i++) {
	    if($paired == 0) {
		my $size_of_group1 = $num_reps[0];
		my $size_of_group2 = $num_reps[$i];
		my $sum = $size_of_group1 + $size_of_group2;
		my $n;
		if($sum-$size_of_group1 > $size_of_group1) {
		    $n = $size_of_group1;
		}
		else {
		    $n = $sum-$size_of_group1;
		}
		my $m = $sum;
		my $num_all_perms=Binomial($m,$n);
		if($num_all_perms < $num_perms+25) {
		    $perm_array_ref = GetAllSubsetsOfFixedSize($sum, $size_of_group1);
		}
		else {
		    $perm_array_ref = GetRandomSubsetsOfFixedSize($sum, $size_of_group1, $num_perms);
		}
		@{$perm_array[$i]} = @{$perm_array_ref};
	    }
	    else {
		my $size_of_group = $num_reps[0];
		my $number_subsets = 2**$size_of_group;
		my $temp_array_ref;
		if($number_subsets < $num_perms+25) {
		    $temp_array_ref = ListAllSubsets($size_of_group);
		}
		else {
		    $temp_array_ref = GetRandomSubsets($size_of_group, $num_perms);
		}
		my @temp_array = @{$temp_array_ref};
		my $n = @temp_array;
		for(my $p=0; $p<$n; $p++) {
		    for(my $j=0; $j<$num_reps[0]; $j++) {
			$perm_array[$i][$p][$j] = $temp_array[$p][$j];
			$perm_array[$i][$p][$j+$num_reps[0]] = 1-$temp_array[$p][$j];
			if($p==0) {
			    $perm_array[$i][$p][$j] = 1;
			    $perm_array[$i][$p][$j+$num_reps[0]] = 0;
			}
		    }
		}
	    }
	}
    }
    else {
	for(my $i=0; $i<$num_conds; $i++) {
	    my $size_of_group = $num_reps[$i];
	    my $number_subsets = 2**$size_of_group;
	    if($number_subsets < $num_perms+25) {
		$perm_array_ref = ListAllSubsets($size_of_group);
	    }
	    else {
		$perm_array_ref = GetRandomSubsets($size_of_group, $num_perms);
	    }
	    @{$perm_array[$i]} = @{$perm_array_ref};
	}
    }
    return \@perm_array;
}

sub ListAllSubsets {
    my($size_of_set) = @_;
    my @perm_array;
    my $num_subsets = 2**$size_of_set;

    for(my $i=0; $i<$num_subsets; $i++) {
	for(my $j=0; $j<$size_of_set; $j++) {
	    $perm_array[$i][$j] = 0;
	}
    }
    my @counter;
    my $perm_counter = 0;
    for(my $subsetsize=1; $subsetsize < $size_of_set+1; $subsetsize++) {
	for(my $i=1; $i<$subsetsize+1; $i++) {
	    $counter[$i]=$i;
	}
	my $flag=0;
	while($flag==0) {
	    $perm_array[$perm_counter][$counter[1]-1]++;
	    for(my $p=2; $p<$subsetsize+1; $p++) {
		$perm_array[$perm_counter][$counter[$p]-1]++;
	    }
	    $perm_counter++;
	    my $j=$size_of_set;
	    my $jj=$subsetsize;
	    while(($counter[$jj]==$j) && ($j>0)) {
		$jj--;
		$j--;
	    }
	    if($jj==0) {
		$flag=1;
	    }
	    if($jj>0) {
		$counter[$jj]++;
		my $k=1;
		for(my $i=$jj+1; $i<$subsetsize+1; $i++) {
		    $counter[$i]=$counter[$jj]+$k;
		    $k++;
		}
	    }
	}
    }

    return \@perm_array;
}

sub GetRandomSubsets {
    my ($size_of_set, $num_subsets) = @_;
    my @subset_array;

    for(my $i=0; $i<$num_subsets; $i++) {
	for(my $j=0; $j<$size_of_set; $j++) {
	    $subset_array[$i][$j] = 0;
	}
    }
    for(my $i=1; $i<$num_subsets; $i++) {
	for(my $j=0; $j<$size_of_set; $j++) {
	    my $flip = int(rand(2));
	    if($flip == 1) {
		$subset_array[$i][$j]++;
	    }
	}
    }
    return \@subset_array;
}
sub GetRandomSubsetsOfFixedSize {

    my ($size_of_set, $size_of_subset, $num_subsets) = @_;

    my @subset_array;
    my $counter=0;
    my $subset_ref;
    for(my $j=0; $j<$size_of_subset; $j++) {
	$subset_array[0][$j]=1;
    }
    for(my $j=$size_of_subset; $j<$size_of_set; $j++) {
	$subset_array[0][$j]=0;
    }
    for(my $i=1; $i<$num_subsets; $i++) {
	$subset_ref = ChooseRand($size_of_subset, $size_of_set);
	my @subset = @{$subset_ref};
	for(my $j=0; $j<$size_of_set; $j++) {
	    $subset_array[$i][$j]=0;
	}
	my $n = @subset;
	for(my $j=0; $j<$n; $j++) {
	    $subset_array[$i][$subset[$j]-1]++;
	}
    }
    return \@subset_array;
}

sub ChooseRand {
    my ($subsetsize, $groupsize)=@_;
    my $flag=0;
    my $x;
    my @subset;

    for(my $i=0; $i<$subsetsize; $i++) {
	$flag=0;
	while($flag==0) {
	    $flag=1;
	    $x=int(rand($groupsize))+1;
	    for(my $j=0; $j<$i; $j++) {
		if($x==$subset[$j]) {
		    $flag=0;
		}
	    }
	}
	$subset[$i]=$x;
    }
    return \@subset;
}


sub Binomial {
    my ($m, $n) = @_;
    my $total=1;
    for(my $j=0; $j<$n; $j++) {
	$total = $total * ($m-$j)/($n-$j);
    }
    return $total;
}

sub GetAllSubsetsOfFixedSize {

    my ($size_of_set, $size_of_subset) = @_;

    my @subset_array;
    my @counter;

    for(my $i=1; $i<$size_of_subset+1; $i++) {
	$counter[$i]=$i;
    }
    my $flag=0;
    my $subset_counter = 0;

    my $num_subs = Binomial($size_of_set, $size_of_subset);

    for(my $i=0; $i<$num_subs; $i++) {
	for(my $j=0; $j<$size_of_set; $j++) {
	    $subset_array[$i][$j]=0;
	}
    }

    while($flag==0) {
	$subset_array[$subset_counter][$counter[1]-1]++;
	for(my $p=2; $p<$size_of_subset+1; $p++) {
	    $subset_array[$subset_counter][$counter[$p]-1]++;
	}
	my $j=$size_of_set;
	my $jj=$size_of_subset;
	while(($counter[$jj]==$j) && ($j>0)) {
	    $jj--;
	    $j--;
	}
	if($jj==0) {
	    $flag=1;
	}
	if($jj>0) {
	    $counter[$jj]++;
	    my $k=1;
	    for(my $i=$jj+1; $i<$size_of_subset+1; $i++) {
		$counter[$i]=$counter[$jj]+$k;
		$k++;
	    }
	}
	$subset_counter++;
    }

    return \@subset_array;
}


sub PrintData {

# this prints out the entire data matrix (for DEBUG)

    my ($data_ref, $data_A_ref, $ids_ref) = @_;
    my @data = @{$data_ref};
    my @data_A = @{$data_A_ref};
    my @ids = @{$ids_ref};

    my $num_ids = @data;
    print "num_ids = $num_ids\n";
    my $num_conds = @{$data[1]};
    for(my $k=0; $k<$num_ids; $k++) {
	print "ID:$ids[$k]\n";
	for(my $kk=0; $kk<$num_conds; $kk++) {
	    print "CONDITION: $kk\n";
	    print "\nV:";
	    my $num_reps = @{$data[$k][$kk]};
	    for(my $kkk=0; $kkk<$num_reps; $kkk++) {
		print "$data[$k][$kk][$kkk]\t";
	    }
	    print "\nA:";
	    for(my $kkk=0; $kkk<$num_reps; $kkk++) {
		print "$data_A[$k][$kk][$kkk]\t";
	    }
	    print "\n";
	}
	print "\n";
    }
}

sub ReadInfile {

    my ($num_channels, $paired, $avalues, $design, $data_is_logged, $use_logged_data, $shift, $infile, $noavalues, $unpaired, $data_not_logged, $missing_value_designator, $infile, $num_cols, $cond_ref, $rep_ref, $cond_A_ref, $rep_A_ref, $num_conds, $num_reps_ref, $num_cols) = @_;
    my @cond = @{$cond_ref};
    my @rep = @{$rep_ref};
    my @cond_A = @{$cond_A_ref};
    my @rep_A = @{$rep_A_ref};
    my @num_reps = @{$num_reps_ref};
    my %ids_hash;

# the next lines get the ids to use if id_filter_file is set

    my $ids = ":::abc:::def:::";
    if($id_filter_file ne "") {
	my $openflag = open(IDFILE, $id_filter_file);
	if($openflag == 0) {
	    die("\nError: The file name given for the ID's file does not appear to exists\n\n");
	}
	while(my $id = <IDFILE>) {
	    chomp($id);
	    $ids = $ids . "$id";
	    $ids = $ids . ":::abc:::def:::";
	}
	close(IDFILE);
    }

# the following reads in the data line by line, and puts it in a data structure

    open(INFILE, $infile);
    my $line_h = <INFILE>;
    while($line_h =~ /^\s*#/) {
	  $line_h = <INFILE>;
      }

    my $line_counter=0;
    my @data;
    my @data_A;
    my @ids;
    my $min_intensity = 10000000;
    my $max_intensity = -10000000;
    while(my $line=<INFILE>) {
	my @a = split(/\t/,$line);
	my $n = @a;
	if($id_filter_file eq "" || $ids =~ /:::abc:::def:::$a[0]:::abc:::def:::/) {
	    if($n != $num_cols) {
		die("\nError: The data in line $line_counter has a different number of columns than the header line.  Please check the file format and restart.\n\n");
	    }
	    for(my $i=1; $i<$n; $i++) {
		if(($cond[$i] ne "" && $rep[$i] ne "")  || ($cond_A[$i] ne "" && $rep_A[$i] ne "")) {
		    my $value = $a[$i];
		    chomp($value);
		    $value =~ s/[\s]$//;
		    if(!($value =~ /\S/) || $value eq $missing_value_designator) {
			if($cond[$i] ne "") {
			    $data[$line_counter][$cond[$i]][$rep[$i]] = "missing";
			}
			else {
			    $data_A[$line_counter][$cond[$i]][$rep[$i]] = "missing";
			}
		    }
		    else {
			if ((((!($value =~ /^-?[0-9]+$/)) && (!($value =~ /^-?[0-9]*\.[0-9]*$/))) && (!($value=~/^-?[0-9]+\.[0-9]*$/))) && (!($value=~/^-?[0-9]*\.?[0-9]*[eE][+-]\d*$/))) {
			    if($missing_value_designator eq " ") {
				my $l = $line_counter+2;
				print "\nThe value in row $l, column $i is not a number, it is $value.";
				print "\nIs $value your missing value designator? (answer Y or N):\n";
				my $answer = <STDIN>;
				chomp($answer);
				if($answer eq "y" || $answer eq "Y") {
				    $missing_value_designator = $value;
				    $answer = "Y";
				    $data[$line_counter][$cond[$i]][$rep[$i]] = "missing";
				}
				if($answer eq "n" || $answer eq "N") {
				    die("\nError: Please fix your input file and restart.\n\n");
				    $answer = "N";
				}
				while($answer ne "Y" && $answer ne "N") {
				    print "\nPlease input Y or N:\n";
				    $answer = <STDIN>;
				    chomp($answer);
				    if($answer eq "y" || $answer eq "Y") {
					$missing_value_designator = $value;
					$answer = "Y";
					$data[$line_counter][$cond[$i]][$rep[$i]] = "missing";
				    }
				    if($answer eq "n" || $answer eq "N") {
					die("\nError: Please fix your input file and restart.\n\n");
					$answer = "N";
				    }
				}
			    }
			    else {
				my $l = $line_counter+2;
				die("\nError: The value in row $l, column $i is not a number (it is \"$value\"), and it is not equal to your missing value designator, which is \"$missing_value_designator\".\n");
			    }

			}
			else {
			    if($cond[$i] =~ /\S/) {
				$data[$line_counter][$cond[$i]][$rep[$i]] = $value;
				if($value<$min_intensity) {
				    $min_intensity = $value;
				}
				if($value>$max_intensity) {
				    $max_intensity = $value;
				}
			    }
			    else {
				$data_A[$line_counter][$cond_A[$i]][$rep_A[$i]] = $value;
			    }
			}
		    }
		}
	    }
	    $ids[$line_counter] = $a[0];
	    $ids_hash{$a[0]} = $line_counter;
	    $line_counter++;
	}
    }


# Set the NEGS variable (this notes a special case where must shift to use ratios)
    my $NEGS;
    if(($num_channels == 1 && $data_is_logged == 0) && ($min_intensity<=0)) {
	$NEGS = 1;
    }
    else {
	$NEGS = 0;
    }

# Check that they are not claiming to have unlogged 2-channel data when there are
# actually negatives in the file

    if(($num_channels == 2 && $data_is_logged == 0) && ($min_intensity<=0)) {
	die("\nError: You cannot have negative values in your data file if it is two channel unlogged data.\nThe intensity $min_intensity appears in your data file.\nPlease fix this and restart.\n\n");
    }
#    if($NEGS == 1 && $use_logged_data == 1) {
#	$shift = CheckForNegsSpecialCase($shift, $num_channels, $NEGS, $means, $medians, $min_intensity, $max_intensity, $use_logged_data);
#    }
    if(($num_channels == 2 && $design eq "R") && $num_conds == 1) {
	die("\nError: If there are two channels and a reference design,\nthen there must be at least two conditions.\n\n");
    }
#    if(($num_channels == 2 && $design eq "D") && $num_conds > 1) {
#	die("\nError: If there are two channels and a direct comparison design,\nthen there must be only one condition.\n\n");
#    }
    my $num_ids = @ids;
    print "\nthere are $num_ids rows in your data file\n";
    return (\@data, \@data_A, \@ids, \%ids_hash, $min_intensity, $max_intensity, $NEGS);
}


sub GetUserInput {

    my ($num_channels, $paired, $avalues, $design, $data_is_logged, $noavalues, $unpaired, $data_not_logged, $missing_value_designator, $no_background_filter, $background_filter_strictness, $level_confidence_list, $level_confidence, $min_presence, $min_presence_list, $infile, $medians, $means, $tstat, $use_logged_data, $use_unlogged_data, $pool, $outfile) = @_;


    my $stat;

# allow for variations on the design string

    if($design eq "r") { $design = "R"; }
    if($design eq "Ref") { $design = "R"; }
    if($design eq "Reference") { $design = "R"; }
    if($design eq "ref") { $design = "R"; }
    if($design eq "reference") { $design = "R"; }

    if($design eq "d") { $design = "D"; }
    if($design eq "Dir") { $design = "D"; }
    if($design eq "Direct") { $design = "D"; }
    if($design eq "dir") { $design = "D"; }
    if($design eq "direct") { $design = "D"; }

    if(($design eq "R" || $design eq "D") && $num_channels == 1)
    {
	die("\nError: if the number of channels is 1, do not specify the design type\n\n");
    }
    if($design eq "R" || $design eq "D")
    {
	$num_channels = 2;
    }

# get number of channels
    if($num_channels != 1 && $num_channels !=2) {
	if($num_channels ne "") {
	    print "\nThere cannot be \"$num_channels\" of channels, please enter the number of channels (1 or 2): ";
	}
	else {
	    print "\nAre the arrays 1-Channel or 2-Channel arrays?  (enter \"1\" or \"2\")  ";
	}
	$num_channels = <STDIN>;
	chomp($num_channels);
	while($num_channels != 1 && $num_channels != 2) {
	    print "\nPlease input 1 or 2:\n";
	    $num_channels = <STDIN>;
	    chomp($num_channels);
	}
    }
    else {
	print "\nNumber of Channels = $num_channels\n";
    }

# things relating to two channel data:
    if($num_channels == 2) {
# the A-values, if not specified command line, ask user
# uncomment out the following when the background filter is implemented
#	if($noavalues ne "1" && $avalues ne "1") {
#	    print "\nAre the A-values included? (enter Y or N): ";
#	    $avalues = <STDIN>;
#	    chomp($avalues);
#	    if($avalues eq "y" || $avalues eq "Y") {
#		$avalues = "1";
#	    }
#	    if($avalues eq "n" || $avalues eq "N") {
#		$avalues = "0";
#	    }
#	    while($avalues ne "0" && $avalues ne "1") {
#		print "\nPlease input Y or N:\n";
#		$avalues = <STDIN>;
#		chomp($avalues);
#		if($avalues eq "y" || $avalues eq "Y") {
#		    $avalues = "1";
#		}
#		if($avalues eq "n" || $avalues eq "N") {
#		    $avalues = "0";
#		}
#	    }
#	}
#	else {
#	    if($avalues == 1) {
#		print "The A-values are expected in the input file\n";
#	    }
#	    if($noavalues == 1) {
#		$avalues = 0;
#	    }
#	}
	$avalues = 0;

# if num channels is 2 and arrays are paired, then this must be a reference design
	if($paired == 1) {
	    $design = "R";
	}
# if design not specified command line, ask user
	if($design ne "R" && $design ne "D") {
	    print "\nIs it a reference design or direct comparison design?: (enter R or D): ";
	    $design = <STDIN>;
	    chomp($design);
	    if($design eq "r") {
		$design = "R";
	    }
	    if($design eq "d") {
		$design = "D";
	    }
	    while($design ne "R" && $design ne "D") {
		print "\nPlease input R or D:\n";
		$design = <STDIN>;
		chomp($design);
		if($design eq "r") {
		    $design = "R";
		}
		if($design eq "d") {
		    $design = "D";
		}
	    }
	}
	else {
	    if($design eq "D") {
		print "\nDirect comparision design expected\n";
	    }
	    if($design eq "R") {
		print "\nReference design expected\n";
	    }
	}
	if($design eq "D") {
	    $paired = "NA";
	}
    }
    else {
# if number of channels is one, a-values and design type not relevant
	$avalues = "NA";
	$design = "NA";
    }
    my $start;
    if($design ne "D") {
	$start = 1;
    }
    else {
	$start = 0;
    }

    if($medians == 1 && $paired == 1) {
	die("\nError: Do not use the median statistic with paired data.\n\n");
    }
# in cases paired matters, and if not specified command line, ask the user
    if($design ne "D" || $num_channels == 1) {
	if($stat == 2 || $medians ==1) {
	    $paired = 0;
	    $unpaired = 1;
	}
	if($unpaired ne "1" && $paired ne "1") {
	    print "\nAre the arrays paired? (enter Y or N): ";
	    $paired = <STDIN>;
	    chomp($paired);
	    if($paired =~ /(y|Y)/) {
		$paired = 1;
	    }
	    if($paired =~ /(n|N)/) {
		$paired = 0;
	    }
	    while($paired ne "0" && $paired ne "1") {
		print "\nPlease input Y or N:\n";
		$paired = <STDIN>;
		chomp($paired);
		if($paired =~ /(y|Y)/) {
		    $paired = 1;
		}
		if($paired =~ /(n|N)/) {
		    $paired = 0;
		}
	    }
	}
	else {
	    if($paired == "1" && $unpaired == "1") {
		die("\nError: Do not specify both --paired and --unpaired\n\n");
	    }
	    if($paired == "1") {
		print "\nPaired data expected\n";
	    }
	    if($unpaired == "1") {
		$paired = 0;
	    }
	}
    }
# if not specified command line whether the data is logged, ask user
    if($data_not_logged ne "1" && $data_is_logged ne "1") {
	print "\nIs the data log transformed? (enter Y or N): ";
	$data_is_logged = <STDIN>;
	chomp($data_is_logged);
	if($data_is_logged eq "y" || $data_is_logged eq "Y") {
	    $data_is_logged = 1;
	}
	if($data_is_logged eq "n" || $data_is_logged eq "N") {
	    $data_is_logged = 0;
	}
	while($data_is_logged ne "0" && $data_is_logged ne "1") {
	    print "\nPlease input Y or N:\n";
	    $data_is_logged = <STDIN>;
	    chomp($data_is_logged);
	    if($data_is_logged =~ /(y|Y)/) {
		$data_is_logged = 1;
	    }
	    if($data_is_logged =~ /(n|N)/) {
		$data_is_logged = 0;
	    }
	}
    }
    else {
	if($data_is_logged == 1) {
	    print "\nLogged data expected\n";
	}
	if($data_not_logged == 1) {
	    print "\nUnlogged data expected\n";
	    $data_is_logged = 0;
	}
    }

    if(!($missing_value_designator =~ /\S/)) {
	$missing_value_designator = " ";
    }

# When background filter implemented uncomment the following
# if not specified command line whether to filter out background, ask user
#    if($no_background_filter == "" && $background_filter_strictness == "") {
#	print "\nDo you want to filter out genes which are expressed at background level everywhere? (Y or N): ";
#	my $answer = <STDIN>;
#	chomp($answer);
#        if($answer eq "y") {
#            $answer = 1;
#        }
#        if($answer eq "n") {
#            $answer = 0;
#        }
#       while($answer ne "0" && $answer ne "1") {
#          print "\nPlease input Y or N:\n";
#            $answer = <STDIN>;
#            chomp($answer);
#            if($answer eq "y" || $answer eq "Y") {
#                $answer = 1;
#            }
#            if($answer eq "n" || $answer eq "N") {
#                $answer = 0;
#            }
#	    while($answer ne "0" && $answer ne "1") {
#		print "\nPlease input Y or N:\n";
#		$answer = <STDIN>;
#		chomp($answer);
#		if($answer eq "y" || $answer eq "Y") {
#		    $answer = 1;
#		}
#		if($answer eq "n" || $answer eq "N") {
#		    $answer = 0;
#		}
#	    }
#        }
#    }

# read in the header and check for validity, and get data on number of conditions and reps
# the first four arguments returned are needed for the ReadInfile subroutine
    my ($cond_ref, $rep_ref, $cond_A_ref, $rep_A_ref, $num_conds, $num_reps_ref, $infile, $num_cols) = ReadHeader($infile, $avalues, $num_channels, $paired);

    my @num_reps = @{$num_reps_ref};


# if level confidence not specified command line, ask user

    my @level_confidence_array;
# if neither level_confidence nor level_confidence_list specified:
    if(!$level_confidence =~ /\S/ && !$level_confidence_list =~ /\S/) {
	if($num_conds>1+$start) {
	    print "\nPlease enter the level confidence (a number between 0 and 1)\n(or enter S to specify a separate confidence for each group\nor enter L to give the confidence later): ";
	}
	else {
	   print "\nPlease enter the level confidence (a number between 0 and 1)\n(or L to give the confidence later): ";
	}
	$level_confidence = <STDIN>;
	chomp($level_confidence);
	while(((!(0 < $level_confidence && $level_confidence < 1) || !($level_confidence =~ /^0?\.\d*$/)) && ((($level_confidence ne "S" && $level_confidence ne "s")) || $num_conds == 1+$start))  && ($level_confidence ne "L" && $level_confidence ne "l")) {
	    if($num_conds>1+$start) {
		print "\n\nThe level confidence must be a number strictly beween 0 and 1.\n(or enter S to specify a separate confidence for each group\nor L to specify the confidence later)\nPlease re-enter it: ";
	    }
	    else {
		print "\n\nThe level confidence must be a number strictly beween 0 and 1\n(or enter L to specify the confidence later).\nPlease re-enter it: ";
	    }
	    $level_confidence = <STDIN>;
	    chomp($level_confidence);
	}
	if($level_confidence eq "L" || $level_confidence eq "l") {

	}
	else {
	    if($level_confidence eq "S" || $level_confidence eq "s") {
		for(my $i=$start; $i<$num_conds; $i++) {
		    print "\nEnter the level confidence for group $i: ";
		    $level_confidence_array[$i] = <STDIN>;
		    chomp($level_confidence_array[$i]);
		    while(!(0 < $level_confidence_array[$i] && $level_confidence_array[$i] < 1) || !($level_confidence_array[$i] =~ /^0?\.\d*$/)) {
			print "\nThe level confidence must be a number strictly beween 0 and 1.\nPlease re-enter it: ";
			$level_confidence_array[$i] = <STDIN>;
			chomp($level_confidence_array[$i]);
		    }
		}
	    }
	    else {
		for(my $i=0; $i<$num_conds; $i++) {
		    $level_confidence_array[$i] = $level_confidence;
		}
	    }
	}
    }
    else {
  # if level_confidence specified:
	if(!($level_confidence =~ /^(L|l)$/) ||  $level_confidence_list =~ /\S/) {
	    if($level_confidence =~ /\S/  && !$level_confidence_list =~ /\S/) {
		if((!(0 < $level_confidence && $level_confidence < 1)) || !($level_confidence =~ /^0?\.\d*$/)) {
		    if($num_conds>1+$start) {
			print "\nThe level confidence must be a number strictly beween 0 and 1.\nPlease re-enter it\n(or enter S to specify a separate confidence for each group): ";
		    }
		    else {
			print "\nThe level confidence must be a number strictly beween 0 and 1.\nPlease re-enter it: ";
		    }
		    $level_confidence = <STDIN>;
		    chomp($level_confidence);
		}
		while((!(0 < $level_confidence && $level_confidence < 1) || !($level_confidence =~ /^0?\.\d*$/)) && (($level_confidence ne "S" && $level_confidence ne "s") || $num_conds == (1+$start))) {
		    if($num_conds>1+$start) {
			print "\nThe level confidence must be a number strictly beween 0 and 1.\nPlease re-enter it\n(or enter S to specify a separate confidence for each group): ";
		    }
		    else {
			print "\nThe level confidence must be a number strictly beween 0 and 1.\nPlease re-enter it: ";
		    }
		    $level_confidence = <STDIN>;
		    chomp($level_confidence);
		}
		if($level_confidence eq "S" || $level_confidence eq "s") {
		    for(my $i=$start; $i<$num_conds; $i++) {
			print "\nEnter the level confidence for group $i: ";
			$level_confidence_array[$i] = <STDIN>;
			chomp($level_confidence_array[$i]);
			while(!(0 < $level_confidence_array[$i] && $level_confidence_array[$i] < 1) || !($level_confidence_array[$i] =~ /^0?\.\d*$/)) {
			    print "\nThe level confidence must be a number strictly beween 0 and 1.\nPlease re-enter it: ";
			    $level_confidence_array[$i] = <STDIN>;
			    chomp($level_confidence_array[$i]);
			}
		    }
		}
		else {
		    for(my $i=$start; $i<$num_conds; $i++) {
			$level_confidence_array[$i] = $level_confidence;
		    }
		}
	    }
  # if level_confidence_list specified:
	    else {
		my @a = split(/,/, $level_confidence_list);
		my $n = @a;
		my $temp = $num_conds-$start;
		if($n != $num_conds-$start) {
		    die("\nError: There is something wrong with the format of your level confidence list.\nIt should be a comma separated list of length $temp\n\n");
		}
		for(my $j=0; $j<$n; $j++) {
		    if(!(0 < $a[$j] && $a[$j] < 1) || !($a[$j] =~ /^0?\.\d*$/)) {
			my $k=$j+1;
			die("\nError: There is something wrong with the format of your level confidence list \"$level_confidence_list\".\nPosition $k is $a[$j] while it should be a number between 0 and 1\n\n");
		    }
		    else {
			$level_confidence_array[$j+$start] = $a[$j];
		    }
		}
	    }
	}
    }

    if(!$level_confidence =~ /(l|L)/) {
	print "\n";
	for(my $i=$start; $i<$num_conds; $i++) {
	    print "level_confidence for group $i = $level_confidence_array[$i]\n";
	}
    }

# parse the min_presence_list, if provided, otherwise ask for it

    my @min_presence_array;
    if(!$min_presence =~ /\S/ && !$min_presence_list =~ /\S/) {
	if($num_conds>1) {
	    print "\nPlease enter the min number of non-missing values there must be in each\ncondition for a row to not be ignored (a positive integer greater than 1)\n(or enter S to specify a separate one for each condition): ";
	}
	else {
	    print "\nPlease enter the min number of non-missing values necessary\nfor a row to not be ignored (a positive integer greater than 1): ";
	}
	$min_presence = <STDIN>;
	chomp($min_presence);
	while((!(1 < $min_presence) || !($min_presence =~ /^\d+$/)) && (($min_presence ne "S" && $min_presence ne "s") || $num_conds == $start)) {
	    if($num_conds>2) {
		    print "\nIt must be a positive integer greater than 1\nPlease re-enter it (or enter S to specify a separate one for each group): ";
	    }
	    else {
		    print "\nIt must be a positive integer greater than 1\nPlease re-enter it: ";
	    }
	    $min_presence = <STDIN>;
	    chomp($min_presence);
	}
	if($min_presence eq "S" || $min_presence eq "s") {
	    for(my $i=0; $i<$num_conds; $i++) {
		print "\nEnter for condition $i: ";
		$min_presence_array[$i] = <STDIN>;
		chomp($min_presence_array[$i]);
		while(!(1 < $min_presence_array[$i]) || !($min_presence_array[$i] =~ /^\d+$/)) {
		    print "\nIt must be a positive integer greater than 1.\nPlease re-enter it: ";
		    $min_presence_array[$i] = <STDIN>;
		    chomp($min_presence_array[$i]);
		}
	    }
	}
	else {
	    for(my $i=0; $i<$num_conds; $i++) {
		$min_presence_array[$i] = $min_presence;
	    }
	}
    }
    else {
	if($min_presence =~ /\S/) {
	    if((!(1 < $min_presence)) || !($min_presence =~ /^\d+$/)) {
		if($num_conds>2) {
		    print "\nIt must be a positive integer greater than 1\nPlease re-enter it (or enter S to specify a separate one for each group): ";
		}
		else {
		    print "\nIt must be a positive integer greater than 1\nPlease re-enter it: ";
		}
		$min_presence = <STDIN>;
		chomp($min_presence);
	    }
	    while((!(1 < $min_presence) || !($min_presence =~ /^\d+$/)) && (($min_presence ne "S" && $min_presence ne "s") || $num_conds == 2)) {
		if($num_conds>2) {
		    print "\nIt must be a positive integer greater than 1\nPlease re-enter it (or enter S to specify a separate one for each group): ";
		}
		else {
		    print "\nIt must be a positive integer greater than 1\nPlease re-enter it: ";
		}
		$min_presence = <STDIN>;
		chomp($min_presence);
	    }
	    if($min_presence eq "S" || $min_presence eq "s") {
		for(my $i=1; $i<$num_conds; $i++) {
		    print "\nEnter for group $i: ";
		    $min_presence_array[$i-1] = <STDIN>;
		    chomp($min_presence_array[$i-1]);
		    while(!(1 < $min_presence_array[$i-1]) || !($min_presence_array[$i-1] =~ /^\d+$/)) {
			    print "\nIt must be a positive integer greater than 1.\nPlease re-enter it: ";
			$min_presence_array[$i-1] = <STDIN>;
			chomp($min_presence_array[$i-1]);
		    }
		}
	    }
	    else {
		for(my $i=0; $i<$num_conds; $i++) {
		    $min_presence_array[$i] = $min_presence;
		}
	    }
	}
	else {
	    my @a = split(/,/, $min_presence_list);
	    my $n = @a;
	    if($n != $num_conds) {
		die("\nError: There is something wrong with the format of your min presence list.\nIt should be a comma separated list of length $num_conds\n\n");
	    }
	    for(my $j=0; $j<$n; $j++) {
		if(!(1 < $a[$j]) || !($a[$j] =~ /^\d+$/)) {
		    my $k=$j+1;
		    die("\nError: There is something wrong with the format of your min presence list \"$min_presence_list\".\nPosition $k is $a[$j] while it should be an integer greater than 1\n\n");
		}
		else {
		    $min_presence_array[$j] = $a[$j];
		}
	    }
	}
    }

    for(my $i=0; $i<$num_conds; $i++) {
	if($min_presence_array[$i]>$num_reps[$i]) {
	    $min_presence_array[$i] = $num_reps[$i];
	}
    }
    print "\n";
    for (my $i=0; $i<$num_conds; $i++) {
	print "Min presence required for condition $i: $min_presence_array[$i]\n";
    }


# DEBUG
#    print STDERR "\n";
#    for(my $i=0; $i<$num_conds; $i++) {
#	print STDERR "min_presence_array[$i] = $min_presence_array[$i]\n";
#    }
# DEBUG


# Ask user for the statistic to use:

    if(!$medians =~ /\S/) {
	$medians = 0;
    }
    else {
	if($design eq "D") {
	    die("\nError: For direct design you can specify only means or tstat, do not specify medians.\n\n");
	}
	if($paired == 1) {
	    die("\nError: For a paired analysis you can specify only means or tstat, do not specify medians.\n\n");
	}
    }
    if(!$means =~ /\S/) {
	$means = 0;
    }
    if(!$tstat =~ /\S/) {
	$tstat = 0;
    }
    if($medians + $means + $tstat > 1) {
	die("\nError: only one of means, medians, tstat can be defined.\n\n");
    }
    if($medians + $means + $tstat == 0) {
	if($design eq "D") {
	    print "\nWhat statistic would you like to use?\n";
	    print "The T-statistic (enter 0)\n";
	    print "The Gemoetric Mean (enter 1)\n";
	    $stat = <STDIN>;
	    chomp($stat);
	    while($stat ne "0" && $stat ne "1") {
		print "\nPlease enter 0 or 1: ";
		$stat = <STDIN>;
		chomp($stat);
	    }
	    if($stat eq "0") {
		$tstat = 1;
	    }
	    if($stat eq "1") {
		$means = 1;
	    }
	}
	if($design eq "R" || $num_channels == 1) {
	    if($paired == 1) {
		print "\nWhat statistic would you like to use?\n";
		print "The T-statistic (enter 0)\n";
		print "The Ratio (enter 1)\n";
		my $stat = <STDIN>;
		chomp($stat);
		while($stat ne "0" && $stat ne "1") {
		    print "\nPlease enter 0 or 1: ";
		    $stat = <STDIN>;
		    chomp($stat);
		}
		if($stat eq "0") {
		    $tstat = 1;
		}
		if($stat eq "1") {
		    $means = 1;
		}
	    }
	    else {
		print "\nWhat statistic would you like to use?\n";
		print "The T-statistic (enter 0)\n";
		print "The Ratio with means (enter 1)\n";
#		print "The Ratio with medians (enter 2)\n";
		$stat = <STDIN>;
		chomp($stat);
#		while(($stat ne "0" && $stat ne "1") && $stat ne "2") {
#		    print "\nPlease enter 0, 1, or 2: ";
		while($stat ne "0" && $stat ne "1") {
		    print "\nPlease enter 0 or 1: ";
		    $stat = <STDIN>;
		    chomp($stat);
		}
		if($stat eq "0") {
		    $tstat = 1;
		}
		if($stat eq "1") {
		    $means = 1;
		}
		if($stat eq "2") {
		    $medians = 1;
		}
	    }
	}
    }
    if(($tstat == 1 || $stat == 0) && ($design eq "D" && $use_unlogged_data == 1)) {
	die("\nError: if using the t-stat with direct comparison design, then cannot use --use_unlogged_data option.\n\n");
    }
    if(($tstat == 1 || $stat == 0) && $design eq "D") {
	$use_logged_data = 1;
    }

    if(($means == 1 || $stat == 1) && ($design eq "D" && $use_logged_data == 1)) {
	die("\nError: if using the means with direct comparison design, then cannot use --use_logged_data option.\n\n");
    }
    if(($means == 1 || $stat == 1) && $design eq "D") {
	$use_logged_data = 0;
    }
    if($tstat == 1 && $use_logged_data =~ /^\s*$/) {
	print "\nDo you want to run algorithm on the log transformed data? (Y or N): ";
	my $answer = <STDIN>;
	chomp($answer);
        if($answer eq "y") {
            $answer = 1;
        }
        if($answer eq "n") {
            $answer = 0;
        }
        while($answer ne "0" && $answer ne "1") {
            print "\nPlease input Y or N:\n";
            $answer = <STDIN>;
            chomp($answer);
            if($answer eq "y" || $answer eq "Y") {
                $answer = 1;
            }
            if($answer eq "n" || $answer eq "N") {
                $answer = 0;
            }
	    while($answer ne "0" && $answer ne "1") {
		print "\nPlease input Y or N:\n";
		$answer = <STDIN>;
		chomp($answer);
		if($answer eq "y" || $answer eq "Y") {
		    $answer = 1;
		}
		if($answer eq "n" || $answer eq "N") {
		    $answer = 0;
		}
	    }
        }
	$use_logged_data = $answer;
    }
    if($tstat != 1 && $use_logged_data == 1) {
	die("\nError: You can only use the --use_logged_data option when using the t-statistic.\n\n");
    }
    if($tstat == 1 && $shift > 0) {
	die("\nError: Do not use a shift when using the t-statistic.\n\n");
    }

#    print STDERR "\nmeans=$means, tstat=$tstat, medians=$medians\n";

    if (!defined $outfile) {
	my $backslash_flag=0;
        if($infile =~ /\\/) {
	    $backslash_flag = 1;
	    $infile =~ s/\\/\//g;
        }
	my $path;
	if($infile =~ /\//) {
	    $infile =~ /(^.*)\/([^\/]*)/;
	    $path = $1 . "/";
	    $outfile = "PaGE-results-for-" . $2;
	    $outfile = $path . $outfile;
	}
	else {
	    $outfile = "PaGE-results-for-" . $infile;
	}
	$outfile =~ s/\s//g;
	$outfile =~ s/.txt$//i;
	$outfile =~ s/_txt$//i;
	$outfile =~ s/.TXT$//i;
	$outfile =~ s/_TXT$//i;

        if($backslash_flag == 1) {
	    $outfile =~ s/\//\\/g;
	}
    }


    my %id2info;
    if (defined $id2info) {
	%id2info = %{&ReadId2Info()};
    }
    my %id2url;
    if (defined $id2url) {
	%id2url = %{&ReadId2Url()};
    }

    return ($num_channels, $paired, $avalues, $design, $data_is_logged, $missing_value_designator, $no_background_filter, $background_filter_strictness, \@level_confidence_array, \@min_presence_array, $infile, $cond_ref, $rep_ref, $cond_A_ref, $rep_A_ref, $num_conds, $num_reps_ref, $num_cols, $tstat, $means, $medians, $use_logged_data, $use_unlogged_data, $pool, $outfile, \%id2info, \%id2url, $level_confidence);
}

sub ReadHeader {

    my ($infile, $avalues, $num_channels, $paired) = @_;

# Get name of infile and make sure it can be opened
# note: infile name may be provided on the command line

    my $ifh;
    my $flag = 1;
    my $open_flag;
    if($infile eq "") {
	print "\nEnter the name of the datafile:\n";
	$infile = <STDIN>;
        chomp($infile);
    }
    $open_flag = open(INFILE, $infile);
    if($open_flag == 0) {
	$flag = 0;
    }
    if($flag == 0) {
	print "\nThe file \"$infile\" does not seem to exist, please try again:\n";
    }
    while($flag == 0) {
	print "\nPlease input the name of the infile:\n";
	$infile = <STDIN>;
	chomp($infile);
	$open_flag = open(INFILE, $infile);
	if($open_flag == 0) {
	    $flag = 0;
	    print "\nThe file \"$infile\" does not seem to exist, please try again:\n";
	}
	else {
	    $flag = 1;
	}
    }

# Read in datafile header and check for validity

    my $header = <INFILE>;

# the following line allows comment lines at the top of the file, those comment lines
# must be preceded by a "#" at the beginning of the line

    while($header =~ /^\s*#/) {
	  $header = <INFILE>;
      }

    my @header_array = split(/\t/,$header);
    my $num_cols = @header_array;

    my $c1flag=0;
    for(my $i=1; $i<$num_cols; $i++) {
	my $h = $header_array[$i];
	chomp($h);
	$h =~ s/[\s]$//;
	if(($h ne "") && (($h ne "I" && $h ne "i"))) {
	    if(!($h =~ /^(c|C)1(r|R)\d+(a|A)?/)) {
		$c1flag = 1;
	    }
	}
    }
    if($c1flag == 0) {
	$header =~ s/(c|C)1/c0/g;
	@header_array = split(/\t/,$header);
    }

    my $there_is_a_condition_0 = 0;
    my $num_conditions = 0;
    my @conds_reps_array;
    my @conds_reps_array_A;
 # this loop checks that there is a condition 0 and finds the max numbered condition
    for(my $i=1; $i<$num_cols; $i++) {
	my $h = $header_array[$i];
	chomp($h);
	$h =~ s/[\s]$//;
	if(($h ne "") && (($h ne "I" && $h ne "i") && !($h =~ /(c|C)\d+(r|R)\d+(A|)/))) {
	    die("\nError: the header of column $i is not of the right format.\n\n");
	}
	if(($h ne "I" && $h ne "i") && ($h ne "")) {
	    $h =~ /(^(c|C)\d+)/;
	    my $x = $1;
	    $x =~ /(\d+)/;
	    my $cond = $1;

	    $h =~ /((r|R)\d+)/;
	    my $y = $1;
	    $y =~ /(\d+)/;
	    my $rep = $1;
	    if(!($h =~ /(A|a)/)) {
		$conds_reps_array[$cond][$rep]++;
	    }
	    else {
		$conds_reps_array_A[$cond][$rep]++;
	    }
	    if($x eq "") {
		die("\nError: The header of column $i is not in the right format\n\n");
	    }
	    if($cond eq "") {
		die("\nError: The header of column $i is not in the right format\n\n");
	    }
	    else {
		if($cond == 0) {
		    $there_is_a_condition_0 = 1;
		}
		if($cond > $num_conditions) {
		    $num_conditions = $cond;
		}

	    }
	}
    }
    if($there_is_a_condition_0 == 0) {
		die("\nError: You must have a condition 0\n\n");
    }
 # this loop checks that there are conditions from 0 to num_conditions
    my $condition_numbering_flag = 0;
    for(my $i=1; $i<$num_conditions; $i++) {
	if(!($header =~ /\tc$i/)) {
		die("\nError: You have a condition labeled number $num_conditions, but there is no condition $i\n\n");
	}
    }
    my $temp = $num_conditions+1;
    if($temp > 1) {
	print "\n   There are $temp conditions\n";
    }
    else {
	print "\n   There is one condition\n";
    }

 # this loop checks there are there are consecutively numbered reps for each condition, and at least two
 # of each.  It also checks the a-values are there for each rep, if a-values are supposed to be included

    my @num_reps;
    for(my $i=0; $i<$num_conditions+1; $i++) {
	$num_reps[$i] = @{$conds_reps_array[$i]} - 1;
	if($num_reps[$i] == 1) {
	    die("\nError: There is only one replicate for condition $i, there must be at least two replicates per condition.\n\n");
	}
	for(my $j=1;$j<$num_reps[$i]+1;$j++) {
	    if($conds_reps_array[$i][$j] != 1) {
		if($conds_reps_array[$i][$j] == 0) {
		    die("\nError: Replicate $j of condition $i is missing, please check the header format and restart.\n\n");
		}
		else {
		    die("\nError: Replicate $j of condition $i is replicated, please check the header format and restart.\n\n");
		}
	    }
	    my $a = $conds_reps_array[$i][$j];
	    my $b = $conds_reps_array_A[$i][$j];
	    if(($a ne $b) && ($avalues eq "1")) {
		die("\nError: You declared A-values were included, but there doesn't seem to be a header indicating the A-value for condition $i, replicate $j\n\n");
	    }
	}
    }
    if($paired == 1) {
        my $right_num_reps_when_paired_flag = 0;
	my $offending_i;
        for(my $i=1; $i<$num_conditions+1; $i++) {
		if($num_reps[$i] != $num_reps[0]) {
			$right_num_reps_when_paired_flag = 1;
			$offending_i = $i;
		}
	}
	if($right_num_reps_when_paired_flag == 1) {
		die("\nError: For paired mode must have the same number of replicates in each group.  Group 0 has $num_reps[0] replicates while group $offending_i has $num_reps[$offending_i] replicates.  Please fix your data file and restart. \n\n");
	}
    }

# The following makes the arrays @cond, @rep, @cond_A, @rep_A
# which map column position to condition and rep, and similarly for
# the positions with the A value header

    my @cond;
    my @cond_A;
    my @rep;
    my @rep_A;
    for(my $i=1; $i<$num_cols; $i++) {
	my $h = $header_array[$i];
	if(($h ne "I" && $h ne "i") && ($h ne "")) {
	    $h =~ /(^(c|C)\d+)/;
	    my $x = $1;
	    $x =~ /(\d+)/;
	    my $cond = $1;
	    $h =~ /((r|R)\d+)/;
	    my $y = $1;
	    $y =~ /(\d+)/;
	    my $rep = $1;
	    if(!($h =~ /(A|a)/)) {
		$cond[$i] = $cond;
		$rep[$i] = $rep;
	    }
	    else {
		$cond_A[$i] = $cond;
		$rep_A[$i] = $rep;
	    }
	}
    }
    $num_conds = $num_conditions+1;
    print "\n";
    for(my $i=0; $i<$num_conds; $i++) {
	my $c = $i+$start;
	print "   There are $num_reps[$i] replicates in condition $c\n";
    }
    if($num_channels == 1 && $num_conds == 1) {
	die("\nError: If you have one channel data you must have at least two conditions.\n\n");
    }
    close(INFILE);
    return (\@cond, \@rep, \@cond_A, \@rep_A, $num_conds, \@num_reps, $infile, $num_cols);
}

sub WriteHtmlOutput {
    my ($data_ref, $confidence_levels_hash_ref, $num_replicates_ref, $num_conds, $silent_mode, $note, $shift, $min_ref, $max_ref, $design, $cutratios_up_ref, $cutratios_down_ref, $cutoffs_ref, $num_neg_levels_ref, $num_levels_ref, $pattern_list_ref, $num_up_ref, $num_down_ref, $clusters_ref, $aux_page_size, $outfile, $id2info, $id2url, $ids_hash, $unpermuted_means_ref, $unpermuted_stat_ref, $outliers_ref, $level_confidence_ref, $output_text, $alpha_up_ref, $alpha_down_ref, $paired, $data_is_logged, $use_logged_data, $tstat, $means, $medians, $num_perms, $min_presence_array_ref, $breakdown) = @_;

    my $min_presence_array = @{$min_presence_array_ref};

    my %id2info;
    my %id2url;
    %id2url = %{$id2url_ref};
    %id2info = %{$id2info_ref};
    my @alpha_up = @{$alpha_up_ref};
    my @alpha_down = @{$alpha_down_ref};
    my @num_replicates=@{$num_replicates_ref};
    my @data = @{$data_ref};
    my %confidence_levels_hash = %{$confidence_levels_hash_ref};
    my @min = @{$min_ref};
    my @max = @{$max_ref};
    my @cutratios_up = @{$cutratios_up_ref};
    my @cutratios_down = @{$cutratios_down_ref};
    my @cutoffs = @{$cutoffs_ref};
    my @num_neg_levels = @{$num_neg_levels_ref};
    my @num_levels = @{$num_levels_ref};
    my @pattern_list = @{$pattern_list_ref};
    my @num_up = @{$num_up_ref};
    my @num_down = @{$num_down_ref};
    my %clusters = %{$clusters_ref};
    my %ids_hash = %{$ids_hash};
    my @unpermuted_means = @{$unpermuted_means_ref};
    my @unpermuted_stat = @{$unpermuted_stat_ref};
    my @level_confidence = @{$level_confidence_ref};
    my $start;
    my %outliers = %{$outliers_ref};
    $outfile =~ s/^.*\///;
    $outfile =~ s/^.*\\//;
    my $outfiletext = $outfile;
    $outfiletext =~ s/.html//i;
    $outfiletext =~ s/.htm//i;
    $outfiletext = $outfiletext . ".txt";
    if($output_text) {
	open(TEXTOUT, ">$outfiletext");
	print TEXTOUT "$note\n\n";
    }

    if($design ne "D") {
	$start = 1;
    }
    else {
	$start = 0;
    }

    if(!($outfile =~ /\.html$/) && !($outfile =~ /\.htm$/)) {
	if(!($outfile =~ /\.HTML$/) && !($outfile =~ /\.HTM$/)) {
	    $outfile = $outfile . ".html";
	}
    }
    my $int_outfile = $outfile;
    $int_outfile =~ s/.html/_intensities1.html/;
    $int_outfile =~ s/^.*\///;
    $int_outfile =~ s/^.*\\//;
    my @print_cluster;
    my $ofh = new IO::File;
    my $ofh_intensities = new IO::File;
    my $intensity_page_counter=1;
    my $intensity_page_limit=2000;
    my $intensity_counter=0;
    my $intensity_storage;
    if($silent_mode==1) {
	print "\nGenerating html output.\n";
    }
    unless ($ofh->open(">$outfile")) {
	die "Cannot open file $outfile for writing.\n";
    }
    unless ($ofh_intensities->open(">$int_outfile")) {
	die "Cannot open file $int_outfile for writing.\n";
    }
    print "\nResults will be output to $outfile\n";

    $ofh_intensities->print("<html><body bgcolor=white>\n<title>intensities for $outfile</title>\n");
    my $title = $outfile;
    $title =~ s/.html?$//i;
    $ofh->print("<HTML>\n<title>PaGE results for $title</title>\n<BODY bgcolor=#ffffff>\n$note\n<br>&nbsp;<br>\n<CENTER><B>REPORT</B></CENTER>\n<PRE>\n");
    my $headerstuff = "Input file: $infile\n\n";
    $headerstuff = $headerstuff . "Number of groups provided by the input file, including the reference group: $num_conds.\n\n";
    if($data_is_logged == 1) {
	$headerstuff = $headerstuff . "Input data is logged.\n";
    }
    else {
	$headerstuff = $headerstuff . "Input data is unlogged.\n";
    }
    if($use_logged_data == 1) {
	$headerstuff = $headerstuff . "Applying algorithm to logged data.\n";
    }
    else {
	$headerstuff = $headerstuff . "Applying algorithm to unlogged data.\n";
    }
    if($paired == 1) {
	$headerstuff = $headerstuff . "data is paired\n";
    }
    if($shift > 0) {
	$headerstuff = $headerstuff . "Shift applied to the intensities: $shift.\n\n";
    }
    if($tstat == 1) {
	$headerstuff = $headerstuff . "Stat: t-statistic.\n";
    }
    if($means == 1) {
	$headerstuff = $headerstuff . "Stat: ratio of means.\n";
    }
   if($medians == 1) {
	$headerstuff = $headerstuff . "Stat: ratio of medians.\n";
    }
    for (my $i=0; $i<$num_conds; $i++) {
	$headerstuff = $headerstuff . "Number of replicates for group $i: $num_replicates[$i].\n";
    }
    $headerstuff = $headerstuff . "\n";
    for (my $i=0; $i<$num_conds; $i++) {
	$headerstuff = $headerstuff . "Min presence required for group $i: $min_presence_array[$i].\n";
    }
    $headerstuff = $headerstuff . "\n";
    for(my $i=$start; $i<$num_conds; $i++) {
	$headerstuff = $headerstuff . "Level confidence for group $i: $level_confidence[$i].\n";
    }
    $headerstuff = $headerstuff . "\n";
#    $headerstuff = $headerstuff . "IF THE INFORMATION ABOVE DOES NOT CORRESPOND TO WHAT YOU EXPECT, PLEASE CHECK:\na. The format of your input file.\nb. The arguments passed into the program.\n\n";
    for (my $i=$start; $i<$num_conds; $i++) {
	if($tstat == 1) {
	    $headerstuff = $headerstuff . "t-stat tuning parameter for group $i: up: $alpha_up[$i], down: $alpha_down[$i]\n";
	}
    }
    $headerstuff = $headerstuff . "\n";
    for (my $i=$start; $i<$num_conds; $i++) {
	$headerstuff = $headerstuff . "Statistic range for group $i: [$min[$i],$max[$i]].\n";
    }
    $headerstuff = $headerstuff . "\n";
	for (my $i=$start; $i<$num_conds; $i++) {
	$headerstuff = $headerstuff . "Lower cutratio for group $i: $cutratios_down[$i].\n";
	$headerstuff = $headerstuff . "Upper cutratio for group $i: $cutratios_up[$i].\n";
    }
   $headerstuff = $headerstuff . "\n";
#    for (my $i=$start; $i<$num_conds; $i++) {
#	my @cutofflist = @{$cutoffs[$i]};
#	$headerstuff = $headerstuff . "List of level cutoffs for group $i:\n";
#	for (my $j=0; $j<@cutofflist-1; $j++) {
#	    $headerstuff = $headerstuff . "$cutofflist[$j]\n";
#	}
#	$headerstuff = $headerstuff . "$cutofflist[@cutofflist-1]\n\n";
#    }

    $ofh->print($headerstuff);
    $ofh->print($breakdown);
    if($output_text) {
	print TEXTOUT "$headerstuff\n";
	print TEXTOUT "$breakdown\n";
    }

    $ofh->print("</PRE>\n");
    for (my $i=$start; $i<$num_conds; $i++) {
	my $level;
	if ($num_neg_levels[$i] == 0) {
	    $level = 0;
	}
	else {
	    $level = -$num_neg_levels[$i];
	}
	$ofh->print("Levels for group $i:&nbsp&nbsp<b> ");
	if($num_conds>1+$start) {
	    for (my $j=0; $j<$num_levels[$i]-1; $j++) {
		$ofh->print("$level, ");
		$level ++;
	    }
	    $ofh->print("$level</b>\n<BR>\n");
	}
	else {
	    for (my $j=0; $j<$num_levels[$i]-1; $j++) {
		if($level != 0) {
		    my $flag=0;
		    my $nnn = @pattern_list;
		    for(my $a=0; $a<$nnn; $a++) {
			if($level == $pattern_list[$a]) {
			    $flag=1;
			}
		    }
		    if($flag==1) {
			$ofh->print("<a href=\"#$level\"><u>$level</u></a>, ");
		    }
		    else {
			$ofh->print("$level, ");
		    }
		}
		else {
		    $ofh->print("$level, ");
		}
		$level ++;
	    }
	    if($level != 0) {
		$ofh->print("<a href=\"#$level\"><u>$level</u></a></b>\n<BR>&nbsp<br>\n");
	    }
	    else {
		$ofh->print("$level</b>\n<BR>&nbsp<br>\n");
	    }
	}
    }
    $ofh->print("</b>\n");
    if($num_conds > 1+$start) {
	$ofh->print("<center><table cellpadding=3>\n<tr><td><center>Patterns</td></tr>\n");
	print TEXTOUT "Patterns:\n";
	my $nnn = @pattern_list;
	for(my $ppp=0; $ppp<$nnn; $ppp++) {
	    my $linker = $pattern_list[$ppp];
	    $linker =~ s/,/_/g;
	    $ofh->print("<tr><td><a href=\"#$linker\"><u>$pattern_list[$ppp]</u></a></td></tr>\n");
	    print TEXTOUT "$pattern_list[$ppp]\n";
	}
	$ofh->print("</table>\n");
	$ofh->print("\n<BR>\n</center>\n<HR>\n\n<BR>\n");
	print TEXTOUT "\n";
    }
    for (my $i=$start; $i<$num_conds; $i++) {
	if($num_up[$i] == 1 && $num_down[$i] != 1) {
	    $ofh->print("For group $i there is $num_up[$i] up symbol and $num_down[$i] down symbols.\n<BR>\n");
	    if($output_text) {
		print TEXTOUT "For group $i there is $num_up[$i] up symbol and $num_down[$i] down symbols.\n";
	    }
	}
	if($num_up[$i] != 1 && $num_down[$i] == 1) {
	    $ofh->print("For group $i there are $num_up[$i] up symbols and $num_down[$i] down symbol.\n<BR>\n");
	    if($output_text) {
		print TEXTOUT "For group $i there are $num_up[$i] up symbols and $num_down[$i] down symbol.\n";
	    }
	}
	if($num_up[$i] != 1 && $num_down[$i] != 1) {
	    $ofh->print("For group $i there are $num_up[$i] up symbols and $num_down[$i] down symbols.\n<BR>\n");
	    if($output_text) {
	    print TEXTOUT "For group $i there are $num_up[$i] up symbols and $num_down[$i] down symbols.\n";
	    }
	}
    }

    $ofh->print("&nbsp<br>\n");

    my $n = @pattern_list;
    my $aa;
    my $bb;
    if($num_conds==$start+1) {
	my $c=0;
	my @pos;
	my @negs;
	while($pattern_list[$c]>0) {
	    $pos[$c]=$pattern_list[$c];
	    $c++;
	}
	for(my $i=$c;$i<$n;$i++) {
	    $negs[$i-$c]=$pattern_list[$i];
	}
	$aa=@negs;
	$bb=@pos;
    }
    $ofh->print("<TABLE WIDTH=100% BORDER=1 cellpadding=2>\n");
    for (my $i=0; $i<$n; $i++) {
#	print "pattern_list[$i]=$pattern_list[$i]\n";
	if($num_conds==($start+1)) {
	    if(($bb>0) && ($i==0)) {
		$ofh->print("<tr><td colspan=6 bgcolor=lightblue>&nbsp;<br><center><font size=+2><font color=red><b>Upregulation<br>&nbsp;</td></tr>");
	    }
	    if(($aa>0) && ($i==$bb)) {
		$ofh->print("<tr><td colspan=6 bgcolor=lightblue>&nbsp;<br><center><font size=+2><font color=red><b>Downregulation<br>&nbsp;</td></tr>");
	    }
	}

	my $linker = $pattern_list[$i];
	$linker =~ s/,/_/g;
	$ofh->print("<TR><TD colspan=6 bgcolor=lightgreen><a name=\"$linker\">&nbsp;<BR>");

	my $m = @{$clusters{$pattern_list[$i]}};
	if($num_conds>1+$start) {
	    $ofh->print("&nbsp;<a name=\"$linker\">Pattern $pattern_list[$i]");
	    if($output_text) {
		print TEXTOUT "\nPATTERN: $pattern_list[$i]\n";
	    }
	}
	else {
	    $ofh->print("&nbsp;<a name=\"$pattern_list[$i]\">Level $pattern_list[$i]");
	    if($output_text) {
		print TEXTOUT "\nLEVEL: $pattern_list[$i]\n";
	    }
	}

	if ($m>1 && $m<$aux_page_size) {
	    $ofh->print(" has been attached to the following $m gene tags:\n");
	}
	elsif ($m>$aux_page_size) {
	    $ofh->print(" has been attached to $m gene tags.\n");
	}
	else {
	    $ofh->print(" has been attached to the following gene tag:\n");
	}
	$ofh->print("<BR>&nbsp<BR></TD></TR>");
	if ($m>$aux_page_size) {
	    my $string1 = $outfile;
	    $string1 =~ s/.html$//;
	    $string1 =~ s/.htm$//;
	    $string1 =~ s/.HTML$//;
	    $string1 =~ s/.HTM$//;
	    my $string2 = "_".$pattern_list[$i];
	    $string2 =~ s/,//g;
	    $string1 .= $string2.".html";
	    $string1 =~ s/^.*\///;

	    $ofh->print("<TR><TD colspan=6>&nbsp<BR>");
	    $ofh->print("<A HREF=\"$string1\">");
	    $ofh->print("<u>Click here for this list of tags</u></A><BR>&nbsp</TD>\n");
	    $ofh->print("<TD></TD><TR>\n");

	    my $ofh_aux = new IO::File;
	    unless ($ofh_aux->open(">$string1")) {
		die "Cannot open file $string1 for writing.\n";
	    }
	    $ofh_aux->print("<HTML><BODY bgcolor=white>\n");
	    if($num_conds>1+$start) {
		$ofh_aux->print("Pattern $pattern_list[$i] has been attached to the following $m gene tags.<br>\n");
	    }
	    else {
		$ofh_aux->print("Level $pattern_list[$i] has been attached to the following $m gene tags.<br>\n");
	    }
	    $ofh_aux->print("<TABLE WIDTH=100% BORDER=1 cellpadding=2>\n");
	    if($num_conds == 1+$start) {
		    $ofh_aux->print("\n<tr bgcolor=beige><td><center>tag</center></td><td><center>confidence that this gene is differentially expressed</center></td><td><center>means</td><td><center>stat</td><td><center>tag information</center></td><td><center>intensities</td></tr>\n");
		    print TEXTOUT "tag\tconfidence\tlevel\tmeans\tstat\tinformation\n";
	    }
	    else {
		    $ofh_aux->print("\n<tr bgcolor=beige><td><center>tag</center></td><td><center>confidences that this gene is differentially expressed in the respective groups</center></td><td><center>means</td><td><center>stats</td><td><center>tag information</center></td><td><center>intensities</td></tr>\n");
		    print TEXTOUT "tag\tconfidence\tpattern\tmeans\tstat\tinformation\n";
	    }

#===========aux page============= sort the genes in the cluster by confidence level and statistic
	    for (my $j=0; $j<$m; $j++) {
		my $ttttt = $clusters{$pattern_list[$i]}[$j];
		my $sss="";
		for(my $b=$start; $b<$num_conds; $b++) {
		    my $vvvvv = $confidence_levels_hash{$ttttt}[$b];
		    if(!($vvvvv =~ /\S/)) {
			$vvvvv = 0;
		    }
		    $sss = $sss . "$vvvvv, ";
		}
		$sss =~ s/, $//;
		$print_cluster[$j][1]=$ttttt;
		$print_cluster[$j][2]=$sss;
		$sss = "";
		for(my $b=$start; $b<$num_conds; $b++) {
		    my $st = $unpermuted_stat[$ids_hash{$ttttt}][$b];
		    $sss = $sss . "$st, ";
		}
		$sss =~ s/, $//;
		$print_cluster[$j][3]=int($sss*1000)/1000;
	    }
	    my $flag_s=1;
	    while($flag_s==1) {
		$flag_s=0;
		for(my $j=0; $j<$m-1; $j++)
		{
		    my $a1=$print_cluster[$j][2];
		    my $a2=$print_cluster[$j+1][2];
		    my $b1=$print_cluster[$j][3];
		    my $b2=$print_cluster[$j+1][3];
		    if(($a1<$a2) || (($a1==$a2) && ($b1<$b2)))
		    {
			my $holder = $print_cluster[$j][1];
			$print_cluster[$j][1] = $print_cluster[$j+1][1];
			$print_cluster[$j+1][1] = $holder;
			$holder = $print_cluster[$j][2];
			$print_cluster[$j][2] = $print_cluster[$j+1][2];
			$print_cluster[$j+1][2] = $holder;

			$holder = $print_cluster[$j][3];
			$print_cluster[$j][3] = $print_cluster[$j+1][3];
			$print_cluster[$j+1][3] = $holder;

			$flag_s=1;
		    }
		}
	    }

# ============aux page================= print the gene tag and link if id2info defined

	    for(my $j=0; $j<$m; $j++) {
		if($output_text) {
		    my $thing=$print_cluster[$j][1];
		    print TEXTOUT "$thing\t";
		}
		$ofh_aux->print("<TR><TD>\n");
		if (defined $id2url{$print_cluster[$j][1]}) {
		    my $x = $id2url{$print_cluster[$j][1]};
		    $ofh_aux->print("<A HREF=\"$x\">");
		    $ofh_aux->print("<u>$print_cluster[$j][1]</u></A>\n");
		}
		else {
		    $ofh_aux->print("$print_cluster[$j][1]\n");
		}
		$ofh_aux->print("</TD>");
# ============aux page================ print the confidences
		if ($print_cluster[$j][2] ne "") {
		    $ofh_aux->print("<TD>");
		    $ofh_aux->print("$print_cluster[$j][2]</TD>");
		    if($output_text) {
			my $thing=$print_cluster[$j][2];
			print TEXTOUT "$thing\t$pattern_list[$i]\t";
		    }
		}
		else {
		    $ofh_aux->print("&nbsp;</td>\n");
		}


# ===========aux page================= print the means
		my $sss="";
		for(my $b=0; $b<$num_conds; $b++) {
		    my $vvvvv = $unpermuted_means[$ids_hash{$print_cluster[$j][1]}][$b];
		    $vvvvv = int(10000*$vvvvv)/10000;
                    $sss = $sss . "$vvvvv, ";
		}
		$sss =~ s/, $//;
		$ofh_aux->print("<td>$sss</td>\n");
		if($output_text) {
		    print TEXTOUT "$sss\t";
		}
# ===========aux page=================== print the stat

		$sss="";
		for(my $b=$start; $b<$num_conds; $b++) {
		    my $vvvvv = $unpermuted_stat[$ids_hash{$print_cluster[$j][1]}][$b];
		    $vvvvv = int(10000*$vvvvv)/10000;
                    $sss = $sss . "$vvvvv, ";
		}
		$sss =~ s/, $//;
		$ofh_aux->print("<td>$sss</td>\n");
		if($output_text) {
		    print TEXTOUT "$sss\t";
		}
# ===========aux page================== print the gene info


		$ofh_aux->print("<TD>");

		if (defined $id2info{$print_cluster[$j][1]}) {
		    my $x = $id2info{$print_cluster[$j][1]};
		    $x =~ s/\s+/ /gs;
		    $ofh_aux->print("$x");
		    if($output_text) {
			print TEXTOUT "$x";
		    }
		}
		$ofh_aux->print("&nbsp</TD>\n");

		print TEXTOUT "\n";
		if($intensity_counter>$intensity_page_limit) {
		    $ofh->print("</BODY>\n</HTML>\n");
		    $ofh_intensities->print("&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;\n");
		    $ofh_intensities->print("&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br></BODY>\n</HTML>\n");
		    close($ofh_intensities);
		    my $n=$intensity_page_counter;
		    $intensity_page_counter++;
		    $int_outfile =~ s/intensities$n/intensities$intensity_page_counter/;

		    unless ($ofh_intensities->open(">$int_outfile")) {
			die "Cannot open file $int_outfile for writing.\n";
		    }
		    $ofh_intensities->print("<html><body bgcolor=white>\n<title>intensities for $outfile</title>\n");
		    $intensity_counter=0;
		}

		my $id=$print_cluster[$j][1];
		$id=~s/\s//g;
		my $id_nowhite = $id;
		my $id2=$print_cluster[$j][1];
		$ofh_aux->print("<TD><center><a href=\"$int_outfile#$id\" title=\"click here for the intensities for tag $id2\">I</a></td></tr>\n");
		$id=$print_cluster[$j][1];
		my $wid=2*$num_conds;
		$ofh_intensities->print("<a name=$id_nowhite><br>\n<table border=1 cellpadding=2>\n<tr><td bgcolor=beige colspan=$wid><center>tag: $ids_hash{$id}</td></tr>\n<tr>");
		for(my $group=0; $group<$num_conds; $group++) {
		    $ofh_intensities->print("<td>group $group</td>");
		}
		$ofh_intensities->print("</tr><tr>");
		for(my $group=0; $group<$num_conds; $group++) {
		    $ofh_intensities->print("<td>");
		    my $l=$num_replicates[$group];
		    undef $intensity_storage;
		    my $count=0;
		    for(my $i=1; $i<$l+1; $i++) {
			$intensity_counter++;
			if(defined $data[$ids_hash{$id}][$group][$i]) {
			    my $x=$data[$ids_hash{$id}][$group][$i];
			    $ofh_intensities->print("file $i: $x<br>\n");
			    $intensity_storage->[$count]=$x;
			    $count++;
			}
			else {
			    if(defined $outliers{$id}[$group][$i]) {
				my $x=$outliers{$id}[$group][$i];
				$ofh_intensities->print("file $i: <font color=red>$x  *** ELIMINATED AS OUTLIER ***<font color=black><br>\n");
			    }
			    else {
				$ofh_intensities->print("file $i:<br>\n");
			    }
			}
		    }
		    my $mean = $unpermuted_means[$ids_hash{$id}][$group];
		    $ofh_intensities->print("&nbsp;<br>group $group mean=$mean</td>");
		}
		$ofh_intensities->print("\n</tr></table>\n<br>&nbsp;<br>\n");

	    }
	    $ofh_aux->print("</TABLE></BODY></HTML>\n");
	    $ofh_aux->close();
	}
	else {
	    if($num_conds == 1 + $start) {
		if($pattern_list[$i]>0) {
		    $ofh->print("\n<tr bgcolor=beige><td><center>tag</center></td><td><center>confidence that this gene is upregulated</center></td><td><center>means</td><td><center>statistic</td><td><center>tag information</center></td><td><center>intensities</td></tr>\n");
		    if($output_text) {
			print TEXTOUT "tag\tconfidence\tlevel\tmeans\tstat\tinformation\n";
		    }
		}
		else {
		    $ofh->print("\n<tr bgcolor=beige><td><center>tag</center></td><td><center>confidence that this gene is downregulated</center></td><td><center>means</td><td><center>statistic</td><td><center>tag information</center></td><td><center>intensities</td></tr>\n");
		    if($output_text) {
			print TEXTOUT "tag\tconfidence\tlevel\tmeans\tstat\tinformation\n";
		    }
		}
	    }
	    else {
		$ofh->print("\n<tr bgcolor=beige><td><center>tag</center></td><td><center>confidences that this gene is differentially expressed in the respective groups</center></td><td><center>means</td><td><center>statistics</td><td><center>tag information</center></td><td><center>intensities</td></tr>\n");
		if($output_text) {
		    print TEXTOUT "tag\tconfidence\tpattern\tmeans\tstat\tinformation\n";
		}
	    }

#======================== sort the genes in the cluster by confidence level and t-statistic
# DEBUG
# foreach my $id (keys %confidence_levels_hash) {
#    for(my $cond=$start; $cond<$num_conds; $cond++) {
#	my $x = $confidence_levels_hash{$id}[$cond];
#	print "confidence_levels_hash{$id}[$cond] = $x\n";
#    }
# }
# DEBUG
	    for (my $j=0; $j<$m; $j++) {
		my $ttttt = $clusters{$pattern_list[$i]}[$j];
		my $sss="";
		for(my $b=$start; $b<$num_conds; $b++) {
		    my $vvvvv = $confidence_levels_hash{$ttttt}[$b];
		    if(!($vvvvv =~ /\S/)) {
			$vvvvv = 0;
		    }
		    $sss = $sss . "$vvvvv, ";
		}
		$sss =~ s/, $//;
		$print_cluster[$j][1]=$ttttt;
		$print_cluster[$j][2]=$sss;
		$sss = "";
		for(my $b=$start; $b<$num_conds; $b++) {
		    my $st = $unpermuted_stat[$ids_hash{$ttttt}][$b];
		    $sss = $sss . "$st, ";
		}
		$sss =~ s/, $//;
		$print_cluster[$j][3]=int($sss*1000)/1000;
	    }
	    my $flag_s=1;
	    while($flag_s==1) {
		$flag_s=0;
		for(my $j=0; $j<$m-1; $j++)
		{
		    my $a1=$print_cluster[$j][2];
		    my $a2=$print_cluster[$j+1][2];
		    my $b1=$print_cluster[$j][3];
		    my $b2=$print_cluster[$j+1][3];
		    if(($a1<$a2) || (($a1==$a2) && ($b1<$b2)))
		    {
			my $holder = $print_cluster[$j][1];
			$print_cluster[$j][1] = $print_cluster[$j+1][1];
			$print_cluster[$j+1][1] = $holder;
			$holder = $print_cluster[$j][2];
			$print_cluster[$j][2] = $print_cluster[$j+1][2];
			$print_cluster[$j+1][2] = $holder;

			$holder = $print_cluster[$j][3];
			$print_cluster[$j][3] = $print_cluster[$j+1][3];
			$print_cluster[$j+1][3] = $holder;

			$flag_s=1;
		    }
		}
	    }

# =================================== print the gene tag and link if id2info defined

	    for(my $j=0; $j<$m; $j++) {
		$ofh->print("<TR><TD>\n");
		if($output_text) {
		    my $thing=$print_cluster[$j][1];
		    print TEXTOUT "$thing\t";
		}
		if (defined $id2url{$print_cluster[$j][1]}) {
		    my $x = $id2url{$print_cluster[$j][1]};
		    $ofh->print("<A HREF=\"$x\">");
		    $ofh->print("<u>$print_cluster[$j][1]</u></A>\n");
		}
		else {
		    $ofh->print("$print_cluster[$j][1]\n");
		}
		$ofh->print("</TD>");

# =================================== print the confidences

		if ($print_cluster[$j][2] ne "") {
		    $ofh->print("<TD>");
		    $ofh->print("$print_cluster[$j][2]</TD>");
		    if($output_text) {
			my $thing=$print_cluster[$j][2];
			print TEXTOUT "$thing\t$pattern_list[$i]\t";
		    }
		}
		else {
		    $ofh->print("&nbsp;</td>\n");
		}

# =================================== print the means
		my $sss="";
		for(my $b=0; $b<$num_conds; $b++) {
		    my $vvvvv = $unpermuted_means[$ids_hash{$print_cluster[$j][1]}][$b];
		    $vvvvv = int(10000*$vvvvv)/10000;
                    $sss = $sss . "$vvvvv, ";
		}
		$sss =~ s/, $//;
		$ofh->print("<td>$sss</td>\n");
		if($output_text) {
		    print TEXTOUT "$sss\t";
		}
# =================================== print the stat

		$sss="";
		for(my $b=$start; $b<$num_conds; $b++) {
		    my $vvvvv = $unpermuted_stat[$ids_hash{$print_cluster[$j][1]}][$b];
		    $vvvvv = int(10000*$vvvvv)/10000;
                    $sss = $sss . "$vvvvv, ";
		}
		$sss =~ s/, $//;
		$ofh->print("<td>$sss</td>\n");
		if($output_text) {
		    print TEXTOUT "$sss\t";
		}
# =================================== print the gene info

		$ofh->print("<TD>");

		if (defined $id2info{$print_cluster[$j][1]}) {
		    my $x = $id2info{$print_cluster[$j][1]};
		    $x =~ s/\s+/ /gs;
		    $ofh->print("$x");
		    if($output_text) {
			print TEXTOUT "$x";
		    }
		}
		$ofh->print("&nbsp</TD>\n");

		print TEXTOUT "\n";

		if($intensity_counter>$intensity_page_limit) {
		    $ofh_intensities->print("&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;\n");
		    $ofh_intensities->print("&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br></BODY>\n</HTML>\n");
		    close($ofh_intensities);
		    my $n=$intensity_page_counter;
		    $intensity_page_counter++;
		    $int_outfile =~ s/intensities$n/intensities$intensity_page_counter/;

		    unless ($ofh_intensities->open(">$int_outfile")) {
			die "Cannot open file $int_outfile for writing.\n";
		    }
		    $ofh_intensities->print("<html><body bgcolor=white>\n<title>intensities for $outfile</title>\n");
		    $intensity_counter=0;
		}

		my $id=$print_cluster[$j][1];
		$id=~s/\s//g;
		my $id_nowhite = $id;
		my $id2=$print_cluster[$j][1];
		$ofh->print("<TD><center><a href=\"$int_outfile#$id\" title=\"click here for the intensities for tag $id2\">I</a></td></tr>\n");
		$id=$print_cluster[$j][1];
		my $wid=2*$num_conds;

		$ofh_intensities->print("<a name=$id_nowhite><br>\n<table border=1 cellpadding=2>\n<tr><td bgcolor=beige colspan=$wid><center>tag: $id</td></tr>\n<tr>");
		for(my $group=0; $group<$num_conds; $group++) {
		    $ofh_intensities->print("<td>group $group</td>");
		}
		$ofh_intensities->print("</tr><tr>");
		for(my $group=0; $group<$num_conds; $group++) {
		    $ofh_intensities->print("<td>");
		    my $l=$num_replicates[$group];
		    undef $intensity_storage;
		    my $count=0;
		    for(my $i=1; $i<$l+1; $i++) {
			$intensity_counter++;
			if(defined $data[$ids_hash{$id}][$group][$i]) {
			    my $x=$data[$ids_hash{$id}][$group][$i];
			    $ofh_intensities->print("file $i: $x<br>\n");
			    $intensity_storage->[$count]=$x;
			    $count++;
			}
			else {
			    if(defined $outliers{$id}[$group][$i]) {
				my $x=$outliers{$id}[$group][$i];
				$ofh_intensities->print("file $i: <font color=red>$x  *** ELIMINATED AS OUTLIER ***<font color=black><br>\n");
			    }
			    else {
				$ofh_intensities->print("file $i:<br>\n");
			    }
			}
		    }
		    my $mean = $unpermuted_means[$ids_hash{$id}][$group];
		    $ofh_intensities->print("&nbsp;<br>group $group mean=$mean</td>");
		}
		$ofh_intensities->print("\n</tr></table>\n<br>&nbsp;<br>\n");
	    }
	}
    }

    $ofh_intensities->print("&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;\n");
    $ofh_intensities->print("&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br></BODY>\n</HTML>\n");

    $ofh->print("</table>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>\n</BODY>\n</HTML>\n");

    $ofh->close();
    $ofh_intensities->close();

    if($output_text) {
	close(TEXTOUT);
    }
    return $headerstuff;
}

sub ReadId2Info {
    my %output;
    my $ifh = new IO::File;
    unless ($ifh->open("<$id2info")) {
	die "Cannot open file $id2info for reading.\n";
    }
    my $count=0;
    while (my $line=<$ifh>) {
	$count++;
	chomp($line);
	$line =~ s/^\s+//g;
	if ($line eq "") {
	    next;
	}
	if (!($line =~ /\t/)) {
	    die "\nError: id2info file is not tab delimited, check line $count.\n\n";
	}
	my $tag;
	my $info;
	($tag, $info) = split(/\t+/, $line);
	$output{$tag} = $info;
    }
    $ifh->close();

    return \%output;
}

sub ReadId2Url {
    my %output;
    my $ifh = new IO::File;
    unless ($ifh->open("<$id2url")) {
	die "Cannot open file $id2url for reading.\n";
    }
    my $count=0;
    while (my $line=<$ifh>) {
	$count++;
	chomp($line);
	if ($line =~ /\t[\S]+\t/) {
	    die "\nError: line $count of the id2url file has more than one tab.\n\n";
	}
	if (!($line =~ /\t/)) {
	    die "\nError: id2url file is not tab delimited, check line $count.\n\n";
	}
	$line =~ s/^\s+$//;
	if(!($line =~ /\S/)) {
	    next;
	}
	my $tag;
	my $url;
	($tag, $url) = split(/\t+/, $line);
	if($url =~ /^[^\"]*\"[^\"]*$/) {
	    die "\nError: line $count of the id2url file has a single double quote,\nthere should be either zero quotes in a URL, or two quotes in a link,\nbut I cannot make sense of one quote.\nPlease fix your id2url file.\n\n";
	}
	if($url =~ /<a\s+href=\"?(http:[^\"\s]*)\"?/i) {
	    $url = $1;
	}
	$output{$tag} = $url;
    }
    $ifh->close();

    return \%output;
}

sub WriteHelp {
    print "\n\n-------------------------------------------------------------------------------";
    print "\n|                                PaGE Help                                     |\n";
    print "-------------------------------------------------------------------------------\n";
print "Note: it is not necessary to give any command line options, those not\nspecified command line will be requested by the program during execution,\nwith the exception of the (optional) commands which give the names of\nfiles which map id's to descriptions and url which must be input command\nline.  For the exact usage of these commands see below.\n-------------------------------------------------------------------------------\n";
    print "Options are specified using --option followed by the option value, if any\n(do not use an = sign).  For example:\n\n> PaGE_5.1.pl --infile input.txt --level_confidence .8\n--min_presence_list 3,5 --data_is_logged --note \"experiment 15\"\n-------------------------------------------------------------------------------\n";
    print "Data File format: A tab delimited table.\n\n";
    print "The first column gives the unique id.\n\n";
    print "There may be any number of comment rows at the top of the file, which\nmust start with a #\n";
	print "\nThe first non-comment row is the header row, with the following format:\n";
    print "For condition i and replicate j the header is \"cirj\".  For example with three\ngroups consisting of 3, 3, and 2 replicates, the header would be\n\nid\tc0r1\tc0r2\tc0r3\tc1r1\tc1r2\tc1r3\tc2r1\tc2r2.\n\n\"c\" stands for \"condition\"\n\"r\" stands for \"replicate\".\n\nConditions start counting at zero, replicates start counting at one.\n\nIf there is only a single direct comparision then there will only one condition\nnumbered zero.\n\nColumns can be labeled in any order.  To ignore a column of data leave\nthe header blank, or put an \"i\" (for \"ignore\")\n-------------------------------------------------------------------------------\n";
    print "|                                PaGE Commands                                 |\n-------------------------------------------------------------------------------\n";
    print "--help\n    If set, will show this help page.\n";
    print "--usage\n    Synonym for help.\n";
    print "------------------\n";
    print "| File locations |\n";
    print "------------------\n";
    print "--infile\n    Name of the input file containing the table of data.\n    This file must conform to the format in the README file.\n";
    print "--outfile\n    Optional. Name of the output file, if not specified outfile name will be\n    derived from the infile name.\n";
    print "--id2info\n    Optional. Name of the file containing a mapping of gene id's to names\n    or descriptions.\n";
    print "--id2url\n    Optional. Name ot the file containing a mapping of gene id's to urls.\n";
    print "--id_filter_file\n    If you just want to run the algorithm on a subset of the genes in your\n    data file, you can put the id's for those genes in a file, one per line,\n    and specify that file with this option.\n";
    print "------------------\n";
    print "| Output Options |\n";
    print "------------------\n";
    print "--output_gene_confidence_list\n    Optional.  Set this to output a tab delimited file that maps every gene to\n    its confidence of differential expression.  For each comparison gives\n    separate lists for up and down regulation.\n";
    print "--output_gene_confidence_list_combined\n    Optional.  Set this to output a tab delimited file that maps every gene to\n    its confidence of differential expression.  For each comparison gives one\n    list with up and down regulation combined.\n";
    print "--output_text\n    Optional.  Set this to output the results also in text format.\n";
    print "--note\n    Optional. A string that will be included at the top of the output file.\n";
    print "--aux_page_size\n    Optional.  A whole number greater than zero.  This specifies the minimum\n    number of tags there can be in one pattern before the results for that\n    pattern are written to an auxiliary page (this keeps the main results page\n    from getting too large).  This argument is optional, the default is 500.\n";
    print "---------------------------------------------\n";
    print "| Study Design and Nature of the Input Data |\n";
    print "---------------------------------------------\n";
    print "--num_channels\n    Is your data one or two channels?  (note: Affymetrix is considered one\n    channel).\n";
    print "--design\n    For two channel data, either set this to \"R\" for \"reference\" design,\n    or \"D\" for \"direct comparisons\" (see the documentation for more\n    information on this setting).\n";
    print "--data_is_logged\n    Use this option if your data has already been log transformed.\n";
    print "--data_not_logged\n    Use this option if your data has not been log transformed.\n";
    print "--paired\n    The data is paired.\n";
    print "--unpaired\n    The data is not paired.\n";
    print "--missing_value\n    If you have missing values designated by a string (such as \"NA\"), specify\n    that string with this option.  You can either put quotes around the string\n    or not, it doesn't matter as long as the string has no spaces.\n";
    print "-------------------------------------\n";
    print "| Statistics and Parameter Settings |\n";
    print "-------------------------------------\n";
    print "--level_confidence\n    A number between 0 and 1.  Generate the levels with this confidence.\n    See the README file for more information on this parameter.  This can\n    be set separately for each group using --level_confidence_list (see\n    below)\n    NOTE: This parameter can be set at the end of the run after the program has\n    displayed a summary breakdown of how many genes are found with what\n    confidence.  To do this either set the command line option to \"L\" (for\n    \"later\"), or do not specify this command line option and enter \"L\" when\n    the program prompts for the level confidence\n";
    print "--level_confidence_list\n    Comma-separated list of confidences.  If there are more than two\n    conditions (or more than one direct comparision), each position in the\n    pattern can have its own confidence specified by this list.  E.g. if\n    there are 4 conditions, the list might be .8,.7,.9 (note four conditions\n    gives patterns of length 3)\n";
    print "--min_presence\n    A positive integer specifying the minimum number of values a tag should\n    have in each condition in order to not be discarded.  This can be set\n    separately for each condition using --min_presence_list\n";
    print "--min_presence_list\n    Comma-separated list of positive integers, one for each condition,\n    specifying the minimum number of values a tag should have, for each\n    condition, in order not to be discarded.  E.g. if there are three\n    conditions, the list might be 4,6,3\n";
    print "--use_logged_data\n    Use this option to run the algorithm on the logged data (you can only\n    use this option if using the t-statistic as statistic).  Logging the\n    data usually give better results, but there is no rule.  Sometimes\n    different genes can be picked up either way.  It is generally best,\n    if using the t-statistic, to go with the logged data.  You might try\n    both ways and see if it makes much difference.  Both ways give valid\n    results, what can be effected is the power.\n";
    print "--use_unlogged_data\n    Use this option to run the algorithm on the unlogged data.  (See\n    --use_loggged_data option above for more information.)\n";
    print "--tstat\n    Use the t-statistic as statistic.\n";
    print "--means\n    Use the ratio of the means of the two groups as statistic.\n";
#    print "--medians\n    Use the ratio of the medians of the two groups as statistic.\n";
    print "--tstat_tuning_parameter\n    Optional.  The value of the t-statistic tuning parameter.  This is set to\n    a default value determined separately for each pattern position, but can be\n    set by hand using this command.  See the documentation for more\n    information on this parameter.\n";
    print "--shift\n    Optional.  A real number greater than zero.  This number will be added to\n    all intensities (of the unlogged data).  See the documentation for more on\n    why you might use this parameter.\n";
    print "-----------------\n";
    print "| Configuration |\n";
    print "-----------------\n";
    print "--silent_mode\n    Optional. Do not output warning messages or progress to screen.\n";
    print "--num_permutations\n    Optional.  The number of permutations to use.  The default is to use all\n    or 200, whichever is smaller.  You might want to lower it to increase the\n    speed, though at a possible loss power or accuracy.  If the total number of\n    possible permutations is less than 25 more than the number requested by this\n    command, then the program will use the total number.\n";

    print "--num_bins\n    Optional.  The number of bins to use in granularizing the statistic over\n    its range.  This is set to a default of 1000 and you probably shouldn't\n    need to change it.\n";

    exit();
}

# --help
# --usage
# --infile
# --outfile
# --id2info
# --id2url
# --silent_mode
# --min_presence
# --min_presence_list
# --shift
# --level_confidence
# --level_confidence_list
# --aux_page_size
# --no_outlier_filter
# --keep_outliers
# --outlier_filter_strictness
# --data_is_logged
# --data_not_logged
# --note
# --medians (--median)
# --means (--mean)
# --tstat
# --pvalstat
# --num_permutations (--num_perms)
# --only_report_outliers
# --num_channels
# --paired
# --unpaired
# --avalues
# --noavalues
# --design
# --missing_value
# --id_filter_file
# --no_background_filter
# --background_filter_strictness
# --use_logged_data
# --use_unlogged_data
# --num_bins
# --pool
# --output_gene_confidence_list
# --output_text
# --tstat_tuning_parameter (--tstat_tuning_param)
