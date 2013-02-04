// PaGE Version 5.2 - this class holds the mathematical operators we will needs, such as means,
// t-stats, etc.. of vectors and permutation functions that permute vectors, etc..

import java.lang.reflect.Array;
import java.util.Date;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Comparator;
import java.util.Hashtable;
import java.io.*;
import java.lang.Exception;
import java.lang.NullPointerException;
import java.util.*;
import java.lang.reflect.Array;
import java.util.Date;

public class MathematicalOperations {

    // CONSTRUCTOR
    public MathematicalOperations() {

    }

    // the following computes the mean of a vector, ignoring missing values and returning 'false' if
    // there are fewer values than min_presence

    public Object[] ComputeMean(double[] data, boolean[] missing_values, int min_presence) {
	Object[] returnArray = new Object[2];
	int len = Array.getLength(data);
	double m=0;
	int n=0;
	boolean valid = false;
	for(int i=0; i<len; i++) {
	    if(!missing_values[i]) {
		m=m+data[i];
		n++;
	    }
	}
	if(n >= min_presence) {
	    m=m/n;
	    valid = true;
	}
	else {
	    valid = false;
	}
	returnArray[0] = m;
	returnArray[1] = valid;

	return returnArray;
    }

    // the following initializes the num_perm permutations

    public int[][][] InitializePermuationArray(int num_conds, int[] num_reps, int num_perms, boolean paired, char design) {
	System.out.println("num_conds = " + num_conds);
	int[][][] perm_array = new int[num_conds][0][0];
	if(design == 'D') {
	    for(int i=0; i<num_conds; i++) {
		int size_of_group = num_reps[i];
		int number_subsets = power(2, size_of_group);
		if(number_subsets < num_perms+25) {
		    perm_array[i] = ListAllSubsets(size_of_group);
		}
		else {
		    perm_array[i] = GetRandomSubsets(size_of_group, num_perms);
		}
	    }
	}
	else {
	    for(int i=1; i<num_conds; i++) {
		if(paired == false) {
		    int size_of_group1 = num_reps[0];
		    int size_of_group2 = num_reps[i];
		    int sum = size_of_group1 + size_of_group2;
		    int n=0;
		    if(sum-size_of_group1 > size_of_group1) {
			n = size_of_group1;
		    }
		    else {
			n = sum-size_of_group1;
		    }
		    int m = sum;
		    int num_all_perms=Binomial(m,n);
		    int num_subs = Binomial(sum, size_of_group1);
		    if(num_all_perms < num_perms+25) {
			perm_array[i] = GetAllSubsetsOfFixedSize(sum, size_of_group1);
		    }
		    else {
			perm_array[i] = GetRandomSubsetsOfFixedSize(sum, size_of_group1, num_perms);
		    }
		}
		else {
		    int size_of_group = num_reps[0];
		    int number_subsets = power(2, size_of_group);
		    int[][] temp_array;
		    System.out.println("number_subsets = " + number_subsets);
		    if(number_subsets < num_perms+25) {
			temp_array = ListAllSubsets(size_of_group);
			System.out.println("here 1");
		    }
		    else {
			temp_array = GetRandomSubsets(size_of_group, num_perms);
			System.out.println("here 2");
		    }
		    int n = Array.getLength(temp_array);
		    perm_array = new int[num_conds][n][2*num_reps[0]];
		    for(int p=0; p<n; p++) {
			for(int j=0; j<num_reps[0]; j++) {
			    perm_array[i][p][j] = temp_array[p][j];
			    perm_array[i][p][j+num_reps[0]] = 1-temp_array[p][j];
			    if(p==0) {
				perm_array[i][p][j] = 1;
				perm_array[i][p][j+num_reps[0]] = 0;
			    }
			}
		    }
		}
	    }
	}
	return perm_array;
    }

    // m choose n, or how many subsets of size n can be made of m things

    public int Binomial(int m, int n) {
	double total=1;
	for(int j=0; j<n; j++) {
	    total = total * (m-j)/(n-j);
	}
	return (int)total;
    }

    public int[][] GetAllSubsetsOfFixedSize(int size_of_set, int size_of_subset) {
	int[] counter = new int[size_of_subset+1];
	
	for(int i=1; i<size_of_subset+1; i++) {
	    counter[i]=i;
	}
	boolean flag=false;
	int subset_counter = 0;
	int num_subs = Binomial(size_of_set, size_of_subset);
	int[][] subset_array = new int[num_subs][size_of_set];
	
	for(int i=0; i<num_subs; i++) {
	    for(int j=0; j<size_of_set; j++) {
		subset_array[i][j]=0;
	    }
	}
	
	while(flag==false) {
	    subset_array[subset_counter][counter[1]-1]++;
	    for(int p=2; p<size_of_subset+1; p++) {
		subset_array[subset_counter][counter[p]-1]++;
	    }
	    int j=size_of_set;
	    int jj=size_of_subset;
	    while((counter[jj]==j) && (j>0)) {
		jj--;
		j--;
	    }
	    if(jj==0) {
		flag=true;
	    }
	    if(jj>0) {
		counter[jj]++;
		int k = 1;
		for(int i=jj+1; i<size_of_subset+1; i++) {
		    counter[i]=counter[jj]+k;
		    k++;
		}
	    }
	    subset_counter++;
	}
	return subset_array;
    }
    
    public int[][] GetRandomSubsetsOfFixedSize(int size_of_set, int size_of_subset, int num_subsets) {
	int[][] subset_array = new int[num_subsets][size_of_set];
	int counter=0;
	for(int j=0; j<size_of_subset; j++) {
	    subset_array[0][j]=1;
	}
	for(int j=size_of_subset; j<size_of_set; j++) {
	    subset_array[0][j]=0;
	}
	for(int i=1; i<num_subsets; i++) {
	    int[] subset = ChooseRand(size_of_subset, size_of_set);
	    for(int j=0; j<size_of_set; j++) {
		subset_array[i][j]=0;
	    }
	    int n = Array.getLength(subset);
	    for(int j=0; j<n; j++) {
		subset_array[i][subset[j]-1]++;
	    }
	}
	return subset_array;
    }

    public int[][] GetRandomSubsets(int size_of_set, int num_subsets) {
	int[][] subset_array = new int[num_subsets][size_of_set];

	for(int i=0; i<num_subsets; i++) {
	    for(int j=0; j<size_of_set; j++) {
		subset_array[i][j] = 0;
	    }
	}
	for(int i=1; i<num_subsets; i++) {
	    for(int j=0; j<size_of_set; j++) {
		Random generator = new Random();
		int flip = generator.nextInt(2);
		if(flip == 1) {
		    subset_array[i][j]++;
		}
	    }
	}
	return subset_array;
    }

    public int[][] ListAllSubsets(int size_of_set) {
	int num_subsets = power(2, size_of_set);
	int[][] perm_array = new int[num_subsets][size_of_set];
	for(int i=0; i<num_subsets; i++) {
	    for(int j=0; j<size_of_set; j++) {
		perm_array[i][j] = 0;
	    }
	}
	int[] counter = new int[size_of_set + 1];
	int perm_counter = 0;
	for(int subsetsize=1; subsetsize < size_of_set+1; subsetsize++) {
	    for(int i=1; i<subsetsize+1; i++) {
		counter[i]=i;
	    }
	    boolean flag=false;
	    while(flag==false) {
		perm_array[perm_counter][counter[1]-1]++;
		for(int p=2; p<subsetsize+1; p++) {
		    perm_array[perm_counter][counter[p]-1]++;
		}
		perm_counter++;
		int j=size_of_set;
		int jj=subsetsize;
		while((counter[jj]==j) && (j>0)) {
		    jj--;
		    j--;
		}
		if(jj==0) {
		    flag=true;
		}
		if(jj>0) {
		    counter[jj]++;
		    int k=1;
		    for(int i=jj+1; i<subsetsize+1; i++) {
			counter[i]=counter[jj]+k;
			k++;
		    }
		}
	    }
	}
	
	return perm_array;
    }

    public int[] ChooseRand(int subsetsize, int groupsize) {
	boolean flag = false;
	int x = 0;
	int[] subset = new int[subsetsize];
	for(int i=0; i<subsetsize; i++) {
	    flag=false;
	    while(flag==false) {
		flag=true;
		Random generator = new Random();
		x = generator.nextInt(groupsize) + 1;
		for(int j=0; j<i; j++) {
		    if(x==subset[j]) {
			flag=false;
		    }
		}
	    }
	    subset[i]=x;
	}
	return subset;
    }

    public int power(int base, int exponent) {
	int x = base;
	for(int i=1; i<exponent; i++) {
	    x = x * base;
	}
	return x;
    }

    public void PrintPermutationMatrix(int num_conds, int[][][] permutations, char design, int start) {

	// this subroutine prints the matrix of permutations for DEBUG
	for(int i=start; i<num_conds; i++) {
	    System.out.println("---- condition " + i + " ----");
	    int n = Array.getLength(permutations[i]);
	    for(int j=0; j<n; j++) {
		int m = Array.getLength(permutations[i][j]);
		System.out.print(permutations[i][j][0]);
		for(int k=1; k<m; k++) {
		    System.out.print("," + permutations[i][j][k]);
		}
		System.out.print("\n");
	    }
	}
    }

    public Object[] ComputeDistanceBetweenVectors(double[] v1, double[] v2) {
	boolean valid = true;
	int n1 = Array.getLength(v1);
	int n2 = Array.getLength(v2);
	double dist = 0;
	if(n1 != n2) {
	    valid = false;
	}
	else {
	    for(int i=0; i<n1; i++) {
		dist = dist + (v1[i] - v2[i]) * (v1[i] - v2[i]);
	    }
	}
	Object[] returnVector = new Object[2];
	returnVector[0] = (double)Math.sqrt(dist);
	returnVector[1] = valid;

	return returnVector;
    }

    public double[] FindDefaultAlpha(double[][][] data, double[][][][] data_v, int num_conds, boolean[][][] missingvalues, int[] min_presence_list, char design, int[] num_reps, Parameters parameters) {

	// This chooses a default value of alpha which will be multiplied by a range of values
        // to create a range of t-stat-tuning parameters.  It's basically the scaled mean of
	// the denominator of the normal t-stat - I can't remember why I scaled it and didn't
	// just use the mean...

	boolean paired = parameters.get_paired();
	int num_ids = Array.getLength(data);
	double[] alpha = new double[num_conds];
	double value = 0;
	int start;
	int num=0;
	double mean=0;
	if(!(design == 'D')) {
	    start = 1;
	}
	else {
	    start = 0;
	}
	for(int cond=start; cond<num_conds; cond++) {
	    num=0;
	    mean=0;
	    boolean valid = true;
	    for(int id=0; id<num_ids; id++) {
		if(!(design == 'D')) {
		    if(!parameters.get_vector_analysis()) {
			double[] vector1 = new double[parameters.get_num_reps()[0]];
			double[] vector2 = new double[parameters.get_num_reps()[cond]];
			boolean[] missing1 = new boolean[parameters.get_num_reps()[0]];
			boolean[] missing2 = new boolean[parameters.get_num_reps()[cond]];
			for(int j=1; j<parameters.get_num_reps()[0]+1; j++) {
			    vector1[j-1] = data[id][0][j];
			    missing1[j-1] = missingvalues[id][0][j];
			}
			for(int j=1; j<parameters.get_num_reps()[cond]+1; j++) {
			    vector2[j-1] = data[id][cond][j];
			    missing2[j-1] = missingvalues[id][cond][j];
			}
			if(paired == false) {
			    Object[] ret = ComputeS(vector2, vector1, missing2, missing1, parameters.get_min_presence_list()[cond], parameters.get_min_presence_list()[0]);
			    valid = (Boolean)ret[1];
			    if(valid) {
				value = (Double)ret[0];
			    }
			}
			else {
			    Object[] ret = ComputePairedS(vector2, vector1, missing2, missing1, parameters.get_min_presence_list()[cond], parameters.get_min_presence_list()[0]);
			    valid = (Boolean)ret[1];
			    if(valid) {
				value = (Double)ret[0];
			    }
			}
		    }
		    else {
			double[][] vector1 = new double[parameters.get_num_reps()[0]][parameters.get_vector_length()];
			double[][] vector2 = new double[parameters.get_num_reps()[cond]][parameters.get_vector_length()];
			boolean[] missing1 = new boolean[parameters.get_num_reps()[0]];
			boolean[] missing2 = new boolean[parameters.get_num_reps()[cond]];
			for(int j=1; j<parameters.get_num_reps()[0]+1; j++) {
			    for(int k=0; k<parameters.get_vector_length(); k++) {
				vector1[j-1][k] = data_v[id][0][j][k];
			    }
			    missing1[j-1] = missingvalues[id][0][j];
			}
			for(int j=1; j<parameters.get_num_reps()[cond]+1; j++) {
			    for(int k=0; k<parameters.get_vector_length(); k++) {
				vector2[j-1][k] = data_v[id][cond][j][k];
			    }
			    missing2[j-1] = missingvalues[id][cond][j];
			}
			if(paired == false) {
			    Object[] ret = ComputeS_v(vector2, vector1, missing2, missing1, parameters.get_min_presence_list()[cond], parameters.get_min_presence_list()[0]);
			    valid = (Boolean)ret[1];
			    if(valid) {
				value = (Double)ret[0];
			    }
			}
			else {
			    Object[] ret = ComputePairedS_v(vector2, vector1, missing2, missing1, parameters.get_min_presence_list()[cond], parameters.get_min_presence_list()[0]);
			    valid = (Boolean)ret[1];
			    if(valid) {
				value = (Double)ret[0];
			    }
			}
		    }
		    if(valid) {
			mean = mean + value;
			num++;
		    }
		}
		else {
		    if(!parameters.get_vector_analysis()) {
			double[] vector = new double[parameters.get_num_reps()[cond]];
			boolean[] missing = new boolean[parameters.get_num_reps()[cond]];
			for(int j=1; j<parameters.get_num_reps()[0]+1; j++) {
			    vector[j-1] = data[id][cond][j];
			    missing[j-1] = missingvalues[id][cond][j];
			}
			Object[] ret = ComputeOneSampleS(vector, missing, parameters.get_min_presence_list()[cond]);
			valid = (Boolean)ret[1];
			if(valid) {
			    value = (Double)ret[0];
			    mean = mean + value;
			    num++;
			}
		    }
		    else {
			double[][] vector = new double[parameters.get_num_reps()[cond]][parameters.get_vector_length()];
			boolean[] missing = new boolean[parameters.get_num_reps()[cond]];
			for(int j=1; j<parameters.get_num_reps()[0]+1; j++) {
			    for(int k=0; k<parameters.get_vector_length(); k++) {
				vector[j-1][k] = data_v[id][cond][j][k];
			    }
			    missing[j-1] = missingvalues[id][cond][j];
			}
			Object[] ret = ComputeOneSampleS_v(vector, missing, parameters.get_min_presence_list()[cond]);
			valid = (Boolean)ret[1];
			if(valid) {
			    value = (Double)ret[0];
			    mean = mean + value;
			    num++;
			}
		    }
		}
	    }
	    mean = mean / num;
	    alpha[cond] = (double) mean * 2 / (double) Math.sqrt(num_reps[cond] + num_reps[0]);
	}
	return alpha;
    }

    public Object[] ComputeS(double[] v1, double[] v2, boolean[] missingvalues1, boolean[] missingvalues2, double min_presence1, double min_presence2) {
	int length_v1 = Array.getLength(v1);
	int length_v2 = Array.getLength(v2);
	double mean1=0;
	double mean2=0;
	double[] values1 = new double[length_v1];
	double[] values2 = new double[length_v2];
	int length_values1;
	int length_values2;
	double S=0;
	double sd1;
	double sd2;
	boolean valid=true;
	int j=0;
	double m=0;
	for(int i=0;i<length_v1;i++) {
	    if(!missingvalues1[i]) {
		values1[j] = v1[i];
		m=m+v1[i];
		j++;
	    }
	}
	length_values1 = j;
	if(length_values1 < min_presence1) {
	    valid = false;
	}
	else {
	    mean1 = (double) m/length_values1;
	}
	j=0;
	m=0;
	for(int i=0;i<length_v2;i++) {
	    if(!missingvalues2[i]) {
		values2[j] = v2[i];
		m=m+v2[i];
		j++;
	    }
	}
	length_values2 = j;
	if(length_values2 < min_presence2) {
	    valid = false;
	}
	else {
	    mean2 = (double) m/length_values2;
	}
	if(valid) {
	    sd1 = 0;
	    for(int i=0; i<length_values1; i++) {
		sd1 = sd1 + (values1[i]-mean1)*(values1[i]-mean1);
	    }
	    sd1 = (double) Math.sqrt(sd1/(length_values1-1));
	    sd2 = 0;
	    for(int i=0; i<length_values2; i++) {
		sd2 = sd2 + (values2[i]-mean2)*(values2[i]-mean2);
	    }
	    sd2 = Math.sqrt(sd2/(length_values2-1));
	    S = Math.sqrt((sd1*sd1*(length_values1-1) + sd2*sd2*(length_values2-1))/(length_values1+length_values2-2));
	}
	
	Object[] returnArray = new Object[2];
	returnArray[0] = S;
	returnArray[1] = valid;
	
	return returnArray;
    }
    
    public Object[] ComputeS_v(double[][] v1, double[][] v2, boolean[] missingvalues1, boolean[] missingvalues2, double min_presence1, double min_presence2) {
	int length_vector = Array.getLength(v1[0]);
	int length_v1 = Array.getLength(v1);
	int length_v2 = Array.getLength(v2);
	double[] mean1 = new double[length_vector];
	double[] mean2 = new double[length_vector];
	double[][] values1 = new double[length_v1][length_vector];
	double[][] values2 = new double[length_v2][length_vector];
	int length_values1;
	int length_values2;
	double S=0;
	double sd1=0;
	double sd2=0;
	boolean valid=false;
	int j=0;
	double[] m = new double[length_vector];
	for(int i=0;i<length_v1;i++) {
	    if(!missingvalues1[i]) {
		for(int k=0; k<length_vector; k++) {
		    values1[j][k] = v1[i][k];
		    m[k]=m[k]+v1[i][k];
		}
		j++;
	    }
	}
	length_values1 = j;
	if(length_values1 < min_presence1) {
	    valid = false;
	}
	else {
	    valid = true;
	    for(j=0; j<length_vector; j++) {
		mean1[j] = (double) m[j]/length_values1;
	    }
	}
	j=0;
	for(int k=0; k<length_vector; k++) {    
	    m[k]=0;
	}
	for(int i=0;i<length_v2;i++) {
	    if(!missingvalues2[i]) {
		for(int k=0; k<length_vector; k++) {
		    values2[j][k] = v2[i][k];
		    m[k]=m[k]+v2[i][k];
		}
		j++;
	    }
	}
	length_values2 = j;
	if(length_values2 < min_presence2) {
	    valid = false;
	}
	else {
	    valid = true;
	    for(j=0; j<length_vector; j++) {
		mean2[j] = (double) m[j]/length_values2;
	    }
	}
	if(valid) {
	    sd1 = 0;
	    for(int i=0; i<length_values1; i++) {
		Object[] ret = ComputeDistanceBetweenVectors(values1[i],mean1);
		double x = (Double)ret[0];
		sd1 = sd1 + x*x;
	    }
	    sd1 = (double) Math.sqrt(sd1/(length_values1-1));
	    sd2 = 0;
	    for(int i=0; i<length_values2; i++) {
		Object[] ret = ComputeDistanceBetweenVectors(values2[i],mean2);
		double x = (Double)ret[0];
		sd2 = sd2 + x*x;
	    }
	    sd2 = (double) Math.sqrt(sd2/(length_values2-1));
	    S = (double) Math.sqrt((sd1*sd1*(length_values1-1) + sd2*sd2*(length_values2-1))/(length_values1+length_values2-2));
	}
	
	Object[] returnArray = new Object[2];
	returnArray[0] = S;
	returnArray[1] = valid;
	
	return returnArray;
    }
    
    public Object[] ComputePairedS(double[] v1, double[] v2, boolean[] missingvalues1, boolean[] missingvalues2, double min_presence1, double min_presence2) {

	// this is going to compute the paired t-stat, by taking differences
	// and then calling the one-sample t-stat routine.  If the two groups
	// are ratios to a common reference, then probably should take logs and
	// input only the M values

	int length_v1 = Array.getLength(v1);
	int length_v2 = Array.getLength(v2);
	int j=0;
	double[] values = new double[length_v1];
	boolean[] missing = new boolean[length_v1];
	int length_values;
	double S=0;
	for(int i=0;i<length_v1;i++) {
	    if(!missingvalues1[i] && !missingvalues2[i]) {
		values[j] = v1[i] - v2[i];
		missing[j] = false;
		j++;
	    }
	}
	length_values = j;
	boolean valid;
	if(j<min_presence1 || j<min_presence2) {
	    valid = false;
	}
	else {
	    valid = true;
	    double[] values2 = new double[length_values];
	    boolean[] missing2 = new boolean[length_values];
	    for(int i=0; i<length_values; i++) {
		values2[i] = values[i];
		missing2[i] = missing[i];
	    }
	    Object[] ret = ComputeOneSampleS(values2, missing2, 2); // third argument a dummy since missing2[j] = true for all j in this case
	    S = (Double)ret[0];
	}
	Object[] returnArray = new Object[2];
	returnArray[0] = S;
	returnArray[1] = valid;

	return returnArray;
    }

    public Object[] ComputePairedS_v(double[][] v1, double[][] v2, boolean[] missingvalues1, boolean[] missingvalues2, double min_presence1, double min_presence2) {

	// this is going to compute the paired t-stat, by taking differences
	// and then calling the one-sample t-stat routine.  This is why the paired
	// design requires running on logged data.

	int length_vector = Array.getLength(v1[0]);
	int length_v1 = Array.getLength(v1);
	int length_v2 = Array.getLength(v2);
	int j=0;
	double[][] values = new double[length_v1][length_vector];
	boolean[] missing = new boolean[length_v1];
	int length_values;
	double S=0;
	for(int i=0;i<length_v1;i++) {
	    if(!missingvalues1[i] && !missingvalues2[i]) {
		for(int k=0; k<length_vector; k++) {		
		    values[j][k] = v1[i][k] - v2[i][k];
		}
		missing[j] = false;
		j++;
	    }
	}
	length_values = j;
	boolean valid;
	if(j<min_presence1 || j<min_presence2) {
	    valid = false;
	}
	else {
	    valid = true;
	    // resize the array to be the exact size (length_values) before feeding to ComputeOneSampleS_v
	    double[][] values2 = new double[length_values][length_vector];
	    boolean[] missing2 = new boolean[length_values];
	    for(int i=0; i<length_values; i++) {
		for(int k=0; k<length_vector; k++) {
		    values2[i][k] = values[i][k];
		}
		missing2[i] = missing[i];
	    }
	    Object[] ret = ComputeOneSampleS_v(values2, missing2, 2); // third argument a dummy, as before, since missing2[j]=true for all j in this case
	    S = (Double)ret[0];
	}
	Object[] returnArray = new Object[2];
	returnArray[0] = S;
	returnArray[1] = valid;

	return returnArray;
    }

    // one sample s is just the sample standard deviation
    public Object[] ComputeOneSampleS(double[] v, boolean[] missing, double min_presence) {
	int j=0;
	double m=0;
	double mean;
	int length_v = Array.getLength(v);
	double[] values = new double[length_v];
	int length_values;
	double sd=0;
	for(int i=0; i<length_v;i++) {
	    if(!missing[i]) {
		values[j] = v[i];
		m=m+v[i];
		j++;
	    }
	}
	length_values = j;
	boolean valid;
	if(j<min_presence) {
	    valid=false;
	}
	else {
	    valid = true;
	    mean = (double) m/length_values;
	    sd = 0;
	    for(int i=0; i<length_values; i++) {
		sd = sd + (values[i]-mean)*(values[i]-mean);
	    }
	    sd = (double)Math.sqrt(sd/(length_values-1));
	    sd = sd / Math.sqrt(2);
	}
	Object[] returnArray = new Object[2];
	returnArray[0] = sd;
	returnArray[1] = valid;

	return returnArray;
    }

    public Object[] ComputeOneSampleS_v(double[][] v, boolean[] missing, double min_presence) {
	int j=0;
	int length_vector = Array.getLength(v[0]);
	double[] m = new double[length_vector];
	double[] mean = new double[length_vector];
	int length_v = Array.getLength(v);
	double[][] values = new double[length_v][length_vector];
	int length_values;
	double sd=0;
	for(int i=0; i<length_v; i++) {
	    if(!missing[i]) {
		for(int k=0; k<length_vector; k++) {
		    values[j][k] = v[i][k];
		    m[k]=m[k]+v[i][k];
		}
		j++;
	    }
	}
	length_values = j;
	boolean valid = true;
	if(j<min_presence) {
	    valid=false;
	}
	else {
	    valid = true;
	    for(int k=0; k<length_vector; k++) {
		mean[k] = (double) m[k]/length_values;
	    }
	    sd = 0;
	    for(int i=0; i<length_values; i++) {
		Object[] ret = ComputeDistanceBetweenVectors(values[i],mean);
		double x = (Double)ret[0];
		sd = sd + x*x;
	    }
	    sd = (double)Math.sqrt(sd/(length_values-1));
	}
	sd = sd / Math.sqrt(2);
	Object[] returnArray = new Object[2];
	returnArray[0] = sd;
	returnArray[1] = valid;

	return returnArray;
    }

    
    // Class Variables


}
