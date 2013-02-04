// PaGE Version 5.2

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

public class Data {

    // CONSTRUCTOR
    public Data(Parameters parameters) {
	MathematicalOperations MO = new MathematicalOperations();
	ReadDataFile(parameters);
	FindMeansUnpermuted(parameters);
	perms = MO.InitializePermuationArray(parameters.get_num_conds(), parameters.get_num_reps(), parameters.get_num_perms(), parameters.get_paired(), parameters.get_design());
	//	MO.PrintPermutationMatrix(parameters.get_num_conds(), perms, parameters.get_design(), parameters.get_start());
	parameters.set_alpha_default(MO.FindDefaultAlpha(data, data_v, parameters.get_num_conds(), missingvalue, parameters.get_min_presence_list(), parameters.get_design(), parameters.get_num_reps(), parameters));
	Print1DVectorDoubles(parameters.get_alpha_default(), "alpha_default", 10);
	Print2DVectorDoubles(unpermuted_mean, "unpermuted_mean", 10);
	Print2DVectorVectorOfDoubles(unpermuted_mean_v, "unpermuted_mean_v", 10);
	Print1DVectorStrings(ids, "ids", 10);
    }

    public void ReadDataFile (Parameters parameters) {
	data = new double[0][0][0];
	data_v = new double[0][0][0][0];
	ids_hash = new Hashtable<String, Integer>();
	// read the infile and set some summary stat variables
	int line_counter = 0;
	int[] cond = parameters.get_COND();
	int[] rep = parameters.get_REP();
	// count the number of data lines in the file
	if(!parameters.get_silent_mode()) 
	    System.out.print("Reading data file, please wait.");
	try {
	    FileInputStream inFile = new FileInputStream(parameters.get_infile());
	    BufferedReader br = new BufferedReader(new InputStreamReader(inFile));
	    String line = "#";
	    while(line.startsWith("#")) { // skip the 'comment' lines that start with "#"
		line = br.readLine();  
	    }
	    while((line = br.readLine()) != null) {
		line_counter++;
	    }
	    inFile.close();
	    if(!parameters.get_silent_mode()) 
		System.out.print(".");
	    data_v = new double[line_counter][parameters.get_num_conds()][parameters.get_max_num_reps() + 1][parameters.get_vector_length()];
	    data = new double[line_counter][parameters.get_num_conds()][parameters.get_max_num_reps() + 1];
	    missingvalue = new boolean[line_counter][parameters.get_num_conds()][parameters.get_max_num_reps()+1];
	    ids = new String[line_counter];
	    inFile = new FileInputStream(parameters.get_infile());
	    br = new BufferedReader(new InputStreamReader(inFile));
	    int num_lines = line_counter;
	    int increment = num_lines / 10;
	    line_counter = 0;  // this counts the line number *excluding* header and 'comment' lines
	    int true_line_counter = 0;  // this counts the file line number *including* header and 'comment' lines
	    line = "#";
	    while(line.startsWith("#")) { // skip the 'comment' lines that start with "#"
		line = br.readLine();  
		true_line_counter++;
	    }
	    missingvalue_flag = false;
	    while((line = br.readLine()) != null) {
		if(line_counter %  increment == 0 && line_counter > 0) {
		    if(!parameters.get_silent_mode()) 
			System.out.print(".");
		}
		true_line_counter++;
		String[] a = line.split("\t");
		int n = Array.getLength(a);
		if(n != parameters.get_num_cols()) {
		    System.out.println("\nError: The data in line " + true_line_counter + " has a different number of columns than the\nheader line.  Please check the file format and restart.\n");
		    System.exit(0);
		}
		for(int i=1; i<n; i++) {
		    if(cond[i] != -1 && rep[i] != -1) {
			double value = 0;
			try {
			    if(a[i].equals(parameters.get_missing_value_designator())) {
				missingvalue[line_counter][cond[i]][rep[i]] = true;
				missingvalue_flag = true;
			    }
			    else {
				if(parameters.get_vector_analysis()) {
				    String[] b = a[i].split(",");
				    int n2 = Array.getLength(b);
				    if(n2 > 1 && n2 != parameters.get_vector_length()) {
					int i2 = i+1;
					System.out.println("\nError: The vector in line " + true_line_counter + " column " + i2 + " has a different number of\nentries than the previous vectors.  Please check the file format and restart.\n");
					System.exit(0);
				    }
				    for(int j=0; j<parameters.get_vector_length(); j++) {
					value = Double.parseDouble(b[j]);
					data_v[line_counter][cond[i]][rep[i]][j] = value;
					if(value<min_intensity) {
					    min_intensity = value;
					    if(min_intensity < 0 && parameters.get_num_channels() == 2 && !parameters.get_data_is_logged()) {
						int ii = i+1;
						System.out.println("\n\nERROR: The data is two channels unlogged, you cannot therefore have\nnegative values.  Check line " + true_line_counter + " column " + ii + " of your file.\n");
						System.exit(0);				
					    }
					}
					if(value>max_intensity) {
					    max_intensity = value;
					}
				    }
				}
				else {
				    value = Double.parseDouble(a[i]);
				    data[line_counter][cond[i]][rep[i]] = value;
				    if(value<min_intensity) {
					min_intensity = value;
					if(min_intensity < 0 && parameters.get_num_channels() == 2 && !parameters.get_data_is_logged()) {
					    int ii = i+1;
					    System.out.println("\n\nERROR: The data is two channels unlogged, you cannot therefore have\nnegative values.  Check line " + true_line_counter + " column " + ii + " of your file.\n");
					    System.exit(0);				
					}
				    }
				    if(value>max_intensity) {
					max_intensity = value;
				    }
				}
			    }
			} catch (Exception E2) {
			    if(missingvalue_flag == false) {
				System.out.println("\nThe value in row " + true_line_counter + " column " + i + " is not a number, it is " + a[i]);
				System.out.print("Is " + a[i] + " your missing value designator? (answer Y or N): ");
				BufferedReader br2 = new BufferedReader(new InputStreamReader(System.in));
				String userinput = null;
				try {
				    userinput = br2.readLine();
				    if(userinput.equals("y") || userinput.equals("Y")) {
					parameters.set_missing_value_designator(a[i]);
					missingvalue[line_counter][cond[i]][rep[i]] = true;
					missingvalue_flag = true;					
				    }
				    else {
					System.out.println("\nERROR: If \"" + a[i] + "\" is not your missing value designator, then we don't know what it is or how to handle it...\n");
					System.exit(0);
				    }
				} catch (Exception E3) {
				    System.out.println("\nERROR: something went wrong with trying to input the data.\n");
				    System.exit(0);
				}
			    }
			    else {
				System.out.println("\nERROR: It appears you have more than one non-numeric value in your data file \"" + parameters.get_missing_value_designator() + "\" and \"" + a[i] + "\".\nOnly one non-numeric value (to represent missing values) is allowed.\n");
				System.exit(0);
			    }
			}
		    }
		}
		ids[line_counter] = a[0];
		ids_hash.put(a[0], line_counter);  // the perl was: $ids_hash{$a[0]} = $line_counter;
		line_counter++;
	    }
	    inFile.close();
	    if(!parameters.get_silent_mode()) {
		System.out.println("\n------------------------");
		System.out.println(line_counter + " rows of data.");
		System.out.println("max intensity = " + max_intensity);
		System.out.println("min intensity = " + min_intensity);
		System.out.println("------------------------");
	    }
	} catch (Exception E) {
	    System.out.println("\nSomething appears to be wrong with the input file \"" + parameters.get_infile() + "\".");
	    E.printStackTrace(System.err);
	    System.exit(0);
	}

	// if data not logged but want to use log data, then log the data table
	if(!parameters.get_data_is_logged() && parameters.get_use_logged_data()) {
	    int num_ids = Array.getLength(ids);
	    for(int id=0; id<num_ids; id++) {
		for(int c=0; c<parameters.get_num_conds(); c++) {
		    for(int r=0; r<parameters.get_num_reps()[c]; r++) {
			if(!missingvalue[id][c][r]) {
			    if(parameters.get_vector_analysis()) {
				for(int k=0; k<parameters.get_vector_length(); k++) {
				    data_v[id][c][r][k] = Math.log(data_v[id][c][r][k]);
				}
			    }
			    else {
				data[id][c][r] = Math.log(data[id][c][r]);
			    }
			}
		    }
		}
	    }
	}
	// if data logged but want to use unlogged data, then exp the data table
	int num_ids = Array.getLength(ids);
	if(parameters.get_data_is_logged() && !parameters.get_use_logged_data()) {
	    for(int id=0; id<num_ids; id++) {
		for(int c=0; c<parameters.get_num_conds(); c++) {
		    for(int r=0; r<parameters.get_num_reps()[c]; r++) {
			if(!missingvalue[id][c][r]) {
			    if(parameters.get_vector_analysis()) {
				for(int k=0; k<parameters.get_vector_length(); k++) {
				    data_v[id][c][r][k] = Math.exp(data_v[id][c][r][k]);
				}
			    }
			    else {
				    data[id][c][r] = Math.exp(data[id][c][r]);
			    }
			}
		    }
		}
	    }
	}
    }

    public void FindMeansUnpermuted(Parameters parameters) {
	MathematicalOperations MO = new MathematicalOperations();
	int num_ids = Array.getLength(ids);
	int num_conds = parameters.get_num_conds();
	unpermuted_mean = new double[num_ids][num_conds];
	unpermuted_mean_v = new double[num_ids][num_conds][parameters.get_vector_length()];
	unpermuted_mean_valid = new boolean[num_ids][num_conds];
	for(int cond=0; cond<num_conds; cond++) {
	    for(int id=0; id<num_ids; id++) {
		if(parameters.get_vector_analysis()) {
		    for(int k=0; k<parameters.get_vector_length(); k++) {
			double[] d = new double[parameters.get_num_reps()[cond]];
			boolean[] m = new boolean[parameters.get_num_reps()[cond]];
			for(int i=1; i<parameters.get_num_reps()[cond]+1; i++) {
			    d[i-1] = data_v[id][cond][i][k];
			    m[i-1] = missingvalue[id][cond][i];
			}
			Object[] returnArray = MO.ComputeMean(d, missingvalue[id][cond], parameters.get_min_presence_list()[cond]);
			unpermuted_mean_v[id][cond][k] = (Double)returnArray[0];
			unpermuted_mean_valid[id][cond] = (Boolean)returnArray[1];  // this will either be valid for every k or invalid for every k
		    }
		}
		else {
		    double[] d = new double[parameters.get_num_reps()[cond]];
		    boolean[] m = new boolean[parameters.get_num_reps()[cond]];
		    for(int i=1; i<parameters.get_num_reps()[cond]+1; i++) {
			d[i-1] = data[id][cond][i];
			m[i-1] = missingvalue[id][cond][i];
		    }
		    Object[] returnArray = MO.ComputeMean(data[id][cond], missingvalue[id][cond], parameters.get_min_presence_list()[cond]);
		    unpermuted_mean[id][cond] = (Double)returnArray[0];
		    unpermuted_mean_valid[id][cond] = (Boolean)returnArray[1];
		}
	    }
	}
    }
    
    public void Print1DVectorDoubles(double[] vector, String vector_name, int num) {
	try {
	    System.out.println("------------------------\n" + vector_name);
	    for(int i=0; i<num; i++) {
		System.out.println(i + "\t" + vector[i]);
	    }
	    System.out.println("------------------------");
	} catch (Exception E) {}
    }

    public void Print2DVectorDoubles(double[][] vector, String vector_name, int num) {
	try {
	    System.out.println("------------------------\n" + vector_name);
	    System.out.print("\t");
	    int sz = Array.getLength(vector[0]);
	    for(int j=0; j<sz; j++) {
		System.out.print("\t" + j);
	    }
	    System.out.print("\n");
	    for(int i=0; i<num; i++) {
		System.out.print(i);
		sz = Array.getLength(vector[i]);
		for(int j=0; j<sz; j++) {
		    System.out.print("\t" + vector[i][j]);
		}
		System.out.print("\n");
	    }
	    System.out.println("------------------------");
	} catch (Exception E) {}
    }

    public void Print2DVectorVectorOfDoubles(double[][][] vector, String vector_name, int num) {
	try {
	    System.out.println("------------------------\n" + vector_name);
	    System.out.print("\t");
	    int sz = Array.getLength(vector[0]);
	    for(int j=0; j<sz; j++) {
		System.out.print("\t" + j);
	    }
	    System.out.print("\n");
	    for(int i=0; i<num; i++) {
		System.out.print(i);
		sz = Array.getLength(vector[i]);
		for(int j=0; j<sz; j++) {
		    int sz2 = Array.getLength(vector[i][j]);
		    System.out.print("\t");
		    double v = (double)Math.round(vector[i][j][0] * 1000) / 1000;
		    System.out.print(v);
		    for(int k=1; k<sz2; k++) {
			v = (double)Math.round(vector[i][j][k] * 1000) / 1000;
			System.out.print("," + v);
		    }
		}
		System.out.print("\n");
	    }
	    System.out.println("------------------------");
	} catch (Exception E) {}
    }

    public void Print1DVectorInts(int[] vector, String vector_name, int num) {
	try {
	    System.out.println("------------------------\n" + vector_name);
	    for(int i=0; i<num; i++) {
		System.out.println(i + "\t" + vector[i]);
	    }
	    System.out.println("------------------------");
	} catch (Exception E) {}
    }

    public void Print2DVectorInts(int[][] vector, String vector_name, int num) {
	try {
	    System.out.println("------------------------\n" + vector_name);
	    System.out.print("\t");
	    int sz = Array.getLength(vector[0]);
	    for(int j=0; j<sz; j++) {
		System.out.print("\t" + j);
	    }
	    System.out.print("\n");
	    for(int i=0; i<num; i++) {
		System.out.print(i);
		sz = Array.getLength(vector[i]);
		for(int j=0; j<sz; j++) {
		    System.out.print("\t" + vector[i][j]);
		}
		System.out.print("\n");
	    }
	    System.out.println("------------------------");
	} catch (Exception E) {}
    }
    
    public void Print1DVectorStrings(String[] vector, String vector_name, int num) {
	try {
	    System.out.println("------------------------\n" + vector_name);
	    for(int i=0; i<num; i++) {
		System.out.println(i + "\t" + vector[i]);
	    }
	    System.out.println("------------------------");
	} catch (Exception E) {}
    }

    public void Print2DVectorDoubles(String[][] vector, String vector_name, int num) {
	try {
	    System.out.println("------------------------\n" + vector_name);
	    System.out.print("\t");
	    int sz = Array.getLength(vector[0]);
	    for(int j=0; j<sz; j++) {
		System.out.print("\t" + j);
	    }
	    System.out.print("\n");
	    for(int i=0; i<num; i++) {
		System.out.print(i);
		sz = Array.getLength(vector[i]);
		for(int j=0; j<sz; j++) {
		    System.out.print("\t" + vector[i][j]);
		}
		System.out.print("\n");
		System.out.println("------------------------");
	    }
	} catch (Exception E) {}
    }
    
    // Class Variables
    double[][][] data;
    double[][][][] data_v;
    boolean[][][] missingvalue;
    boolean missingvalue_flag; // this flag indicates there are missing values in the data file
    String[] ids;
    double min_intensity;
    double max_intensity;
    Hashtable<String, Integer> ids_hash;
    double[][] unpermuted_mean;
    double[][][] unpermuted_mean_v;
    boolean[][] unpermuted_mean_valid;
    int[][][] perms;
}
