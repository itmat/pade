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

public class Parameters {

    // CONSTRUCTOR
    public Parameters(String[] args) {

	// We first set things which take default values, then everything
        // else is determined either from the command line or from input
	// the user is prompted for during the run
	aux_page_size = 500;
	silent_mode = false;
	num_perms = 200;
	num_perms_default = 200;
	num_bins = 1000;
	num_bins_default = 1000;
	note = "";
	missing_value_designator = "";
	WriteFirstStuffToScreen(args);
	ParseCommandlineArgs(args);
	GetParametersNotSpecifiedOnCommandLineByPromptingUser();
	CheckParameterCompatibilityAndDoFinalParameterInitialization();
    }

    public void ParseCommandlineArgs(String[] args_temp) {
	boolean argument_is_valid = false;
	int numArgs = Array.getLength(args_temp);
	String[] args = new String[numArgs+1];
	for(int i=0; i<numArgs; i++) {
	    args[i] = args_temp[i];
	}
	args[numArgs] = "end_of_params";
	for(int arg = 0; arg<numArgs; arg++) {
	    argument_is_valid = false;
	    if(args[arg].startsWith("--")) {
		if(args[arg].endsWith("tstat_tuning_parameter")) {
		    if(args[arg+1].startsWith("--") || args[arg+1].startsWith("end_of_params")) {
			System.err.println("\nERROR: The parameter '--tstat_tuning_parameter' was given without a value\n");
			System.exit(0);
		    }
		    try {
			tstat_tuning_parameter = Double.parseDouble(args[arg+1]);
			arg++;
			argument_is_valid = true;
			tstat_tuning_parameter_specified = true;
		    } catch (Exception e) { 
			System.err.println("\nERROR: The parameter '--tstat_tuning_parameter' must be a number: you have entered \"" + args[arg+1] + "\"\n");
			System.exit(0);
		    }
		}
		if(args[arg].endsWith("aux_page_size")) {
		    if(args[arg+1].startsWith("--") || args[arg+1].startsWith("end_of_params")) {
			System.err.println("\nERROR: The parameter '--aux_page_size' was given without a value\n");
			System.exit(0);
		    }
		    try {
			aux_page_size = Integer.parseInt(args[arg+1]);
			arg++;
			argument_is_valid = true;
		    } catch (Exception e) { 
			System.err.println("\nERROR: The parameter '--aux_page_size' must be an integer: you have entered \"" + args[arg+1] + "\"\n");
			System.exit(0);
		    }
		}
		if(args[arg].endsWith("use_logged_data")) {
		    if(args[arg+1].startsWith("--") || args[arg+1].startsWith("end_of_params")) {
			System.err.println("\nERROR: The parameter '--use_logged_data' was given without a value, it should be 'true' or 'false'\n");
			System.exit(0);
		    }
		    try {
			argument_is_valid = false;
			if(args[arg+1].equals("true")) {
			    use_logged_data = true;
			    argument_is_valid = true;
			}
			if(args[arg+1].equals("false")) {
			    use_logged_data = false;
			    argument_is_valid = true;
			}
			if(!argument_is_valid) {
			    System.err.println("\nERROR: The parameter '--use_logged_data' must be 'true' or 'false', you have put " + args[arg+1] + "\n");
			    System.exit(0);
			}
			arg++;
			use_logged_data_specified = true;
		    } catch (Exception e) { 
			System.err.println("\nERROR: The parameter '--use_logged_data' must be a 'true' or 'false': you have entered \"" + args[arg+1] + "\"\n");
			System.exit(0);
		    }
		}
		if(args[arg].endsWith("data_is_logged")) {
		    if(args[arg+1].startsWith("--") || args[arg+1].startsWith("end_of_params")) {
			System.err.println("\nERROR: The parameter '--data_is_logged' was given without a value, it should be 'true' or 'false'\n");
			System.exit(0);
		    }
		    try {
			argument_is_valid = false;
			if(args[arg+1].equals("true")) {
			    data_is_logged = true;
			    argument_is_valid = true;
			}
			if(args[arg+1].equals("false")) {
			    data_is_logged = false;
			    argument_is_valid = true;
			}
			if(!argument_is_valid) {
			    System.err.println("\nERROR: The parameter '--data_is_logged' must be 'true' or 'false', you have put " + args[arg+1] + "\n");
			    System.exit(0);
			}
			arg++;
			data_is_logged_specified = true;
		    } catch (Exception e) { 
			System.err.println("\nERROR: The parameter '--data_is_logged' must be a 'true' or 'false': you have entered \"" + args[arg+1] + "\"\n");
			System.exit(0);
		    }
		}
		if(args[arg].endsWith("silent_mode")) {
		    silent_mode = true;
		    argument_is_valid = true;
		}
		if(args[arg].endsWith("print_warnings")) {
		    printWarnings = true;
		    argument_is_valid = true;
		}
		if(args[arg].endsWith("vector_analysis")) {
		    vector_analysis = true;
		    argument_is_valid = true;
		}
		if(args[arg].endsWith("output_gene_confidence_list")) {
		    output_gene_confidence_list = true;
		    argument_is_valid = true;
		}
		if(args[arg].endsWith("output_gene_confidence_list_combined")) {
		    output_gene_confidence_list_combined = true;
		    argument_is_valid = true;
		}
		if(args[arg].endsWith("output_text")) {
		    output_text = true;
		    argument_is_valid = true;
		}
		if(args[arg].matches("--paired")) {
		    if(args[arg+1].startsWith("--") || args[arg+1].startsWith("end_of_params")) {
			System.err.println("\nERROR: The parameter '--paired' was given without a value (should be true or false)\n");
			System.exit(0);
		    }
		    if(args[arg+1].equals("true") || args[arg+1].equals("yes") || args[arg+1].equals("YES") || args[arg+1].equals("True") || args[arg+1].equals("TRUE") || args[arg+1].equals("Yes")) {
			paired = true;
			arg++;
			argument_is_valid = true;
		    }
		    if(args[arg+1].equals("false") || args[arg+1].equals("no") || args[arg+1].equals("NO") || args[arg+1].equals("False") || args[arg+1].equals("FALSE") || args[arg+1].equals("No")) {
			paired = false;
			arg++;
			argument_is_valid = true;
		    }
		    if(argument_is_valid) {
			paired_specified = true;
		    }
		    else {
			System.err.println("\nERROR: The parameter '--paired' must be true or false: you have entered \"" + args[arg+1] + "\"\n");
			System.exit(0);
		    }
		}
		if(args[arg].endsWith("num_perms") || args[arg].endsWith("num_permutations")) {
		    if(args[arg+1].startsWith("--") || args[arg+1].startsWith("end_of_params")) {
			System.err.println("\nERROR: The parameter '--num_perms' was given without a value\n");
			System.exit(0);
		    }
		    try {
			num_perms = Integer.parseInt(args[arg+1]);
			arg++;
			argument_is_valid = true;
			if(num_perms < 2) {
			    if(!silent_mode) 
				System.out.println("\n **** Note: the number of permutations must be at least two. ****\n      It is being set to the default of " + num_perms_default + ".\n");
			    num_perms = num_perms_default;
			}

		    } catch (Exception e) { 
			System.err.println("\nERROR: The parameter '--num_permutations' must be an integer: you have entered \"" + args[arg+1] + "\"\n");
			System.exit(0);
		    }
		}
		if(args[arg].endsWith("num_bins") || args[arg].endsWith("number_bins")) {
		    if(args[arg+1].startsWith("--") || args[arg+1].startsWith("end_of_params")) {
			System.err.println("\nERROR: The parameter '--num_bins' was given without a value\n");
			System.exit(0);
		    }
		    try {
			num_bins = Integer.parseInt(args[arg+1]);
			arg++;
			argument_is_valid = true;
			if(num_bins <10) {
			    if(!silent_mode) 
				System.out.println("\n **** Note: the number of bins  must be at least ten. ****\n      It is being set to the default of " + num_bins_default + ".\n");
			    num_bins = num_bins_default;
			}
		    } catch (Exception e) { 
			System.err.println("\nERROR: The parameter '--num_bins' must be an integer: you have entered \"" + args[arg+1] + "\"\n");
			System.exit(0);
		    }
		}
		if(args[arg].endsWith("infile")) {
		    if(args[arg+1].startsWith("--") || args[arg+1].startsWith("end_of_params")) {
			System.err.println("\nERROR: The command line argument '--infile' was given without a value\n");
			System.exit(0);
		    }
		    try {
			infile = new File(args[arg+1]);
			arg++;
			argument_is_valid = true;
			if(!(infile.exists())) {
			    System.err.println("\nERROR: The file \"" + infile + "\" does not appear to exist\n");
			    System.exit(0);	    
			}
		    } catch (Exception e) { 
			e.printStackTrace(System.err);
			System.exit(0);
		    }
		}
		if(args[arg].endsWith("id2info")) {
		    if(args[arg+1].startsWith("--") || args[arg+1].startsWith("end_of_params")) {
			System.err.println("\nERROR: The command line argument '--id2info' was given without a value\n");
			System.exit(0);
		    }
		    try {
			id2info_file = new File(args[arg+1]);
			arg++;
			argument_is_valid = true;
			id2info_specified = true;
		    } catch (Exception e) { 
			e.printStackTrace(System.err);
			System.exit(0);
		    }
		}
		if(args[arg].endsWith("id2url")) {
		    if(args[arg+1].startsWith("--") || args[arg+1].startsWith("end_of_params")) {
			System.err.println("\nERROR: The command line argument '--id2url' was given without a value\n");
			System.exit(0);
		    }
		    try {
			id2url_file = new File(args[arg+1]);
			arg++;
			argument_is_valid = true;
			id2url_specified = true;
		    } catch (Exception e) { 
			e.printStackTrace(System.err);
			System.exit(0);
		    }
		}
		if(args[arg].endsWith("outfile")) {
		    if(args[arg+1].startsWith("--") || args[arg+1].startsWith("end_of_params")) {
			System.err.println("\nERROR: The command line argument '--outfile' was given without a value\n");
			System.exit(0);
		    }
		    try {
			outfile = new File(args[arg+1]);
			arg++;
			argument_is_valid = true;
		    } catch (Exception e) { 
			e.printStackTrace(System.err);
			System.exit(0);
		    }
		}
		if(args[arg].endsWith("min_presence")) {
		    if(args[arg+1].startsWith("--") || args[arg+1].startsWith("end_of_params")) {
			System.err.println("\nERROR: The parameter '--min_presence' was given without a value\n");
			System.exit(0);
		    }
		    try {
			min_presence = Integer.parseInt(args[arg+1]);
			arg++;
			argument_is_valid = true;
			min_presence_specified = true;
		    } catch (Exception e) { 
			System.err.println("\nERROR: The parameter '--min_presence' must be an integer: you have entered \"" + args[arg+1] + "\"\n");
			System.exit(0);
		    }
		}
		if(args[arg].endsWith("min_presence_list")) {
		    if(args[arg+1].startsWith("--") || args[arg+1].startsWith("end_of_params")) {
			System.err.println("\nERROR: The parameter '--min_presence_list' was given without a value\n");
			System.exit(0);
		    }
		    try {
			String[] temp = args[arg+1].split(",");
			int sz = Array.getLength(temp);
			min_presence_list = new int[sz];
			for(int i=0; i<sz; i++) {
			    min_presence_list[i] = Integer.parseInt(temp[i]);
			}
			arg++;
			argument_is_valid = true;
			min_presence_list_specified = true;
		    } catch (Exception e) { 
			System.err.println("\nERROR: The parameter '--min_presence_list' must be a comma delimited list of integers (without whitespace): you have entered \"" + args[arg+1] + "\"\n");
			System.exit(0);
		    }
		}
		if(args[arg].endsWith("level_confidence")) {
		    if(args[arg+1].startsWith("--") || args[arg+1].startsWith("end_of_params")) {
			System.err.println("\nERROR: The parameter '--level_confidence' was given without a value\n");
			System.exit(0);
		    }
		    try {
			if(args[arg+1].equals("L") || args[arg+1].equals("l")) {
			    level_confidence_later = true;
			}
			else {
			    level_confidence = Double.parseDouble(args[arg+1]);
			    if(level_confidence <= 0 || level_confidence >= 1) {
				System.err.println("\nERROR: The parameter '--level_confidence' must be a number (strictly) between 0 and 1: you have entered \"" + level_confidence + "\"\n");
				System.exit(0);
			    }
			}
			argument_is_valid = true;
			level_confidence_specified = true;
			arg++;
		    } catch (Exception e) { 
			System.err.println("\nERROR: The parameter '--level_confidence' must be a number (strictly) between 0 and 1: you have entered \"" + args[arg+1] + "\"\n");
			System.exit(0);
		    }
		}
		if(args[arg].endsWith("level_confidence_list")) {
		    if(args[arg+1].startsWith("--") || args[arg+1].startsWith("end_of_params")) {
			System.err.println("\nERROR: The parameter '--level_confidence_list' was given without a value\n");
			System.exit(0);
		    }
		    try {
			String[] temp = args[arg+1].split(",");
			int sz = Array.getLength(temp);
			level_confidence_list = new double[sz+1];
			level_confidence_list_specified = true;
			for(int i=0; i<sz; i++)
			    level_confidence_list[i+1] = Double.parseDouble(temp[i]);
			arg++;
			argument_is_valid = true;
		    } catch (Exception e) { 
			System.err.println("\nERROR: The parameter '--level_confidence_list' must be a comma delimited list of numbers (without whitespace): you have entered \"" + args[arg+1] + "\"\n");
			System.exit(0);
		    }
		}
		if(args[arg].endsWith("note")) {
		    if(args[arg+1].startsWith("--") || args[arg+1].startsWith("end_of_params")) {
			System.err.println("\nERROR: The parameter '--note' was given without a value\n");
			System.exit(0);
		    }
		    note = args[arg+1];
		    arg++;
		    argument_is_valid = true;
		}
		if(args[arg].endsWith("num_channels") || args[arg].endsWith("num_permutations")) {
		    if(args[arg+1].startsWith("--") || args[arg+1].startsWith("end_of_params")) {
			System.err.println("\nERROR: The parameter '--num_channels' was given without a value\n");
			System.exit(0);
		    }
		    try {
			num_channels = Integer.parseInt(args[arg+1]);
			argument_is_valid = true;
			if(num_channels < 1 || num_channels > 2) {
			    System.err.println("\nERROR: The parameter '--num_channels' must be 1 or 2: you have entered \"" + args[arg+1] + "\"\n");
			System.exit(0);
			}
			arg++;
		    } catch (Exception e) { 
			System.err.println("\nERROR: The parameter '--num_channels' must be an integer: you have entered \"" + args[arg+1] + "\"\n");
			System.exit(0);
		    }
		}
		if(args[arg].endsWith("design")) {
		    if(args[arg+1].startsWith("--") || args[arg+1].startsWith("end_of_params")) {
			System.err.println("\nERROR: The parameter '--design' was given without a value\n");
			System.exit(0);
		    }
		    try {
			char[] temp = args[arg+1].toCharArray();
			int n = Array.getLength(temp);
			if(n > 1) {
			    System.err.println("\nERROR: The parameter '--design' must be 'D' or 'R': you have entered \"" + args[arg+1] + "\"\n");
			    System.exit(0);
			}
			design = temp[0];
			// allow input to be case insensitive
			if(design == 'r') {design = 'R';}
			if(design == 'd') {design = 'D';}
			argument_is_valid = true;
			if(!(design == 'D' || design == 'R')) {
			    System.err.println("\nERROR: The parameter '--design' must be 'D' or 'R': you have entered \"" + args[arg+1] + "\"\n");
			    System.exit(0);
			}
			arg++;
			if((design == 'R' || design == 'D') && num_channels == 1) {
			    System.err.println("\nError: if the number of channels is 1, do not specify the design type\n\n"); 
			    System.exit(0);
			}
			if(design == 'R' || design == 'D') {
			    num_channels = 2; 
			}
		    } catch (Exception e) { 
			System.err.println("\nERROR: The parameter '--design' must be 'D' or 'R': you have entered \"" + args[arg+1] + "\"\n");
			System.exit(0);
		    }

		}
		if(args[arg].endsWith("missing_value_designator")) {
		    if(args[arg+1].startsWith("--") || args[arg+1].startsWith("end_of_params")) {
			System.err.println("\nERROR: The parameter '--missing_value_designator' was given without a value\n");
			System.exit(0);
		    }
		    missing_value_designator = args[arg+1];
		    arg++;
		    argument_is_valid = true;
		}
	    }
	    if(argument_is_valid == false) {
		if(args[arg].matches("-[^-].*")) {
		    System.err.println("\nERROR: The command line argument '" + args[arg] + "' is unrecognized - it starts with only one dash, please use two dashes\n");
		}
		else {
		    System.err.println("\nERROR: The command line argument '" + args[arg] + "' is unrecognized - please check the spelling\n");
		}
		System.exit(0);		    
	    }
	}
    }

    public void GetParametersNotSpecifiedOnCommandLineByPromptingUser() {

	if(num_channels != 1 && num_channels !=2) {
	    boolean flag = true;
	    while(num_channels != 1 && num_channels != 2) {
		if(flag) {
		    System.out.print("\nAre the arrays 1-Channel or 2-Channel arrays?  (enter \"1\" or \"2\")  ");
		    flag = false;
		}
		else {
		    System.out.print("Please input 1 or 2: ");
		}
		BufferedReader br2 = new BufferedReader(new InputStreamReader(System.in));
		String userinput = null;
		try {
		    userinput = br2.readLine();
		    num_channels = Integer.parseInt(userinput);
		} catch (Exception E) {
		    flag = false;
		}
	    }
	}
	
	// Things relating to two channel data
	if(num_channels == 2) {
	    if(paired == true) {
		design = 'R';
	    }
	    if(!(design == 'R') && !(design == 'D')) {
		System.out.print("\nIs it a reference design or direct comparison design?: (enter R or D): ");
		BufferedReader br2 = new BufferedReader(new InputStreamReader(System.in));
		String userinput = null;
		try {
		    userinput = br2.readLine();
		} catch (Exception ioe) {
		    System.out.println("Sorry, there was an error in trying to input the design.");
		    System.exit(0);
		}
		if(userinput.equals("r") || userinput.equals("R")) {
		    design = 'R';
		}
		if(userinput.equals("d") || userinput.equals("D")) {
		    design = 'D';
		}
		while(!(design == 'R') && !(design == 'D')) {
		    System.out.print("Please input R or D: ");
		    br2 = new BufferedReader(new InputStreamReader(System.in));
		    userinput = null;
		    try {
			userinput = br2.readLine();
		    } catch (IOException ioe) {
			System.out.println("Sorry, there was an error in trying to input the design.");
			System.exit(1);
		    }
		    if(userinput.equals("r") || userinput.equals("R")) {
			design = 'R';
		    }
		    if(userinput.equals("d") || userinput.equals("D")) {
			design = 'D';
		    }
		}
	    }
	}

	// if not specified command line whether the data is logged, ask user
	if(!data_is_logged_specified) {
	    boolean flag = false;
	    boolean flag2 = true;
	    while(flag == false) {
		if(flag2) {
		    System.out.print("\nIs the data log transformed? (enter Y or N): ");
		    flag2 = false;
		}
		else {
		    System.out.print("\nEnter Y or N: ");
		}
		BufferedReader br2 = new BufferedReader(new InputStreamReader(System.in));
		String userinput = null;
		try {
		    userinput = br2.readLine();
		} catch (IOException ioe) {
		    System.out.println("Sorry, there was an error in trying process your input.");
		    System.exit(1);
		}
		if(userinput.equals("y") || userinput.equals("Y")) {
		    data_is_logged = true;
		    flag = true;
		}
		if(userinput.equals("n") || userinput.equals("N")) {
		    data_is_logged = false;
		    flag = true;
		}
	    }
	}
	// Get name of infile and make sure it can be opened
	// note: infile name may be provided on the command line
	if(infile ==  null) {
	    boolean flag = false;
	    while(flag == false) {
		System.out.print("\nEnter the name of the datafile: ");
		BufferedReader br2 = new BufferedReader(new InputStreamReader(System.in));
		String userinput = null;
		try {
		    userinput = br2.readLine();
		    infile = new File(userinput);
		    if(infile.exists()) {
			flag = true;
		    }
		    else {
			System.out.println("\nThe file \"" + infile + "\" does not appear to exist.");
		    }
		} catch (IOException ioe) {
		    System.out.println("\nSorry, there was an error in trying process your input.\n");
		    flag = false;
		}
	    }
	}

	ReadHeader();

	boolean pflag = true;
	for(int c=1; c<num_conds; c++) {
	    if(num_reps[c] != num_reps[0]) {
		pflag = false;
	    }
	}

	if(pflag && !paired_specified && ((design == 'R' && num_channels == 2) || num_channels == 1)) {
	    System.out.print("\nAre the arrays paired? (enter Y or N): ");
	    BufferedReader br2 = new BufferedReader(new InputStreamReader(System.in));
	    String userinput = null;
	    try {
		userinput = br2.readLine();
	    } catch (IOException ioe) {
		System.out.println("Sorry, there was an error in trying process your input.");
		System.exit(1);
	    }
	    boolean flag = false;
	    if(userinput.equals("y") || userinput.equals("Y")) {
		paired = true;
		flag = true;
	    }
	    if(userinput.equals("n") || userinput.equals("N")) {
		paired = false;
		flag = true;
	    }
	    while(flag == false) {
		System.out.print("Please input Y or N: ");
		br2 = new BufferedReader(new InputStreamReader(System.in));
		userinput = null;
		try {
		    userinput = br2.readLine();
		} catch (IOException ioe) {
		    System.out.println("Sorry, there was an error in trying process your input.");
		    System.exit(1);
		}
		if(userinput.equals("y") || userinput.equals("Y")) {
		    paired = true;
		    flag = true;
		}
		if(userinput.equals("n") || userinput.equals("N")) {
		    paired = false;
		    flag = true;
		}
	    }
	}

	if(design == 'R' && num_conds == 1) {
	    System.out.println("\nError: Reference Design with only one condition...\n");
	    System.exit(0);
	}
	if(num_channels == 1 && num_conds == 1) {
	    System.out.println("\nError: One-Channel data with only one condition...\n");
	    System.exit(0);
	}

	// if neither level_confidence nor level_confidence_list specified, ask user
	if(!level_confidence_specified && !level_confidence_list_specified && !level_confidence_later) {
	    Set_level_confidence();
	}

	// if neither min_presence nor min_presence_list specified, ask user
	if(!min_presence_specified && !min_presence_list_specified) {
	    Set_min_presence();
	}

	// if not specified command line whether the data is logged, ask user
	if(!use_logged_data_specified && !(num_channels == 2 && design == 'D') && !paired) {
	    boolean flag = false;
	    boolean flag2 = true;
	    while(flag == false) {
		if(flag2) {
		    System.out.print("\nUse log transformed data? (enter Y or N): ");
		    flag2 = false;
		}
		else {
		    System.out.print("\nEnter Y or N: ");
		}
		BufferedReader br2 = new BufferedReader(new InputStreamReader(System.in));
		String userinput = null;
		try {
		    userinput = br2.readLine();
		} catch (IOException ioe) {
		    System.out.println("Sorry, there was an error in trying process your input.");
		    System.exit(1);
		}
		if(userinput.equals("y") || userinput.equals("Y") || userinput.equals("t") || userinput.equals("T")) {
		    use_logged_data = true;
		    flag = true;
		}
		if(userinput.equals("n") || userinput.equals("N") || userinput.equals("f") || userinput.equals("F")) {
		    use_logged_data = false;
		    flag = true;
		}
	    }
	}
	if(outfile == null) {
	    boolean backslash_flag=false;
	    String path = infile.getAbsolutePath();
	    path = path.replaceAll("[^/]*$","");
	    String fname = infile.getName();
	    String fullname =  path + "PaGE-results-for-" + fname;
	    fullname = fullname.replaceAll("\\s","");
	    fullname = fullname.replaceAll(".txt$","");
	    fullname = fullname.replaceAll("_txt$","");
	    fullname = fullname.replaceAll(".TXT$","");
	    fullname = fullname.replaceAll("_TXT$","");
	    fullname = fullname.replaceAll("_Txt$","");
	    fullname = fullname.replaceAll(".Txt$","");
	    outfile = new File(fullname);
	}
	if(id2info_specified) {
	    id2info_hash = ReadId2infoFile(id2info_file);
	}
	if(id2url_specified) {
	    id2url_hash = ReadId2urlFile(id2url_file);
	}
    }

    public Hashtable ReadId2infoFile(File id2info_file) {
	Hashtable h = new Hashtable();
	try {
	    boolean flag = false;
            FileInputStream inFile = new FileInputStream(id2info_file);
            BufferedReader br = new BufferedReader(new InputStreamReader(inFile));
            String line = "";
            while((line = br.readLine()) != null) {
		String[] temp = line.split("\\t", 0);
		if(Array.getLength(temp) != 2) {
		    System.out.println("\n\nError: id2info file must be tab delimited, two columns mapping id to info.");
		    System.exit(0);
		}
		if(h.containsKey(temp[0])) {
		    String x = (String)(h.get(temp[0]));
		    String y = (String)temp[1];
		    x = x.trim();
		    y = y.trim();
		    if(!x.equals(y)) {
			if(printWarnings)
			    System.out.println("Warning: id \"" + temp[0] + "\" is repeated in the id2info file.");
			flag = true;
		    }
		}
		h.put(temp[0],temp[1]);
	    }
	    if(flag && !printWarnings && !silent_mode) {
		System.out.println("WARNING: ids repeated in the id2info file with inconsistent info.\nRerun with --printwarnings to print out all repeated ids");
	    }
	} catch (Exception E) {
	    System.out.println("\n\nError: Something is wrong with your id2info file\n.");
	    System.exit(0);
	}
	return h;
    }

    public Hashtable ReadId2urlFile(File id2info_file) {
	Hashtable h = new Hashtable();
	try {
            FileInputStream inFile = new FileInputStream(id2url_file);
            BufferedReader br = new BufferedReader(new InputStreamReader(inFile));
            String line = "";
	    boolean flag = false;
            while((line = br.readLine()) != null) {
		String[] temp = line.split("\\t", 0);
		if(Array.getLength(temp) != 2) {
		    System.out.println("\n\nError: id2url file must be tab delimited, two columns mapping id to url.");
		    System.exit(0);
		}
		if(h.containsKey(temp[0])) {
		    String x = (String)(h.get(temp[0]));
		    String y = (String)temp[1];
		    x = x.trim();
		    y = y.trim();
		    if(!x.equals(y)) {
			if(printWarnings)
			    System.out.println("Warning: id \"" + temp[0] + "\" is repeated in the id2url file.");
			flag = true;
		    }
		}
		h.put(temp[0],temp[1]);
	    }
	    if(flag && !printWarnings && !silent_mode) {
		System.out.println("WARNING: ids repeated in the id2url file with inconsistent info.\nRerun with --printwarnings to print out all repeated ids");
	    }
	} catch (Exception E) {
	    System.out.println("\n\nError: Something is wrong with your id2url file\n.");
	    System.exit(0);
	}
	return h;
    }

    public void Set_min_presence () {
	if(num_conds>1) {
	    System.out.print("\nPlease enter the min number of non-missing values there must be in each\ncondition for a row to not be ignored (a positive integer greater than 1)\n(or enter S to specify a separate one for each condition): ");
	}
	else {
	    System.out.print("\nPlease enter the min number of non-missing values necessary\nfor a row to not be ignored (a positive integer greater than 1): ");
	}
	BufferedReader br2 = new BufferedReader(new InputStreamReader(System.in));
	String userinput = null;
	boolean flag = false;
	min_presence_separate = false;
	while(flag == false) {
	    try {
		userinput = br2.readLine();
		if((!userinput.equals("S") && !userinput.equals("s")) || num_conds<2) {
		    min_presence = Integer.parseInt(userinput);
		    if(min_presence>1) {
			flag = true;
			min_presence_specified = true;
		    }
		    else {
			if(num_conds>1) {
			    System.out.print("\n\nThe min presence must be an integer greater than 1\n(or enter S to specify a separate min presence for each condition.\nPlease re-enter it: ");
			}
			else {
			    System.out.print("\n\nThe min presence must be an integer greater than 1.\nPlease re-enter it: ");
			}
		    }
		}
		else {
		    min_presence_separate = true;
		    flag = true;
		}
	    } catch (Exception e) {
		if(num_conds>1) {
		    System.out.print("\n\nThe min presence must be an integer greater than 1\n(or enter S to specify a separate min presence for each condition).\nPlease re-enter it: ");
		}
		else {
		    System.out.print("\n\nThe min presence must be an integer greater than 1.\nPlease re-enter it: ");
		}
		flag = false;
	    }
	}
	if(min_presence_separate) {
	    for(int i=0; i<num_conds; i++) {
		System.out.print("Enter the min presence for condition " + i + ": ");
		br2 = new BufferedReader(new InputStreamReader(System.in));
		userinput = null;
		flag = false;
		while(flag == false) {
		    try {
			userinput = br2.readLine();
			min_presence_list[i] = Integer.parseInt(userinput);
			if(min_presence_list[i] > 1) {
			    flag = true;
			}
			else {
			    System.out.print("\nThe min presence must be an integer greater than 1.\nPlease re-enter it: ");
			    flag = false;
			}
		    } catch (Exception E) {
			System.out.print("\nThe min presence must be an integer greater than 1.\nPlease re-enter it: ");
			flag = false;
		    }
		}
	    }
	    min_presence_list_specified = true;
	}

    }

    public void Set_level_confidence () {
	if(num_conds>2) {
	    System.out.print("Please enter the level confidence (a number between 0 and 1)\n(or enter S to specify a separate confidence for each group\nor enter L to give the confidence later): ");
	}
	else {
	    System.out.print("Please enter the level confidence (a number between 0 and 1)\n(or L to give the confidence later): ");
	}
	BufferedReader br2 = new BufferedReader(new InputStreamReader(System.in));
	String userinput = null;
	boolean flag = false;
	level_confidence_separate = false;
	while(flag == false) {
	    try {
		userinput = br2.readLine();
		if(userinput.equals("L") || userinput.equals("l")) {
		    level_confidence_later = true;
		    flag = true;
		    level_confidence_specified = false;
		}
		else {
		    if((!userinput.equals("S") && !userinput.equals("s")) || num_conds<3) {
			level_confidence = Double.parseDouble(userinput);
			if(0<level_confidence && level_confidence<1) {
			    flag = true;
			    level_confidence_specified = true;
			}
			else {
			    if(num_conds>2) {
				System.out.print("\nThe level confidence must be a number strictly beween 0 and 1\n(or enter S to specify a separate confidence for each group\nor enter L to specify the confidence later).\nPlease re-enter it: ");
			    }
			    else {
				System.out.print("\nThe level confidence must be a number strictly beween 0 and 1\n(or enter L to specify the confidence later).\nPlease re-enter it: ");
			    }
			}
		    }
		    else {
			level_confidence_separate = true;
			flag = true;
		    }
		}
	    } catch (Exception E) {
		if(num_conds>2) {
		    System.out.print("\nThe level confidence must be a number strictly beween 0 and 1\n(or enter S to specify a separate confidence for each group\nor enter L to specify the confidence later).\nPlease re-enter it: ");
		}
		else {
		    System.out.print("\nThe level confidence must be a number strictly beween 0 and 1\n(or enter L to specify the confidence later).\nPlease re-enter it: ");
		}
		flag = false;
	    }
	}
	if(level_confidence_separate) {
	    for(int i=1; i<num_conds; i++) {
		System.out.print("Enter the level confidence for group " + i + ": ");
		br2 = new BufferedReader(new InputStreamReader(System.in));
		userinput = null;
		flag = false;
		while(flag == false) {
		    try {
			userinput = br2.readLine();
			level_confidence_list[i] = Double.parseDouble(userinput);
			if(level_confidence_list[i] > 0 && level_confidence_list[i] < 1) {
			    flag = true;
			}
			else {
			    System.out.print("\nThe level confidence must be a number strictly beween 0 and 1.\nPlease re-enter it: ");
			    flag = false;
			}
		    } catch (Exception E) {
			System.out.print("\nThe level confidence must be a number strictly beween 0 and 1.\nPlease re-enter it: ");
			flag = false;
		    }
		}
	    }
	    level_confidence_list_specified = true;
	}
    }

    public void CheckParameterCompatibilityAndDoFinalParameterInitialization() {
    	if(paired == true && design == 'D' && num_channels == 2) {
	    System.out.println("\nError: If the arrays are paired and the number of channels is two, then the design must be 'Reference'\n");
	    System.exit(0);
	}

	if(design == 'D') {
	    start = 0;
	}
	else {
	    start = 1;
	}

	if(tstat_tuning_parameter_specified) {
	    if(num_conds == 1) {
		alpha[0] = tstat_tuning_parameter;
	    }
	    else {
		for(int i=1; i<num_conds; i++) {
		    alpha[i] = tstat_tuning_parameter;
		}
	    }
	}

	if(level_confidence_specified && !level_confidence_list_specified && !level_confidence_later) {
	    if(num_conds == 1) {
		level_confidence_list[0] = level_confidence;
	    }
	    else {
		for(int i=1; i<num_conds; i++) {
		    level_confidence_list[i] = level_confidence;
		}
	    }
	}
	if(level_confidence_list_specified && design == 'D') {
	    System.out.println("\nError: This is a direct comparison, do not specify a level confidence list,\njust a single level confidence.\n\n");
	    System.exit(0);
	}
	int x = Array.getLength(level_confidence_list) - 1;
	if(num_conds != x + 1) {
	    int z = num_conds - 1;
	    if(x != 1) {
		System.out.println("\nError: You have " + num_conds + " conditions, so the level confidence list must have " + z + " entries, but yours has " + x + " entries\n\n");
	    }
	    else {
		System.out.println("\nError: You have " + num_conds + " conditions, so the level confidence list must have " + z + " entries, but yours has " + x + " entry\n\n");
	    }
	    System.exit(0);
	}

	if(min_presence_specified && !min_presence_list_specified) {
	    for(int i=0; i<num_conds; i++) {
		min_presence_list[i] = min_presence;
	    }
	}
	x = Array.getLength(min_presence_list) - 1;
	if(num_conds != x + 1) {
	    int z = num_conds - 1;
	    if(x != 1) {
		System.out.println("\nError: You have " + num_conds + " conditions, so the min presence list must have " + z + " entries, but yours has " + x + " entries\n\n");
	    }
	    else {
		System.out.println("\nError: You have " + num_conds + " conditions, so the min presence list must have " + z + " entries, but yours has " + x + " entry\n\n");
	    }
	    System.exit(0);
	}
	for(int i=0; i<num_conds; i++) {
	    if(min_presence_list[i]>num_reps[i]) {
		min_presence_list[i] = num_reps[i];
	    }
	}
	if(design == 'D' && use_logged_data == false) {
	    use_logged_data = true;
	    if(!silent_mode)
		System.out.println("\nWARNING: cannot use unlogged data with direct comparision design, we are going to use logged data\n");
	}
	if(paired) {
	    use_logged_data = true;
	    if(!silent_mode)
		System.out.println("\nWARNING: cannot use unlogged data with paired design, we are going to use logged data\n");
	}

	if(!silent_mode) {
	    System.out.print("------------------------\n");
	    System.out.println(num_channels + "-channel data expected");
	    if(design == 'D' && num_channels == 2) {
		System.out.println("Direct comparision design expected");
	    }
	    if(design == 'R' && num_channels == 2) {
		System.out.println("Reference design expected");
	    }
	    if(data_is_logged) {
		System.out.println("Logged data expected");
	    }
	    else {
		System.out.println("Unlogged data expected");
	    }
	    if(paired == true) {
		System.out.println("Paired data expected");
	    }
	    System.out.print("------------------------\n");
	}
    }

    public void WriteFirstStuffToScreen(String[] args) {

	int numArgs = Array.getLength(args);
	boolean help_flag = false;
	boolean id2info_flag = false;
	boolean id2url_flag = false;
	for(int arg = 0; arg<numArgs; arg++) {
	    if(args[arg].equals("--help") || args[arg].equals("--usage")) {
		help_flag = true;
	    }
	    if(args[arg].equals("--id2info")) {
		id2info_flag = true;
	    }
	    if(args[arg].equals("--id2url")) {
		id2url_flag = true;
	    }
	    if(args[arg].equals("--silent_mode")) {
		silent_mode = true;
	    }
	}
	if(!silent_mode) {
	    System.out.println("-------------------------------------------------------------------------------");
	    System.out.println("|                            Welcome to PaGE 5                                 |");
	    System.out.println("|                          microarray analysis tool                            |");
	    System.out.println("-------------------------------------------------------------------------------\n");
	    System.out.println("\n ****  PLEASE SET YOUR TERMINAL WINDOW TO AT LEAST 80 COLUMNS  ****\n");
	    if(id2info_flag == false) {
		System.out.println("\nInclude a tab delimited file mapping ID's to descipritons with\n      --id2info <filename>");
	    }
	    if(id2url_flag == false) {
		System.out.println("\nInclude a tab delimited file mapping ID's to URL's with\n      --id2url <filename>\n");
	    }
	}

	if(help_flag == true) {
	    WriteHelp();
	    System.exit(0);
	}
    }
    
    public void PrintParams() {
	System.out.println("vector analysis = " + vector_analysis);
	System.out.println("vector length = " + vector_length);
	System.out.println("t-stat tuning parameter = " + tstat_tuning_parameter);
	System.out.println("t-stat tuning parameter specified on command line = " + tstat_tuning_parameter_specified);
	int n = Array.getLength(alpha);
	System.out.print("alpha = " + alpha[0]);
	for(int i=1; i<n; i++) {
	    System.out.print("," + alpha[i]);
	}
	System.out.print("\n");
	System.out.print("alpha_up = " + alpha_up[0]);
	for(int i=1; i<n; i++) {
	    System.out.print("," + alpha_up[i]);
	}
	System.out.print("\n");
	System.out.print("alpha_down = " + alpha_down[0]);
	for(int i=1; i<n; i++) {
	    System.out.print("," + alpha_down[i]);
	}
	System.out.print("\n");
	System.out.print("alpha_default = " + alpha_default[0]);
	for(int i=1; i<n; i++) {
	    System.out.print("," + alpha_default[i]);
	}
	System.out.print("\n");
	System.out.println("aux page size = " + aux_page_size);
	System.out.println("silent mode = " + silent_mode);
	System.out.println("data is logged = " + data_is_logged);
	System.out.println("use logged data = " + use_logged_data);
	System.out.println("paired = " + paired);
	System.out.println("output_gene_confidence_list = " + output_gene_confidence_list);
	System.out.println("output_text = " + output_text);
	System.out.println("number of permutations = " + num_perms);
	System.out.println("number of permutations default = " + num_perms_default);
	System.out.println("number of bins = " + num_bins);
	System.out.println("number of bins default = " + num_bins_default);
	System.out.println("infile = " + infile);
	System.out.println("id2info = " + id2info_file);
	System.out.println("id2url = " + id2url_file);
	System.out.println("outfile = " + outfile);
	System.out.println("min presence = " + min_presence);
	n = 0;
	try {
	    n = Array.getLength(min_presence_list);
	    System.out.print("min presence list = " + min_presence_list[0]);
	    for(int i=1; i<n; i++) {
		System.out.print("," + min_presence_list[i]);
	    }
	    System.out.print("\n");
	} catch (Exception E) {
	    System.out.println("min_presence_list = undefined");
	}
	System.out.println("level_confidence_later = " + level_confidence_later);
	System.out.println("level_confidence = " + level_confidence);
	try {
	    n = Array.getLength(level_confidence_list);
	    System.out.print("level confidence list = " + level_confidence_list[0]);
	    for(int i=1; i<n; i++) {
		System.out.print("," + level_confidence_list[i]);
	    }
	    System.out.print("\n");
	} catch (Exception E) {
	    System.out.println("level_confidence_list = undefined");
	}
	System.out.println("note = " + note);
	System.out.println("num_channels = " + num_channels);
	System.out.println("design = " + design);
	System.out.println("missing_value_designator = " + missing_value_designator);
	System.out.println("start = " + start);
	System.out.println("num_conds = " + num_conds);	
	System.out.print("num_reps = " + num_reps[0]);	
	for(int i=1; i<num_conds; i++) {
	    System.out.print("," + num_reps[i]);
	}
	System.out.print("\n");
	System.out.println("max_num_reps = " + max_num_reps);
	System.out.println("num_cols = " + num_cols);
	System.out.println("printWarnings = " + printWarnings);
    }

    public void ReadHeader() {
	// the following line allows comment lines at the top of the file, those comment lines
	// must be preceded by a "#" at the beginning of the line

	String header = "";
	try {
	    FileInputStream inFile = new FileInputStream(infile);
	    BufferedReader br = new BufferedReader(new InputStreamReader(inFile));
	    int flag = 0;
	    header = br.readLine();
	    while(header.matches("#.*")) {
		header = br.readLine();
	    }
	    if(vector_analysis) {
		String firstDataLine = br.readLine();
		String[] firstDataLineArray = firstDataLine.split("\t", 0);
		String[] firstDataPoint = firstDataLineArray[1].split(",",0);
		vector_length = Array.getLength(firstDataPoint);
	    }
	}
	catch (Exception E) {
	    System.out.println("\nThere was an error reading the input file\n\n");
	    System.exit(0);
	}
	String[] header_array = header.split("\t", 0);
	num_cols = Array.getLength(header_array);

	// check if c1 is the only condition
	boolean c1flag = false;
	for(int i=1; i<num_cols; i++) {
	    String h = header_array[i];
	    h = h.trim();
	    if(!(h.equals("")) && !(h.equals("I")) && !(h.equals("i"))) {
		if(!h.matches("(c|C)1(r|R)\\d+")) {
		    c1flag = true;
		}
	    }
	}
	if(c1flag == false) { // in this case c1 is the only condition, this must be a direct
	                      // comparison, we will rename it c0
	    header = header.replaceAll("c1","c0");
	    header = header.replaceAll("C1","c0");
	    header_array = header.split("\t", 0);
	}
	
	// check that there is a condition 0 and finds the max numbered condition
	boolean there_is_a_condition_0 = false;
	int number_of_condition_with_highest_number = -1;
	int[][] conds_reps_array = new int[num_cols][num_cols];
	for(int i=1; i<num_cols; i++) {
	    String h = header_array[i];
	    h = h.trim();
	    if(!h.equals("") && !h.equals("I") && !h.equals("i") && !h.matches("(c|C)\\d+(r|R)\\d+")) {
		System.out.println("\nError: the header of column " + i + " is not of the right format.\n\n");
		System.exit(0);
	    }
	    if(!h.equals("I") && !h.equals("i") && !h.equals("")) {
		int cond=-1;
		int rep=-1;
		try {
		    String x = h.replaceAll("(r|R)\\d+", "");
		    cond = Integer.parseInt(x.replaceAll("(c|C)", ""));
		    String y = h.replaceAll("(c|C)\\d+", "");
		    rep = Integer.parseInt(y.replaceAll("(r|R)", ""));
		    if(rep == 0) {
			System.out.println("\nError: The header of column " + i + " is labeled as replicate 0,\nstart counting replicates at 1.\n\n");
			System.exit(0);
		    }
		    conds_reps_array[cond][rep]++;
		} catch (Exception E) {
		    System.out.println("\nError: The header of column " + i + " is not in the right format\n\n");
		    System.exit(0);
		}
		if(cond == 0) {
		    there_is_a_condition_0 = true;
		}
		if(cond > number_of_condition_with_highest_number) {
		    number_of_condition_with_highest_number = cond;
		}
	    }
	}
	if(!there_is_a_condition_0) {
	    System.out.println("\nError: You must have a condition 0\n\n");
	    System.exit(0);
	}
	// this loop checks that there are conditions from 0 to number_of_condition_with_highest_number
	boolean condition_numbering_flag = false;
	for(int i=1; i<number_of_condition_with_highest_number; i++) {
	    String matchstring = ".*\tc" + i + ".*";
	    if(!header.matches(matchstring)) {
		System.out.println("\nError: You have a condition labeled number " + number_of_condition_with_highest_number + ", but there is no condition " + i + "\n\n");
		System.exit(0);
	    }
	}
	int t = number_of_condition_with_highest_number + 1;
	if(!silent_mode) {
	    if(number_of_condition_with_highest_number+1 > 1) {
		System.out.println("\n   There are " + t + " conditions\n");
	    }
	    else {
		System.out.println("\n   There is one condition\n");
	    }
	}

	// this loop checks there are there are consecutively numbered reps for each condition,
	// and at least two of each.

	num_reps = new int[num_cols];
	max_num_reps = 0;
	for(int i=0; i<number_of_condition_with_highest_number+1; i++) {
	    num_reps[i] = 0;
	    for(int k=0; k<num_cols; k++) {
		if(conds_reps_array[i][k] > 0)
		    num_reps[i] = k;
		if(max_num_reps < num_reps[i])
		    max_num_reps = num_reps[i];
	    }
	    if(num_reps[i] == 1) {
		System.out.println("\nError: There is only one replicate for condition " + i + ",\nthere must be at least two replicates per condition.\n\n");
		System.exit(0);
	    }
	    for(int j=1; j< num_reps[i]+1; j++) {
		if(conds_reps_array[i][j] != 1) {
		    if(conds_reps_array[i][j] == 0) {
			System.out.println("\nError: Replicate " + j + " of condition " + i + " is missing,\nplease check the header format and restart.\n\n"); 
			System.exit(0);
		    }
		    else {
			System.out.println("\nError: Replicate " + j + " of condition " + i + " is replicated,\nplease check the header format and restart.\n\n"); 
			System.exit(0);
		    }
		}
	    }
	}
	if(paired == true) {
	    boolean wrong_num_reps_when_paired_flag = false;
	    int offending_i = 0;
	    for(int i=1; i<number_of_condition_with_highest_number+1; i++) {
		if(num_reps[i] != num_reps[0]) {
		    wrong_num_reps_when_paired_flag = true;
		    offending_i = i;
		}
	    }
	    if(wrong_num_reps_when_paired_flag) {
		System.out.println("\nError: For paired mode you must have the same number of replicates in each group.\nGroup 0 has " + num_reps[0] + " replicates while group " + offending_i + " has " + num_reps[offending_i] + " replicates.\n\n");
		System.exit(0);
	    }
	}

	num_conds = number_of_condition_with_highest_number+1;
	if(!level_confidence_list_specified) {
	    level_confidence_list = new double[num_conds];  // will need this later
	}
	if(!min_presence_list_specified) {
	    min_presence_list = new int[num_conds];  // will need this later
	}
	alpha = new double[num_conds];
	alpha_up = new double[num_conds];
	alpha_down = new double[num_conds];
	alpha_default = new double[num_conds];

	// The following makes the arrays COND, REP which map column position to condition and rep
	COND = new int[num_cols];
	REP = new int[num_cols];
	for(int i=1; i<num_cols; i++) {
	    String h = header_array[i];
	    if(!h.equals("I") && !h.equals("i") && !h.equals("")) {
		String x = h.replaceAll("(r|R)\\d+", "");
		int cond = Integer.parseInt(x.replaceAll("(c|C)", ""));
		String y = h.replaceAll("(c|C)\\d+", "");
		int rep = Integer.parseInt(y.replaceAll("(r|R)", ""));
		COND[i] = cond;
		REP[i] = rep;
	    }
	    else {
		COND[i] = -1;
		REP[i] = -1;
	    }
	}
	if(!silent_mode) {
	    for(int i=0; i<num_conds; i++) {
		System.out.println("   There are " + num_reps[i] + " replicates in condition " + i);
	    }
	}
	if(num_channels == 1 && num_conds == 1) {
	    System.out.println("\nError: If you have one channel data you must have at least two conditions.\n\n");
	    System.exit(0);
	}
	if(!silent_mode)
	    System.out.println();
    }

    public boolean get_vector_analysis() {
	return vector_analysis;
    }
    public int get_vector_length() {
	return vector_length;
    }
    public int get_start() {
	return start;
    }
    public int get_num_conds() {
	return num_conds;
    }
    public int[] get_num_reps() {
	return num_reps;
    }
    public int[] get_num_replicates() {
	return num_reps;
    }
    public int get_num_cols() {
	return num_cols;
    }
    public int[] get_COND() {
	return COND;
    }
    public int[] get_REP() {
	return REP;
    }
    public double get_tstat_tuning_parameter() {
	return tstat_tuning_parameter;
    }
    public int get_aux_page_size() {
	return aux_page_size;
    }
    public boolean get_silent_mode() {
	return silent_mode;
    }
    public boolean get_use_logged_data() {
	return use_logged_data;
    }
    public boolean get_data_is_logged() {
	return data_is_logged;
    }
    public boolean get_paired() {
	return paired;
    }
    public boolean get_output_gene_confidence_list() {
	return output_gene_confidence_list;
    }
    public boolean get_output_gene_confidence_list_combined() {
	return output_gene_confidence_list_combined;
    }
    public boolean get_output_text() {
	return output_text;
    }
    public int get_num_perms() {
	return num_perms;
    }
    public int get_num_bins() {
	return num_bins;
    }
    public File get_infile() {
	return infile;
    }
    public File get_id2info_file() {
	return id2info_file;
    }
    public Hashtable get_id2info_hash() {
	return id2info_hash;
    }
    public File get_id2url() {
	return id2url_file;
    }
    public File get_outfile() {
	return outfile;
    }
    public int get_min_presence() {
	return min_presence;
    }
    public int[] get_min_presence_list() {
	return min_presence_list;
    }
    public double get_level_confidence() {
	return level_confidence;
    }
    public double[] get_level_confidence_list() {
	return level_confidence_list;
    }
    public String get_note() {
	return note;
    }
    public int get_num_channels() {
	return num_channels;
    }
    public char get_design() {
	return design;
    }
    public int get_max_num_reps() {
	return max_num_reps;
    }
    public String get_missing_value_designator() {
	return missing_value_designator;
    }
    public void set_missing_value_designator(String s) {
	missing_value_designator = s;
    }
    public double[] get_alpha() {
	return alpha;
    }
    public double[] get_alpha_up() {
	return alpha_up;
    }
    public double[] get_alpha_down() {
	return alpha_down;
    }
    public double[] get_alpha_default() {
	return alpha_default;
    }
    public void set_alpha_default(double[] a) {
	alpha_default = a;
    }

    public void WriteHelp () {
	System.out.println("\n-------------------------------------------------------------------------------");
	System.out.println("|                                PaGE Help                                     |");
	System.out.println("-------------------------------------------------------------------------------");
	System.out.println("Note: it is not necessary to give any command line options, those not\nspecified command line will be requested by the program during execution,\nwith the exception of the (optional) commands which give the names of\nfiles which map id's to descriptions and url which must be input command\nline.  For the exact usage of these commands see below.\n-------------------------------------------------------------------------------");
	System.out.println("Options are specified using --option followed by the option value, if any\n(do not use an = sign).  For example:\n\n> PaGE_5.1.pl --infile input.txt --level_confidence .8\n--min_presence_list 3,5 --data_is_logged --note \"experiment 15\"\n-------------------------------------------------------------------------------");
	System.out.println("Data File format: A tab delimited table.\n");
	System.out.println("The first column gives the unique id.\n");
	System.out.println("There may be any number of comment rows at the top of the file, which\nmust start with a #");
	System.out.println("\nThe first non-comment row is the header row, with the following format:");
	System.out.println("For condition i and replicate j the header is \"cirj\".  For example with three\ngroups consisting of 3, 3, and 2 replicates, the header would be\n\nid\tc0r1\tc0r2\tc0r3\tc1r1\tc1r2\tc1r3\tc2r1\tc2r2.\n\n\"c\" stands for \"condition\"\n\"r\" stands for \"replicate\".\n\nConditions start counting at zero, replicates start counting at one.\n\nIf there is only a single direct comparision then there will only one condition\nnumbered zero.\n\nColumns can be labeled in any order.  To ignore a column of data leave\nthe header blank, or put an \"i\" (for \"ignore\")\n-------------------------------------------------------------------------------");
	System.out.println("|                                PaGE Commands                                 |\n-------------------------------------------------------------------------------");
	System.out.println("--help\n    If set, will show this help page.");
	System.out.println("--usage\n    Synonym for help.");
	System.out.println("------------------");
	System.out.println("| File locations |");
	System.out.println("------------------");
	System.out.println("--infile\n    Name of the input file containing the table of data.\n    This file must conform to the format in the README file.");
	System.out.println("--outfile\n    Optional. Name of the output file, if not specified outfile name will be\n    derived from the infile name.");
	System.out.println("--id2info\n    Optional. Name of the file containing a mapping of gene id's to names\n    or descriptions.");
	System.out.println("--id2url\n    Optional. Name ot the file containing a mapping of gene id's to urls.");
	System.out.println("------------------");
	System.out.println("| Output Options |");
	System.out.println("------------------");
	System.out.println("--output_gene_confidence_list\n    Optional.  Set this to output a tab delimited file that maps every gene to\n    its confidence of differential expression.  For each comparison gives\n    separate lists for up and down regulation.");
	System.out.println("--output_gene_confidence_list_combined\n    Optional.  Set this to output a tab delimited file that maps every gene to\n    its confidence of differential expression.  For each comparison gives one\n    list with up and down regulation combined.");
	System.out.println("--output_text\n    Optional.  Set this to output the results also in text format.");
	System.out.println("--note\n    Optional. A string that will be included at the top of the output file.");
	System.out.println("--aux_page_size\n    Optional.  A whole number greater than zero.  This specifies the minimum\n    number of tags there can be in one pattern before the results for that\n    pattern are written to an auxiliary page (this keeps the main results page\n    from getting too large).  This argument is optional, the default is 500.");
	System.out.println("---------------------------------------------");
	System.out.println("| Study Design and Nature of the Input Data |");
	System.out.println("---------------------------------------------");
	System.out.println("--num_channels\n    Is your data one or two channels?  (note: Affymetrix is considered one\n    channel).");
	System.out.println("--design\n    For two channel data, either set this to \"R\" for \"reference\" design,\n    or \"D\" for \"direct comparisons\" (see the documentation for more\n    information on this setting).");
	System.out.println("--data_is_logged\n    Use this option if your data has already been log transformed (T for true, F for false).");
	System.out.println("--paired\n    The data is paired (T for true, F for false).");
	System.out.println("--missing_value\n    If you have missing values designated by a string (such as \"NA\"), specify\n    that string with this option.  You can either put quotes around the string\n    or not, it doesn't matter as long as the string has no spaces.");
	System.out.println("-------------------------------------");
	System.out.println("| Statistics and Parameter Settings |");
	System.out.println("-------------------------------------");
	System.out.println("--level_confidence\n    A number between 0 and 1.  Generate the levels with this confidence.\n    See the README file for more information on this parameter.  This can\n    be set separately for each group using --level_confidence_list (see\n    below)\n    NOTE: This parameter can be set at the end of the run after the program has\n    displayed a summary breakdown of how many genes are found with what\n    confidence.  To do this either set the command line option to \"L\" (for\n    \"later\"), or do not specify this command line option and enter \"L\" when\n    the program prompts for the level confidence");
	System.out.println("--level_confidence_list\n    Comma-separated list of confidences.  If there are more than two\n    conditions (or more than one direct comparision), each position in the\n    pattern can have its own confidence specified by this list.  E.g. if\n    there are 4 conditions, the list might be .8,.7,.9 (note four conditions\n    gives patterns of length 3)");
	System.out.println("--min_presence\n    A positive integer specifying the minimum number of values a tag should\n    have in each condition in order to not be discarded.  This can be set\n    separately for each condition using --min_presence_list");
	System.out.println("--min_presence_list\n    Comma-separated list of positive integers, one for each condition,\n    specifying the minimum number of values a tag should have, for each\n    condition, in order not to be discarded.  E.g. if there are three\n    conditions, the list might be 4,6,3");
	System.out.println("--use_logged_data\n    Use this option to run the algorithm on the logged data (you can only\n    use this option if using the t-statistic as statistic).  Logging the\n    data usually give better results, but there is no rule.  Sometimes\n    different genes can be picked up either way.  It is generally best,\n    if using the t-statistic, to go with the logged data.  You might try\n    both ways and see if it makes much difference.  Both ways give valid\n    results, what can be effected is the power.");
	System.out.println("--use_unlogged_data\n    Use this option to run the algorithm on the unlogged data.  (See\n    --use_loggged_data option above for more information.)");
	System.out.println("--tstat_tuning_parameter\n    Optional.  The value of the t-statistic tuning parameter.  This is set to\n    a default value determined separately for each pattern position, but can be\n    set by hand using this command.  See the documentation for more\n    information on this parameter.");
	System.out.println("-----------------");
	System.out.println("| Configuration |");
	System.out.println("-----------------");
	System.out.println("--silent_mode\n    Optional. Do not output warning messages or progress to screen.");
	System.out.println("--num_permutations\n    Optional.  The number of permutations to use.  The default is to use all\n    or 200, whichever is smaller.  You might want to lower it to increase the\n    speed, though at a possible loss power or accuracy.  If the total number of\n    possible permutations is less than 25 more than the number requested by this\n    command, then the program will use the total number.");
	
	System.out.println("--num_bins\n    Optional.  The number of bins to use in granularizing the statistic over\n    its range.  This is set to a default of 1000 and you probably shouldn't\n    need to change it.");
}

    // CLASS VARIABLES

    private double tstat_tuning_parameter;
    private boolean tstat_tuning_parameter_specified;
    private double[] alpha;
    private double[] alpha_up;
    private double[] alpha_down;
    private double[] alpha_default;
    private int aux_page_size;
    private boolean silent_mode;
    private boolean use_logged_data;
    private boolean use_logged_data_specified;
    private boolean data_is_logged;
    private boolean data_is_logged_specified;
    private boolean paired;
    private boolean paired_specified;
    private boolean output_gene_confidence_list;
    private boolean output_gene_confidence_list_combined;
    private boolean output_text;
    private int num_perms;
    private int num_perms_default;
    private int num_bins;
    private int num_bins_default;
    private File infile;
    private File id2info_file;
    private Hashtable id2info_hash;
    private boolean id2info_specified;
    private File id2url_file;
    private Hashtable id2url_hash;
    private boolean id2url_specified;    
    private File outfile;
    private int min_presence;
    private int[] min_presence_list;
    private boolean min_presence_specified;
    private boolean min_presence_separate;
    private boolean min_presence_list_specified;
    private double level_confidence;
    private double[] level_confidence_list;
    private boolean level_confidence_specified;
    private boolean level_confidence_separate;
    private boolean level_confidence_later;
    private boolean level_confidence_list_specified;
    private String note;
    private int num_channels;
    private char design;
    private String missing_value_designator;
    private int start;
    private int num_conds;
    private int[] num_reps;
    private int max_num_reps;
    private int num_cols;
    private int[] COND;
    private int[] REP;
    private boolean printWarnings;
    private boolean vector_analysis;
    private int vector_length;
}
