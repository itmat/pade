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

public class PaGE {
    public static void main(String[] args) {
	theApp = new PaGE();
	Parameters parameters = new Parameters(args);
	Data data = new Data(parameters);
	if(!parameters.get_silent_mode()) 
	    parameters.PrintParams();
    }
    public static PaGE theApp;
}
