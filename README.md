PaGE: Patterns from Gene Expression
===================================

<a NAME="publications1"></a><font size=+1><a href="#publications2">Publications,
Errata Corrige and Notes</a></font></li>

<li>
<a NAME="software1"></a><font size=+1><a href="#software2">Download the software</a></font></li>

<li><font size=+1>View the HTML user guides</font>
<ul><li><a href="http://www.cbil.upenn.edu/PaGE/doc/perl/PaGE_5.1_documentation.html">Perl PaGE 5.1 documentatation</a>
</ul>

<li><font size=+1><a href="http://www.cbil.upenn.edu/PaGE/doc/PaGE_documentation_technical_manual.pdf">View the pdf technical manual</a></font>


</ul>

<font size=+2><b>Note: A Java version of PaGE is available now. Please check it out.</b></font>

<hr WIDTH="100%">
<br><a NAME="what2"><font size=+2><b>What is PaGE?</b></font>
<p><font size=+1>PaGE is free downloadable software
for microarray analisys.  PaGE
can be used to produce sets of differentially expressed genes with confidence measures attached.
These lists are generated <a href="http://www.cbil.upenn.edu/PaGE/fdr.html">the False Discovery Rate</a> method of controlling the
false positives.

<p><font size=+1>But PaGE is more than a differential expression analysis tool.
PaGE is a tool to attach <b>descriptive </b>, <b>dependable,</b>
and <b>easily interpretable</b> expression patterns to genes across multiple
conditions, each represented by a set of replicated array experiments.

<p><font size=+1>The input consists of (replicated) intensities from a
collection of array experiments from two or more conditions (or from
a collection of direct comparisons on 2-channel arrays).
The output consists of patterns, one for each row
identifier in the data file.

<p>One condition is used as a reference to which the other types are compared.
The length of a pattern equals the
number of non-reference sample types. The symbols in the patterns are integers,
where positive integers represent up-regulation as compared to the reference
sample type and negative integers represent down-regulation.

<p>The patterns are based on the false discovery rates for each position
in the pattern, so that the number
of positive and negative symbols that appear in each position of the pattern
is as descriptive as the data variability allows.</font>

<p>The patterns generated are easily interpretable
in that integers are used to represent different levels of up- or
down-regulation as compared to the reference sample type.

<p> To illustrate this,&nbsp; the following table gives an excerpt of
data for four of the gene tags in a given of hybridization experiment
and four sample types.&nbsp; There are three replicates for sample types
G<sub>0</sub> and G<sub>2</sub> and two replicates for sample types G<sub>1</sub>
and G<sub>3</sub>. As they are these data are hard to peruse for information.</font>
<br>&nbsp;
<table BORDER COLS=11 WIDTH="75%" BGCOLOR="#FFFFFF" NOSAVE >
<tr NOSAVE>
<td NOSAVE>
<center><b><font size=+1>gene tag</font></b></center>
</td>

<td BGCOLOR="#FFFF99" NOSAVE>
<center><b><font color="#000000"><font size=+1>G<sub>0</sub> I</font></font></b></center>
</td>

<td BGCOLOR="#FFFF99" NOSAVE>
<center><b><font color="#000000"><font size=+1>G<sub>0</sub> II</font></font></b></center>
</td>

<td BGCOLOR="#FFFF99" NOSAVE>
<center><b><font color="#000000"><font size=+1>G<sub>0</sub> III</font></font></b></center>
</td>

<td BGCOLOR="#CCFFFF" NOSAVE>
<center><b><font size=+1>G<sub>1</sub> I</font></b></center>
</td>

<td BGCOLOR="#CCFFFF" NOSAVE>
<center><b><font size=+1>G<sub>1</sub> II</font></b></center>
</td>

<td BGCOLOR="#99FFCC" NOSAVE>
<center><b><font size=+1>G<sub>2 </sub>I</font></b></center>
</td>

<td BGCOLOR="#99FFCC" NOSAVE>
<center><b><font size=+1>G<sub>2</sub> II</font></b></center>
</td>

<td BGCOLOR="#99FFCC" NOSAVE>
<center><b><font size=+1>G<sub>2</sub> III</font></b></center>
</td>

<td BGCOLOR="#FFCCFF" NOSAVE>
<center><b><font size=+1>G<sub>3</sub> I</font></b></center>
</td>

<td BGCOLOR="#FFCCFF" NOSAVE>
<center><b><font size=+1>G<sub>3</sub> II</font></b></center>
</td>
</tr>

<tr>
<td>
<center><b><font size=+1>1</font></b></center>
</td>

<td BGCOLOR="#FFFF99" NOSAVE>
<center><font size=+1>0.0114</font></center>
</td>

<td BGCOLOR="#FFFF99" NOSAVE>
<center><font size=+1>0.0328</font></center>
</td>

<td BGCOLOR="#FFFF99" NOSAVE>
<center><font size=+1>0.0151</font></center>
</td>

<td BGCOLOR="#CCFFFF" NOSAVE>
<center><font size=+1>0.0060</font></center>
</td>

<td BGCOLOR="#CCFFFF" NOSAVE>
<center><font size=+1>0.0236</font></center>
</td>

<td BGCOLOR="#99FFCC" NOSAVE>
<center><font size=+1>0.0436</font></center>
</td>

<td BGCOLOR="#99FFCC" NOSAVE>
<center><font size=+1>0.5640</font></center>
</td>

<td BGCOLOR="#99FFCC" NOSAVE>
<center><font size=+1>0.8920</font></center>
</td>

<td BGCOLOR="#FFCCFF" NOSAVE>
<center><font size=+1>0.0639</font></center>
</td>

<td BGCOLOR="#FFCCFF" NOSAVE>
<center><font size=+1>0.2490</font></center>
</td>
</tr>

<tr>
<td>
<center><b><font size=+1>2</font></b></center>
</td>

<td BGCOLOR="#FFFF99" NOSAVE>
<center><font size=+1>0.0050</font></center>
</td>

<td BGCOLOR="#FFFF99" NOSAVE>
<center><font size=+1>0.0131</font></center>
</td>

<td BGCOLOR="#FFFF99" NOSAVE>
<center><font size=+1>0.0061</font></center>
</td>

<td BGCOLOR="#CCFFFF" NOSAVE>
<center><font size=+1>0.0041</font></center>
</td>

<td BGCOLOR="#CCFFFF" NOSAVE>
<center><font size=+1>0.0364</font></center>
</td>

<td BGCOLOR="#99FFCC" NOSAVE>
<center><font size=+1>0.0296</font></center>
</td>

<td BGCOLOR="#99FFCC" NOSAVE>
<center><font size=+1>0.8830</font></center>
</td>

<td BGCOLOR="#99FFCC" NOSAVE>
<center><font size=+1>0.7000</font></center>
</td>

<td BGCOLOR="#FFCCFF" NOSAVE>
<center><font size=+1>0.0199</font></center>
</td>

<td BGCOLOR="#FFCCFF" NOSAVE>
<center><font size=+1>0.1050</font></center>
</td>
</tr>

<tr>
<td>
<center><b><font size=+1>3</font></b></center>
</td>

<td BGCOLOR="#FFFF99" NOSAVE>
<center><font size=+1>0.0629</font></center>
</td>

<td BGCOLOR="#FFFF99" NOSAVE>
<center><font size=+1>0.2340</font></center>
</td>

<td BGCOLOR="#FFFF99" NOSAVE>
<center><font size=+1>0.0431</font></center>
</td>

<td BGCOLOR="#CCFFFF" NOSAVE>
<center><font size=+1>0.2270</font></center>
</td>

<td BGCOLOR="#CCFFFF" NOSAVE>
<center><font size=+1>0.2120</font></center>
</td>

<td BGCOLOR="#99FFCC" NOSAVE>
<center><font size=+1>0.0105</font></center>
</td>

<td BGCOLOR="#99FFCC" NOSAVE>
<center><font size=+1>0.1400</font></center>
</td>

<td BGCOLOR="#99FFCC" NOSAVE>
<center><font size=+1>0.0243</font></center>
</td>

<td BGCOLOR="#FFCCFF" NOSAVE>
<center><font size=+1>0.0117</font></center>
</td>

<td BGCOLOR="#FFCCFF" NOSAVE>
<center><font size=+1>0.0907</font></center>
</td>
</tr>

<tr NOSAVE>
<td>
<center><b><font size=+1>4</font></b></center>
</td>

<td BGCOLOR="#FFFF99" NOSAVE>
<center><font size=+1>0.0250</font></center>
</td>

<td BGCOLOR="#FFFF99" NOSAVE>
<center><font size=+1>0.0600</font></center>
</td>

<td BGCOLOR="#FFFF99" NOSAVE>
<center><font size=+1>0.0264</font></center>
</td>

<td BGCOLOR="#CCFFFF" NOSAVE>
<center><font size=+1>0.1500</font></center>
</td>

<td BGCOLOR="#CCFFFF" NOSAVE>
<center><font size=+1>0.2660</font></center>
</td>

<td BGCOLOR="#99FFCC" NOSAVE>
<center><font size=+1>0.0134</font></center>
</td>

<td BGCOLOR="#99FFCC" NOSAVE>
<center><font size=+1>0.1860</font></center>
</td>

<td BGCOLOR="#99FFCC" NOSAVE>
<center><font size=+1>0.0851</font></center>
</td>

<td BGCOLOR="#FFCCFF" NOSAVE>
<center><font size=+1>0.0172</font></center>
</td>

<td BGCOLOR="#FFCCFF" NOSAVE>
<center><font size=+1>0.0112</font></center>
</td>
</tr>

<caption ALIGN=BOTTOM>&nbsp;</caption>
</table>
<font size=+1>If G<sub>0 </sub>is used as a reference sample type, the
patterns attached by PaGE to these tags might look like</font>
<center><table BORDER COLS=4 NOSAVE >
<tr NOSAVE>
<td WIDTH="50" NOSAVE>
<center><b><font size=+1>gene tag</font></b></center>
</td>

<td WIDTH="50" BGCOLOR="#CCFFFF" NOSAVE>
<center><b><font size=+1>G<sub>1</sub></font></b></center>
</td>

<td WIDTH="50" BGCOLOR="#99FFCC" NOSAVE>
<center><b><font size=+1>G<sub>2</sub></font></b></center>
</td>

<td WIDTH="50" BGCOLOR="#FFCCFF" NOSAVE>
<center><b><font size=+1>G<sub>3</sub></font></b></center>
</td>
</tr>

<tr>
<td>
<center><b><font size=+1>1</font></b></center>
</td>

<td BGCOLOR="#CCFFFF" NOSAVE>
<center><font size=+1>0</font></center>
</td>

<td BGCOLOR="#99FFCC" NOSAVE>
<center><font size=+1>7</font></center>
</td>

<td BGCOLOR="#FFCCFF" NOSAVE>
<center><font size=+1>2</font></center>
</td>
</tr>

<tr>
<td>
<center><b><font size=+1>2</font></b></center>
</td>

<td BGCOLOR="#CCFFFF" NOSAVE>
<center><font size=+1>0</font></center>
</td>

<td BGCOLOR="#99FFCC" NOSAVE>
<center><font size=+1>8</font></center>
</td>

<td BGCOLOR="#FFCCFF" NOSAVE>
<center><font size=+1>1</font></center>
</td>
</tr>

<tr NOSAVE>
<td>
<center><b><font size=+1>3</font></b></center>
</td>

<td BGCOLOR="#CCFFFF" NOSAVE>
<center><font size=+1>2</font></center>
</td>

<td WIDTH="50%" BGCOLOR="#99FFCC" NOSAVE>
<center><font size=+1>-1</font></center>
</td>

<td BGCOLOR="#FFCCFF" NOSAVE>
<center><font size=+1>-1</font></center>
</td>
</tr>

<tr NOSAVE>
<td>
<center><b><font size=+1>4</font></b></center>
</td>

<td BGCOLOR="#CCFFFF" NOSAVE>
<center><font size=+1>3</font></center>
</td>

<td BGCOLOR="#99FFCC" NOSAVE>
<center><font size=+1>1</font></center>
</td>

<td ALIGN=CENTER VALIGN=CENTER WIDTH="50" BGCOLOR="#FFCCFF" NOSAVE>
<center><font size=+1>0</font></center>
</td>
</tr>
</table></center>

<p><font size=+1>this is an easily interpretable set of patterns. For example
gene tag 3 is detected as up-regulated 2 levels in sample type G<sub>1</sub>
and down-regulated one level in sample types G<sub>2</sub> and G<sub>3</sub>,
as compared to sample type G<sub>0</sub>.</font>

<p><a NAME="publications2"><font size=+2><b>Publications, Errata Corrige and Notes</b></font>

<p><font color="#009900"><font size=+1><b>* Grant G.R., Liu J., Stoeckert C.J.Jr.</b> (2005) A practical false discovery rate approach to identifying patterns of differential expression in microarray data, <i>Bioinformatics</i>, Vol 21 no 11, 2684-2690.</font>

<p>
<table border=1><tr><td><font color=red><b>Note: There is a typo in this publication on page 2686 in formula for mu<sub>k</sub>(i+1) (the second to last displayed forumla on the page).  On the right hand side, the first mu<sub>k</sub>(i) should be  mu-tilde<sub>k</sub>(1) (i replaced by 1).</font></b></td></tr></table>

<p><font color="#009900"><font size=+1><b>* Grant G.R., Liu J., Stoeckert C.J.Jr.</b>  <a href="http://www.cbil.upenn.edu/PaGE/doc/PaGE_documentation_technical_manual.pdf">The technical manual for PaGE 5.1</a>.</font>

<p><font color="#009900"><font size=+1><b>* Grant G.R., Manduchi E., Stoeckert C.J. Jr.</b>  <A HREF="http://www.cbil.upenn.edu/PaGE/camda.pdf">Using non-parametric methods in the context of multiple testing to identify differentially expressed genes</A>. <i>Methods of microarray data analysis</i>, editors
S.M. Lin and K.F. Johnson, Kluwer Academic Publishers (Boston, 2002):
37-55. (Winner of the best presentation award <A
HREF=http://www.bioinformatics.duke.edu/camda>CAMDA'00</A>).

<p><font color="#009900"><font size=+1><b>* Manduchi E., Grant G.R., McKenzie
S.E., Overton G.C., Surrey S., Stoeckert C.J. Jr.</b> (2000) <A HREF="http://bioinformatics.oupjournals.org/cgi/reprint/16/8/685">Generation
of patterns from gene expression data by assigning confidence to differentially
expressed genes</A>,&nbsp; <i>Bioinformatics</i>, <b>16(8)</b>: 685-698.</font></font>
<p>&nbsp;&nbsp;<font size=+1><a href="http://www.cbil.upenn.edu/PaGE/errata.html">Errata Corrige and Notes to the above original paper</a></font>



<font color=black>

<br>&nbsp;<br>

<table cellpadding=4 border>
<tr bgcolor=beige><td>
<p><a NAME="software2"><font size=+2><b>Software</b></font></p>
<p><font size=+1>To download the perl version of the PaGE 5.1 software <a href="http://www.cbil.upenn.edu/PaGE/licensedcode.html">click here</a>. This is the latest stable release of PaGE and presents several improvements as compared to the previous release (4.0).  Besides bug fixes, it offers a more informative and richer output and at the same time it is easier to use as it has less mandatory options.</p>
</td>
</tr>
</table>

If you have questions on the program or its usage or if you want to report any bugs, please contact: <tt><a href="mailto:ggrant@pcbi.upenn.edu">ggrant@pcbi.upenn.edu</a></tt>.</font>

-----

## Install PADE on your mac
1. Make sure you have the clang version of Python. You can 
		brew install python
