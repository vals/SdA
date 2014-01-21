## Dimensionality reduction by Stacked denoising Autoencoders

I wanted to quickly try using deep autoencoders for unsupervised
dimensionality reduction, but couldn't find a small piece of software which
did precisely this.

For this, I took the code from http://deeplearning.net/tutorial/ and modified
it in order to just get a simple runnable script which takes csv file as input
and outputs a csv table of reduced dimensiona that can be used to find
clusters.

The code has some heavy dependencies; numpy, theano, pandas. But should run
fairly quickly, and supposedly even works on GPUs.

If the dependencies are installed, just run `python setup.py install` and you
will get a runnable `sda_script.py` on your system.

Number of hidden layers for the stacked autoencoders are given as a list.

### Example

The following takes a csv file with 2000 variables in 40 samples and reduces
it to two dimensions in a third hidden layer. The intermediate layer sizes are
speciefied to be 500-dimensional and 200-dimensional.

The final output of the script is sent to stdout so you can pipe it to a file for further analysis.

The first column are the index labels of the data, the following columns the hidden layer values.

	$ sda_script.py sim_data.csv -l 500 200 2
	INFO:root:Loading data
	INFO:root:Building model
	INFO:root:2000 -> 500 -> 200 -> 2
	INFO:root:Compiling training functions
	INFO:root:Training model
	INFO:root:Training layer 0, epoch 0, cost -11196.7965486
	INFO:root:Training layer 0, epoch 1, cost -34128.0673827
	INFO:root:Training layer 0, epoch 2, cost -56897.8298813
	INFO:root:Training layer 0, epoch 3, cost -79542.7872575
	INFO:root:Training layer 0, epoch 4, cost -102113.575492
	INFO:root:Training layer 0, epoch 5, cost -124658.995912
	INFO:root:Training layer 0, epoch 6, cost -147196.730773
	INFO:root:Training layer 0, epoch 7, cost -169735.411333
	INFO:root:Training layer 0, epoch 8, cost -192267.104028
	INFO:root:Training layer 0, epoch 9, cost -214794.281903
	INFO:root:Training layer 0, epoch 10, cost -237317.183181
	INFO:root:Training layer 0, epoch 11, cost -259839.352664
	INFO:root:Training layer 0, epoch 12, cost -282359.170618
	INFO:root:Training layer 0, epoch 13, cost -304878.188668
	INFO:root:Training layer 0, epoch 14, cost -327395.845681
	INFO:root:Training layer 1, epoch 0, cost 412.929671567
	INFO:root:Training layer 1, epoch 1, cost 391.038150226
	INFO:root:Training layer 1, epoch 2, cost 372.103678152
	INFO:root:Training layer 1, epoch 3, cost 356.582619491
	INFO:root:Training layer 1, epoch 4, cost 339.193018507
	INFO:root:Training layer 1, epoch 5, cost 325.586034019
	INFO:root:Training layer 1, epoch 6, cost 312.23857715
	INFO:root:Training layer 1, epoch 7, cost 301.686553806
	INFO:root:Training layer 1, epoch 8, cost 288.876326897
	INFO:root:Training layer 1, epoch 9, cost 278.850766161
	INFO:root:Training layer 1, epoch 10, cost 269.182005148
	INFO:root:Training layer 1, epoch 11, cost 260.147499359
	INFO:root:Training layer 1, epoch 12, cost 252.054445182
	INFO:root:Training layer 1, epoch 13, cost 241.759709867
	INFO:root:Training layer 1, epoch 14, cost 235.992144249
	INFO:root:Training layer 2, epoch 0, cost 141.575501386
	INFO:root:Training layer 2, epoch 1, cost 140.868139161
	INFO:root:Training layer 2, epoch 2, cost 141.180940729
	INFO:root:Training layer 2, epoch 3, cost 140.879025994
	INFO:root:Training layer 2, epoch 4, cost 141.224421003
	INFO:root:Training layer 2, epoch 5, cost 140.530103461
	INFO:root:Training layer 2, epoch 6, cost 140.501677621
	INFO:root:Training layer 2, epoch 7, cost 139.894249461
	INFO:root:Training layer 2, epoch 8, cost 140.16130021
	INFO:root:Training layer 2, epoch 9, cost 139.907166108
	INFO:root:Training layer 2, epoch 10, cost 139.32763202
	INFO:root:Training layer 2, epoch 11, cost 139.602953774
	INFO:root:Training layer 2, epoch 12, cost 138.986554787
	INFO:root:Training layer 2, epoch 13, cost 139.030902805
	INFO:root:Training layer 2, epoch 14, cost 138.642359314
	0	0.321563889657	0.494585635544
	1	0.0105997037281	0.808728606594
	2	0.32156388967	0.494585635535
	3	0.32156388967	0.494585635535
	4	0.32156388967	0.494585635535
	5	0.0276066026042	0.695143787708
	6	0.0105997037281	0.808728606594
	7	0.0276066026042	0.695143787707
	8	0.32156388967	0.494585635535
	9	0.0105997037281	0.808728606594
	10	0.0276066026042	0.695143787708
	11	0.0105997037281	0.808728606594
	12	0.0276066026042	0.695143787707
	13	0.32156388967	0.494585635535
	14	0.0276066026042	0.695143787707
	15	0.010599703728	0.808728606594
	16	0.0276066025818	0.695143787743
	17	0.0105997037281	0.808728606594
	18	0.32156388967	0.494585635535
	19	0.0105997037281	0.808728606594
	20	0.0105997037281	0.808728606594
	21	0.32156388967	0.494585635535
	22	0.32156388967	0.494585635535
	23	0.0276066026042	0.695143787708
	24	0.0276066026042	0.695143787708
	25	0.0276066026042	0.695143787707
	26	0.010599703728	0.808728606594
	27	0.0276066026042	0.695143787707
	28	0.0105997037281	0.808728606594
	29	0.32156388967	0.494585635535
	30	0.0276066026042	0.695143787708
	31	0.321563889671	0.494585635535
	32	0.0105997037281	0.808728606594
	33	0.0105997037281	0.808728606594
	34	0.32156388967	0.494585635535
	35	0.0276066026042	0.695143787707
	36	0.32156388967	0.494585635535
	37	0.0105997037281	0.808728606594
	38	0.0105997037281	0.808728606594
	39	0.0276066026042	0.695143787707
