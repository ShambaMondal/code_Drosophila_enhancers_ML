import sys
sys.path.append("/home/Sam/Yoda_2TB/next_gen_analysis_softwares/boruta_py-master")
import pandas
from sklearn.ensemble import RandomForestClassifier
#from boruta_py import boruta_py
import boruta_py
#import numpy as np
from numpy import array, ravel

# load X and y
#X = pd.read_csv('my_X_table.csv', index_col=0).values
#y = pd.read_csv('my_y_vector.csv', index_col=0).values

#setting the data in patients
text = open("TF_matrix_file_18022016").read()
data_all = eval(text)
X = array([array(d[1:119]) for d in data_all[1:]])

#y = array([d[119] for d in data[1:]]) #for CA279
#y = array([sum(d[119:141])>0 for d in data[1:]]) # all CAs
#y = array([sum(d[141:158])>0 for d in data[1:]]) # all CZDs
#y = array([sum(d[158:171])>0 for d in data[1:]]) # all GBMs
#y = array([sum(d[171:])>0 for d in data[1:]]) # all_PAs
#y = array([(sum(d[119:141]) + sum(d[158:171]))>0 for d in data[1:]]) # for all GBM CLASS (CA + GBM)
#y = array([(sum(d[141:158]) + sum(d[171:]))>0 for d in data[1:]]) # for all PA CLASS (CZD + PA)
#y = array([sum(d[119:])>0 for d in data[1:]]) # for BOTH GBM/PA CLASS (CA+GBM+CZD+PA) # DOES NOT make sense. Becuase in that case, all y-label will be "True" and no "False" label will exist.
#print y
#print len(y)

def run_boruta(x_obj, y_obj):
	# define random forest classifier, with utilising all cores and
	# sampling in proportion to y labels
	rf = RandomForestClassifier(n_jobs=-1, class_weight='auto', max_depth=5)

	# define Boruta feature selection method
	feat_selector = boruta_py.BorutaPy(rf, n_estimators='auto', verbose=2)

	# find all relevant features
	#feat_selector.fit(X, y)
        feat_selector.fit(x_obj, y_obj)

	# check selected features
	selected = feat_selector.support_
	#print selected

	# check ranking of features
	ranks = feat_selector.ranking_
	#print ranks

	# call transform() on X to filter it down to selected features
	X_filtered = feat_selector.transform(X)
	#for j in range(20):
	#	print  X_filtered[j][0:]
	return ranks

def run_for_each_patient(dat):  # from and for the mutation project
	data = dat
	global X
        tf_names = ravel(data[0][1:119])
	patients = ravel(data[0][119:])
	patient_y_dict = {}
	for p in range(len(patients)):
	    p_key = "y_"+patients[p]
	    p_val = array([d[p+119] for d in data[1:]])
	    patient_y_dict.update({p_key:p_val})
        #patient_y_dict = {"y_CA279": array([d[119] for d in data[1:]]), "y_CA280": array([d[120] for d in data[1:]])}
	for  p in sorted(patient_y_dict.keys()):
	    #print p
            sys.stderr.write(p+"\n")
	    f_ranks = run_boruta(X, patient_y_dict[p])
            #print f_ranks
            sys.stderr.write(str(f_ranks)+"\n")
            confirmed = []
            tentative = []
            for k in range(118):
                if f_ranks[k] == 1: #or ranks[k] == 2:
                    confirmed.append(tf_names[k])
                    #print "confirmed: ", tf_names[k]
                elif f_ranks[k] == 2:
                    tentative.append(tf_names[k])
                    #print "tentative: ", tf_names[k]
            sys.stderr.write("confirmed:%s "%(str(len(confirmed))) + str(confirmed) + "\n")
            sys.stderr.write("tentative:%s "%(str(len(tentative))) + str(tentative) + "\n")
        return

if __name__ == "__main__":
	run_for_each_patient(data_all)
