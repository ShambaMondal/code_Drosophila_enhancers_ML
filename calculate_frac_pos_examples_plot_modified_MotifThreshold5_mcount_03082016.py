import sys
sys.path.append("/home/Sam/Yoda_2TB/Sam_Work_Nencki/2016_Jan_Drosophila_project/analysis_on_the_rocks")
sys.path.append("/home/Sam/Yoda_2TB/next_gen_analysis_softwares/boruta_py-master")
from activator_repressor_pills_long_fragments_dict import activator_repressor_pills_long_fragments

from sklearn import ensemble
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from matplotlib import pyplot as plt
from numpy import array, mean, std, arange
from math import log
import boruta_py

#print activator_repressor_pills_long_fragments
feature_matrix = "/home/Sam/Yoda_2TB/Sam_Work_Nencki/2016_Jan_Drosophila_project/analysis_on_the_rocks/InR_feature_matrix_MotifThreshold_5_03082016"
text = open(feature_matrix).read()
data = eval(text)

cells = ["Kc", "S2", "Both"]
pills = {'Gen_A': 'GA', 'Gen_R': 'GR', '20E_A': '20EA', '20E_R': '20ER', 'dFOXO_A': 'dFOXOA', 'dFOXO_R': 'dFOXOR'}

def expected_accuracy(x):
	exp_acc = 2 *(x**2) - 2*x + 1
	return exp_acc

def log_odds(p): # 0 < p < 1
	log_odds = log(p/(1-p))
	return log_odds

def calculate_fraction_pos_examples():
	d = activator_repressor_pills_long_fragments
	p_dict = {}
	exp_acc_p_dict = {}
	log_odds_dict = {}
	row_keys = range(1,26)
	#cells = ["Kc", "S2"]
	#pills = ['Gen_A', 'Gen_R', '20E_A', '20E_R', 'dFOXO_A', 'dFOXO_R']

	for j in pills.keys():
		for c in cells:
			p = (25 - [d[i][c][j] for i in row_keys].count(0))/25.0
			#print p
			p_dict.update({"Y_"+ c + "_"+ pills[j] : p})
			#print expected_accuracy(p)
			exp_acc_p_dict.update({"Y_"+ c + "_"+ pills[j] : expected_accuracy(p)})
			#log_odds_dict.update({"Y_"+ c + "_"+ pills[j] : log_odds(p)})
	#tot_neg = 0
	for c in cells:
		neg = [sum(d[i][c].values()) for i in row_keys].count(0)
		#print "negs", neg, (25-neg)
		p_c = (25 - neg)/25.0
		p_dict.update({"Y_"+ c : p_c})
		exp_acc_p_dict.update({"Y_"+ c : expected_accuracy(p_c)})
		#log_odds_dict.update({"Y_"+ c : log_odds(p_c)})
		###tot_neg = tot_neg + neg

	p_all_neg = [sum(d[i]["Kc"].values())+sum(d[i]["S2"].values()) for i in row_keys].count(0)  #(300 - tot_neg)/300.0
	#print "p_all_neg ", p_all_neg
	p_all = (25 - p_all_neg)/25.0
	#print "p_all ", p_all
	exp_acc_p_all = expected_accuracy(p_all)
	#log_odds_p_all = log_odds(p_all)
	p_dict.update({"Y_All" : p_all})
	exp_acc_p_dict.update({"Y_All" : exp_acc_p_all})
	#log_odds_dict.update({"Y_All" : log_odds_p_all})

	return p_dict, exp_acc_p_dict #, log_odds_dict

def get_crossvalidation_score(x_obj, y_obj):
	#print "x_obj: ", x_obj, "y_obj: ", y_obj
	rf = ensemble.RandomForestClassifier(n_estimators=500)
	#rf = ensemble.RandomForestClassifier(n_estimators=500, max_depth=5) # 19042016: testing if limiting max_depth enhances scores.
	cvs = cross_val_score(rf, x_obj, y_obj)
	#print cvs
	return cvs.mean()

def get_NON_crossval_accuracy_score(x_obj, y_obj):
	#print "x_obj: ", x_obj, "y_obj: ", y_obj
	rf = ensemble.RandomForestClassifier(n_estimators=500)
	#rf = ensemble.RandomForestClassifier(n_estimators=500, max_depth=5) #19042016: testing if limiting max_depth enhances scores.
	rf.fit(x_obj, y_obj)
	z = rf.predict(x_obj)
	score = metrics.accuracy_score(y_obj, z)
	#print score
	return score

def return_objects():
	#X_4mers = array([d[1:137] for d in data[1:]])
	#X_3mers = array([d[137:169] for d in data[1:]])
	X_objs = {
	"norm_X_4mers" : array([array(d[1:137],float)/sum(d[1:137]) for d in data[1:]]),
	"norm_X_3mers" : array([array(d[137:169], float)/sum(d[137:169]) for d in data[1:]]),
	"norm_X_mcount" : array([array(d[169:568:3],float)/sum(d[169:568:3]) for d in data[1:]]),
	"X_mmax" : array([d[170:568:3] for d in data[1:]]),
	"norm_X_msum" : array([array(d[171:568:3], float)/sum(d[171:568:3]) for d in data[1:]])}
	
	# Originally used Y_objs:
	Y_objs = {
	"Y_All" : array([sum(d[-12:])>0 for d in data[1:]]),
	"Y_Kc" : array([sum(d[-12:-6])>0 for d in data[1:]]),
	"Y_S2" : array([sum(d[-6:])>0 for d in data[1:]]),
	"Y_Kc_GA" : array([d[-12]>0 for d in data[1:]]),
	"Y_Kc_GR" : array([d[-11]>0 for d in data[1:]]),
	"Y_Kc_20EA" : array([d[-10]>0 for d in data[1:]]),
	"Y_Kc_20ER" : array([d[-9]>0 for d in data[1:]]),
	"Y_Kc_dFOXOA" : array([d[-8]>0 for d in data[1:]]),
	"Y_Kc_dFOXOR" : array([d[-7]>0 for d in data[1:]]),
	"Y_S2_GA" : array([d[-6]>0 for d in data[1:]]),
	"Y_S2_GR" : array([d[-5]>0 for d in data[1:]]),
	"Y_S2_20EA" : array([d[-4]>0 for d in data[1:]]),
	"Y_S2_20ER" : array([d[-3]>0 for d in data[1:]]),
	"Y_S2_dFOXOA" : array([d[-2]>0 for d in data[1:]]),
	"Y_S2_dFOXOR" : array([d[-1]>0 for d in data[1:]])}
	
	"""
	#This is an extension. To merge Both Kc and S2 cells' specifications. GA/GR/20EA/20ER/dFOXOA/dFOXOR irrespective of cell types.Done on 21042016
	Y_objs = {
	"Y_Both_GA" : array([(d[-12]+d[-6])>0 for d in data[1:]]),
	"Y_Both_GR" : array([(d[-11]+d[-5])>0 for d in data[1:]]),
	"Y_Both_20EA" : array([(d[-10]+d[-4])>0 for d in data[1:]]),
	"Y_Both_20ER" : array([(d[-9]+d[-3])>0 for d in data[1:]]),
	"Y_Both_dFOXOA" : array([(d[-8]+d[-2])>0 for d in data[1:]]),
	"Y_Both_dFOXOR" : array([(d[-7]+d[-1])>0 for d in data[1:]])}
	"""
	return X_objs, Y_objs

def run_on_data_for_Scores():
	#classes = {"4mers": {} , "3mers": {}, "mcount":{} , "mmax":{} , "msum": {}}
	X_objs, Y_objs = return_objects()
	final_scores_dict = {}
	#final_score_keys = []
	final_x_keys = ["pos_frac", "exp_acc", "4mers", "3mers", "mcount", "mmax", "msum"]
	#final_x_keys = ["pos_frac", "exp_acc", "4mers_acc", "4mers_crossval", "3mers_acc", "3mers_crossval", "mcount_acc", "mcount_crossval", "mmax_acc", "mmax_crossval", "msum_acc", "msum_crossval"]
	final_scores_dict_keys = Y_objs.keys()
	for k in final_scores_dict_keys:
		final_scores_dict.update({k : {} })
	for k in final_scores_dict.keys():
		for m in final_x_keys:
			final_scores_dict[k].update({m : []})
			#if not k in final_scores_dict.keys():
			#	final_scores_dict.update({k : m })
			#else:
			#	final_scores_dict[k].update({m : []})
	
	p_score_dict, exp_acc_p_score_dict  = calculate_fraction_pos_examples()
	
	for Y in Y_objs:
		final_scores_dict[Y]["pos_frac"].append(p_score_dict[Y])
		final_scores_dict[Y]["exp_acc"].append(exp_acc_p_score_dict[Y])
		for X in X_objs:
			print "X: ", X, "; ", "Y: ", Y
			x_tag = X.split("_")[-1]
			acc = get_NON_crossval_accuracy_score(X_objs[X], Y_objs[Y])
			print "accuracy_score: ", acc
			cvs = get_crossvalidation_score(X_objs[X], Y_objs[Y])
			print "CrossValidation Score: ", cvs
			final_scores_dict[Y][x_tag].extend([acc, cvs])

	return final_scores_dict

#c,d = calculate_fraction_pos_examples()
#print c, d

def plot_scores_dict():
	to_plot_dict = run_on_data_for_Scores()
	#with open("test_to_plot", "w") as f:
	#	f.write(str(to_plot_dict))
	
	for k in to_plot_dict.keys():
		d_tags = ["pos_frac", "exp_acc", "4mers", "3mers", "mcount", "mmax", "msum"]
		x_tags = ["pos_frac", "exp_acc", "4mers_AccS", "4mers_CVS", "3mers_AccS", "3mers_CVS", "mcount_AccS", "mcount_CVS", "mmax_AccS", "mmax_CVS", "msum_AccS", "msum_CVS"]
		bar_colors = ["b", "y", "g", "r", "g", "r", "g", "r", "g", "r", "g", "r"]
		x_locs = range(1, len(x_tags)+1)
		y_vals = []
		for t in d_tags:
			y_vals.extend(to_plot_dict[k][t])
		plt.bar(x_locs, y_vals, width=0.5, color=bar_colors)
		plt.xticks(x_locs, x_tags, rotation=45, size="xx-small")
		plt.title("scores for: %s_26082016"%(k)) #%s_15022016"%(k))
		plt.savefig("scores_for_%s_26082016.png"%(k))
		plt.clf()
	return
def get_features():
	features= {
	"features_4mers": data[0][1:137],
	"features_3mers": data[0][137:169],
	"features_mcount": data[0][169:568:3],
	"features_mmax": data[0][170:568:3],
	"features_msum": data[0][171:568:3]}
	return features

def get_feature_importance(x_obj, y_obj, num_estimators, max_features_param):
	rf = ensemble.RandomForestClassifier(n_estimators=num_estimators, max_features=max_features_param, n_jobs=3)
	rf.fit(x_obj, y_obj)
	feature_imp_array = rf.feature_importances_
	return feature_imp_array

def plot_feature_importance_errorbar(x_tags, sub_feat_imp_mean_list, sub_feat_imp_stdev_list, boruta_cands, top_cands_label, name):
	plt.figure(figsize=(19.20, 10.90), dpi=100)
	x_pos = range(len(x_tags))
	#x_pos = range(1,len(x_tags)+1)
	y_vals = sub_feat_imp_mean_list
	y_err = sub_feat_imp_stdev_list
	#plt.errorbar(x_pos, y_vals, y_err, linestyle='None', marker='o', mfc='red', mec='green', ms=4, mew=0.5)
	for x in x_pos:
                if x_tags[x] in top_cands_label:
                        plt.errorbar([x+1], [y_vals[x]], [y_err[x]], linestyle='None', marker='o', mfc='red', mec='green', ms=4, mew=0.5, label=x_tags[x])
                else:
                        plt.errorbar([x+1], [y_vals[x]], [y_err[x]], linestyle='None', marker='o', mfc='yellow', mec='green', ms=4, mew=0.5,color="k")
		plt.annotate(boruta_cands[x], (x+1, y_vals[x]))
	#where mfc, mec, ms and mew are aliases for the longer property names, markerfacecolor, markeredgecolor, markersize and markeredgewith.
	#plt.xticks(x_pos, x_tags, rotation=90, size="xx-small") 
	#x_ticks_pos = range(1, len(x_tags)+3, 3)
	if not name.split("_")[-3] == "3mers":
		x_ticks_pos = range(1, len(x_tags)+3, 3)
	else:
		x_ticks_pos = range(1, len(x_tags)+1)
	plt.xticks(x_ticks_pos, x_ticks_pos, rotation=90, size="xx-small")
	plt.ylabel("mean and stdev of feature importance")
	plt.title(name)
        plt.legend()
	## trying to modify the legend here:
	leg = plt.gca().get_legend()
	ltext  = leg.get_texts()  # all the text.Text instance in the legend
	frame  = leg.get_frame()
	plt.setp(ltext, fontsize='xx-small')
	plt.setp(frame, alpha=0.0)
	plt.savefig("%s_boruta_iter_10k_MotifThreshold5_03082016.png"%(name))
	plt.clf()
	return

def get_labels_for_top_cands(mean_val_list, feature_name_list,num=7):
	data_len = len(mean_val_list)
	cands_index = sorted(range(data_len), key=lambda i: mean_val_list[i], reverse=True)[:num]
	#cands_labels = [feature_name_list[l] for l in cands_index]
	cands_labels_for_plot = ["" for j in range(data_len)]
	for k in cands_index:
		cands_labels_for_plot[k] = feature_name_list[k]
	#print cands_labels_for_plot
	return cands_labels_for_plot
	
def run_boruta(x_obj, y_obj):
	# define random forest classifier, with utilising all cores and
	# sampling in proportion to y labels
	#rf = RandomForestClassifier(n_jobs=-1, class_weight='auto', max_depth=5) # default in example of Boruta. there's NO "auto" in the possible options of class_weight in RFC documentation.
	rf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=500) # we choose keep default max_depth=none. "If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples"

	# define Boruta feature selection method
	feat_selector = boruta_py.BorutaPy(rf, n_estimators='auto', verbose=2, max_iter=10000) # default in example of Boruta. Added max_iter=10000. default is 100.
	#feat_selector = boruta_py.BorutaPy(rf, verbose=2) # if "n_estimators" is NOT set here in the class-call, then it should be set to 1000, as set by default in the __init__() of the class.

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
	X_filtered = feat_selector.transform(x_obj) #(X)
	#for j in range(20):
	#       print  X_filtered[j][0:]
	ranked_cands = [ "C" if cand==1 else "T" if cand==2 else "" for cand in ranks]
	#return ranks
	return ranked_cands


def compute_feature_importance():
	X_objs, Y_objs = return_objects()
	feature_names = get_features()
	num_est = [100, 200, 300, 400, 500]
	max_f = ["auto", "log2"]
	#f_out = open("output_details_capture_run_29022016", "a+")
	f_out = open("output_details_capture_Kc_S2_run_Boruta_iter10k_MotifThreshold5_mcounts_03082016", "a+")
	for y in Y_objs: #["Y_All"]:  #Y_objs:
		for x in ["norm_X_mcount"]:  #X_objs: #["norm_X_msum", "norm_X_3mers"]: #X_objs: # x is the Broad feature Name
			x_name = x.split("_")[-1]
			sub_features = feature_names["features_"+ x_name]
			sub_features_importance = [get_feature_importance(X_objs[x], Y_objs[y], n, m) for m in max_f for n in num_est]
			#print sub_features_importance
			#mean_vals_each_subfeature = [mean([dz[i] for dz in sub_features_importance]) for i in range(len(dz))] ## these two lines raise UnboundLocalError. Hence scraped.
			#stdev_vals_each_subfeature = [std([dz[i] for dz in sub_features_importance]) for i in range(len(dz))]
			mean_vals_each_subfeature = []
			stdev_vals_each_subfeature = []
			for i in range(len(sub_features_importance[0])):
				each_sub_feature_vals = []
				for d in sub_features_importance:
					each_sub_feature_vals.append(d[i])
				mean_vals_each_subfeature.append(mean(each_sub_feature_vals))
				stdev_vals_each_subfeature.append(std(each_sub_feature_vals))
			cands_labels = get_labels_for_top_cands(mean_vals_each_subfeature, sub_features) # cands_labels are for top cands from RFClassifier mean/std-val top cands. NOT boruta.
			boruta_cands = run_boruta(X_objs[x], Y_objs[y])
			plot_name = y + "_vs_" + x_name + "_feature_stability_and_Boruta_candidates_10K_iter_MotifThreshold5"
			#print plot_name
			f_out.write(plot_name+ "\n" + str(sub_features)+"\n" + str(cands_labels) + "\n" + str(boruta_cands)+"\n")
			plot_feature_importance_errorbar(sub_features, mean_vals_each_subfeature, stdev_vals_each_subfeature, boruta_cands, cands_labels, plot_name)
			#plot all sub_features and their errobar
	f_out.close()
	return

def parse_boruta_ranks():
	#run_borut
	pass

def plot_boruta_ranks():
	#plt.figure()
	pass

def test_plot_boruta_candidates():
	X_objs, Y_objs = return_objects()
	feature_names = get_features()
	for y in ["Y_All"]: # Y_objs:
		for x in ["norm_X_3mers"]: #X_objs:
			x_name = x.split("_")[-1]
			sub_features = feature_names["features_"+ x_name]
			boruta_ranks = run_boruta(X_objs[x], Y_objs[y])
			print boruta_ranks
			

if __name__ == "__main__":
	plot_scores_dict() # this is for performance score plots. used on 26-08-2016
	#print get_features()
	#compute_feature_importance() # this is for feature importance (Boruta) related stuff. used on 03-08-2016
	#test_plot_boruta_candidates()

