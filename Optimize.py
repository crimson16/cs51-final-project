######################################
# Load in training images and labels #
######################################


testing_ks = range(10,20) + range(20, 55, 5)
methods = ["means", "medoids", "medians"]
init_types = ["random", "kplusplus"]

# for k in testing_ks:
m = methods[0]
init_type = init_types[0]
testing_ks = [0]


title = m + "_" + init_type + "_cluster" + str(k)

results_set = []
with open('./' + title + '/' + title + '_results.json') as infile:
    results.append(json.loads(infile, object_hook=json_numpy_obj_hook))

for k in testing_ks:
    print "python main_cluster.py {k} {method} {init_type} {prop}".format(k = k,
                                                                        method = m,
                                                                        init_type =init_type,
                                                                        prop = 5)
    