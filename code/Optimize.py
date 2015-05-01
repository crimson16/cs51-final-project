######################################
# Optimization Function              #
######################################

'''

The Optimize function can be called as it's own item. It will 
run all possible iterations of our k means algorithm and 
will print out the best data from our clusters

'''


import main_cluster
import pandas as pd


def optimize ():
    testing_ks = range(10,20) + range(20, 55, 5)
    methods = ["means", "medoids", "medians"]
    init_types = ["random", "kplusplus"]

    # run for all possible ks:

    results_set = []

    # Loop and make all k's
    for method in methods:
        for k in testing_ks:
            for init_type in init_types:
                main_cluster.main(k, m=method, init_type=init_type)
                title = m + "_" + init_type + "_cluster" + str(k)

                # Then save results to dataframe
                with open('./results/' + title + '/' + title + '_results.json') as infile:
                    results.append(json.loads(infile, object_hook=json_numpy_obj_hook))

    
    results_df = pd.DataFrame(results_set)

    # Here we look at just the absolute best results for each method and initialization type. 
    # Surprisingly we found that random initialization was best.
    for m in methods:
        for init_t in init_types:
            df = results_df[(results_df.method==m) & (results_df.init_type==init_t)]
            print "The best Accuracy was: ", df.prediction_accuracy.max(), " in %s and %s"%(m, init_t)
            df = df[df.prediction_accuracy == df.prediction_accuracy.max()]
            print "And the K value was: ", df.k.max(), ", while runtime was ", df.clustering_time.max()
            print ""
        
   
### overall best results
print "The best Accuracy was ", results_df.prediction_accuracy.max()
df = results_df[results_df.prediction_accuracy == results_df.prediction_accuracy.max()]
print "The method was %s, and the initialization type was %s" %(str(df.method.max()), str(df.init_type.max()))
print "And the K value was: %d" %(int(df.k.max())), " while runtime was ", df.clustering_time.max()




# Helper functions


"""
    If input object is a ndarray it will be converted into a 
    dict holding dtype, shape and the data base64 encoded
"""
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            data_b64 = base64.b64encode(obj.data)
            return dict(__ndarray__=data_b64,
                        dtype=str(obj.dtype),
                        shape=obj.shape)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder(self, obj)

"""
    Decodes a previously encoded numpy ndarray
    with proper shape and dtype
    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
"""
def json_numpy_obj_hook(dct):
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct