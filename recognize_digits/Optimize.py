######################################
# Optimization Function #
######################################





testing_ks = range(10,20) + range(20, 55, 5)
methods = ["means", "medoids", "medians"]
init_types = ["random", "kplusplus"]

# run for all possible ks:

for method in methods:
    for k in testing_ks:
        for init_type in init_types:
            main(k, m=method, init_type=init_type)


title = m + "_" + init_type + "_cluster" + str(k)

results_set = []
with open('./results/' + title + '/' + title + '_results.json') as infile:
    results.append(json.loads(infile, object_hook=json_numpy_obj_hook))





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