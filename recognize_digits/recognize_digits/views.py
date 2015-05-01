
from django.shortcuts import render, get_object_or_404
from django.template import RequestContext
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
import base64
import numpy as np


# Loads home page
def home(request):
    return render(request, 'home_page.html')

# function to load dictionary for best dataset
def json_numpy_obj_hook(dct):
        """
        Decodes a previously encoded numpy ndarray
        with proper shape and dtype
        :param dct: (dict) json encoded ndarray
        :return: (ndarray) if input was an encoded ndarray
        """
        if isinstance(dct, dict) and '__ndarray__' in dct:
            data = base64.b64decode(dct['__ndarray__'])
            return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
        return dct

# gets best dataset
def get_best():

    title = "means_random_cluster50"
    # Load in the file
    with open('static/data/' + title + '_results.json') as data_file:
        best_set = json.load(data_file,object_hook=json_numpy_obj_hook)

    return best_set


# Process submitted image

# Helper function for images

def decode_base64(data):

    import base64
    """Decode base64, padding being optional.

    :param data: Base64 data as an ASCII byte string
    :returns: The decoded byte string.

    """
    missing_padding = 4 - len(data) % 4
    if missing_padding:
        data += b'='* missing_padding
    return base64.decodestring(data)

def shrink_image(img):
    from PIL import Image
    basewidth = 28
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    return img

@csrf_exempt
def predict(request):
    from PIL import Image
    # from StringIO import StringIO
    import base64
    # import requests

    # def transform(image_string):
    #     STANDARD_SIZE = (28, 28)
    #     f = StringIO(decode_base64(image_string))
    #     img = Image.open(f)
    #     img = img.getdata()
    #     img = img.resize(STANDARD_SIZE)
    #     img = map(list, img)
    #     img = np.array(img)
    #     s = img.shape[0] * img.shape[1]
    #     img_wide = img.reshape(1, s)
    #     return [img_wide[0]]

    best_set = get_best()["cluster_means"]
    # # num = classify(get_best["cluster_means"], transform(request.POST["img"]))

    # def transform(image_string):
    #     import re
    #     import cStringIO
    #     import scipy.misc
    #     STANDARD_SIZE = (28, 28)
    #     imgstr = re.search(r'base64,(.*)', image_string).group(1)
    #     tempimg = cStringIO.StringIO(imgstr.decode('base64'))
    #     img = Image.open(tempimg)
    #     img = img.getdata()
    #     img = img.resize(STANDARD_SIZE)
    #     img = scipy.misc.imresize(img, (784,1))
    #     img = np.array(img)
    #     return img

    # number = classify(best_results, transform(request.POST["img"]))
    # print "!@#$%^&#@!%^@&#*#^%$@!%@^#&"
    # print ""
    # print classify(best_results, transform(request.POST["img"]))
    # print ""
    # print "WQEVTHYT$EVWBRGHNTRVWGEHNTRGVB"

    import re
    import cStringIO
    image_string = request.POST["img"]
    STANDARD_SIZE = (28, 28)
    imgstr = re.search(r'base64,(.*)', image_string).group(1)
    tempimg = cStringIO.StringIO(imgstr.decode('base64'))
    img = Image.open(tempimg).convert("L")
    img = img.getdata()

    img = shrink_image(img)
    img_array = np.array(img, dtype=np.uint8)


    number = classify(best_set, [img_array])

    # return HttpResponse(json.dumps({"number": int(number)}))
    return HttpResponse(int(number))
    # print request


import Distance
def classify(cluster_set,test_set,distfn=Distance.sumsq):
    
    # Clusters is the array of final cluster means
    clusters = []
    c_index = []
    for cluster in cluster_set:
        clusters.append(cluster[0])
        c_index.append(cluster[1])

    test_clusters_asgn = np.apply_along_axis(Distance.leastsquares, 1, 
        test_set, clusters[1:49], distfn)

    test_clusters = np.array([c_index[i + 1] for i in test_clusters_asgn])

    return int(test_clusters[0])







