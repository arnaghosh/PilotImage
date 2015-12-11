from django.shortcuts import render
from django.shortcuts import render_to_response
from django.conf import settings
from django.template import RequestContext
from django.http import HttpResponse,HttpResponseRedirect
from .forms import ImageForm
from .ocr import process_image
import json
# Create your views here.

def index(request):
    return render(request, 'index.html')

def ocr(request):
    try:
        url = request.POST['image_url']
        if 'jpg' in url:
            print "yes"
            output = process_image(url)
            print " no"
            return HttpResponse(json.dumps({"output": output}),content_type="application/json")
        else:
            return HttpResponse(json.dumps({"error": "only .jpg files, please"}),content_type="application/json")
    except:
        return HttpResponse(json.dumps({"error": "Did you mean to send: {'image_url': 'some_jpeg_url'}"}), content_type="application/json")

# def category(request, category_name):
#     context_dict = {}
#     context_dict['imgpath'] = category_name

#     f = open(settings.STATIC_PATH + '/TO/' + category_name + '.txt').read()
#     g = open(settings.STATIC_PATH + '/SauvolaTO/' + category_name + '.txt').read()
#     context_dict['TOoutput'] = f
#     context_dict['Sauvolaoutput'] = g

#     return render(request, 'category.html', context_dict)
