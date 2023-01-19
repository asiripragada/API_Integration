from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd

# Create your views here.

def index(request):
    return render(request, "input.html")


def addition(request):

    num1 = request.GET['num1']
    num2 = request.GET['num2']

    if num1.isdigit() and num2.isdigit():
        a = int(num1)
        b = int(num2)
        res = a + b
        
        # initialize list of lists
#         data = [['output', res]]
          
        # Create the pandas DataFrame
#         df = pd.DataFrame(data, columns=['key', 'value'])

        return render(request, "result.html", {"result": res})
    else:
        res = "Only digits are allowed"
        return render(request, "result.html", {"result": res})
