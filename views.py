from django.http import HttpResponse
from django.shortcuts import render
from mystockapp.getdata import get_current_price,get_chart,get_prediction,get_pred_price,get_compare,get_details,get_recommend

import datetime

possible_stocks=['AAPL','GOOGL','TSLA','AMZN','TWTR','NVDA','INTC','DIS']

def index(request):
    
    s_name = request.GET.get('options','default')
    if(s_name in possible_stocks):
        context = {}
        context["valid"] = True;
        context["stock_name"] = s_name
        context["price"] = get_current_price(s_name)
        context["chart"] = get_chart(s_name)
        context["pred"]  = get_prediction(s_name)
        context["pred_price"] = get_pred_price(s_name,s_name)
        context["stock_rec"] = get_recommend(context["pred_price"],context["price"])
    else:
        context ={}
        context["valid"] = False;
    
    return render(request, 'mystockapp/index.html',context) 
	
def compare(request):
    
    s_name1 = request.GET.get('stock1','default')
    s_name2 = request.GET.get('stock2','default')
    if(s_name1 in possible_stocks):
        if(s_name2 in possible_stocks):
            context = {}
            context["stock_name1"] = s_name1
            context["stock_name2"] = s_name2
            context["price1"] = get_current_price(s_name1)
            context["price2"] = get_current_price(s_name2)
            context["chart"] = get_compare(s_name1,s_name2)
            context["valid"] = True
    else:
        context ={}
        context["valid"] = False
    
    return render(request, 'mystockapp/compare.html',context) 

def details(request):
    
    query = request.GET.get('search_query','default')
    context={}
    context["query"] = query
    context["num"] = 10
    if(query in possible_stocks): 
        context["summary"] = get_details(query)
        context["price"] = get_current_price(query)
       

    return render(request, 'mystockapp/details.html',context) 
