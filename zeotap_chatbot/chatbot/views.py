
from django.shortcuts import render
from django.http import JsonResponse
from .utils import find_best_match, preprocess_text
import google.generativeai as genai
import analytics
from django.conf import settings  # Import Django settings

# Configure Segment with the write key
analytics.write_key = settings.SEGMENT_WRITE_KEY



# Initialize Gemini (Make sure to configure API Key in settings.py)
genai.configure(api_key=settings.GOOGLE_API_KEY)  # Access API key from settings
model = genai.GenerativeModel('gemini-1.5-pro-latest')  

def chat_view(request):
    return render(request, 'chatbot/chat.html')

def gemini_answer(user_query):
    try:
        prompt = f"Answer the following question:\n{user_query}"
        response = model.generate_content(prompt)
        gemini_response = response.text #Gets the text from the Gemini response
        return JsonResponse({'response': gemini_response})
    except Exception as e:
        return JsonResponse({'response': f"Error generating response using Gemini: {e}"})

def get_response(request):
    if request.method != 'POST':
        return JsonResponse({'response': "Invalid request"})
    
    user_query = request.POST.get('user_query', '')
    if not user_query:
        return JsonResponse({'response': "Please ask a question."})

    user_id = request.session.get('user_id', 'anonymous')  # Track user sessions
    
    analytics.track(user_id, 'User Asked Question', {
        'query': user_query
    })

    cdp_base_urls = {
        "segment": "https://segment.com/docs/",
        "mparticle": "https://docs.mparticle.com/",
        "lytics": "https://docs.lytics.com/",
        "zeotap": "https://docs.zeotap.com/"
    }

    for keyword, cdp_base_url in cdp_base_urls.items():
        if keyword in user_query.lower():
            response_text = find_best_match(user_query, cdp_base_url)
            if response_text:
                analytics.track(user_id, 'Chatbot Answered with Docs', {
                    'query': user_query,
                    'response': response_text,
                    'source': keyword
                })
                return JsonResponse({'response': response_text})
            else:
                return gemini_answer(user_query)

    try:
        prompt = f"Answer the following question:\n{user_query}"
        response = model.generate_content(prompt)
        gemini_response = response.text  # Extracts text from the Gemini response
        
        analytics.track(user_id, 'Chatbot Answered with Gemini AI', {
            'query': user_query,
            'response': gemini_response
        })

        return JsonResponse({'response': gemini_response})
    
    except Exception as e:
        error_message = f"Error generating response using Gemini: {e}"
        analytics.track(user_id, 'Chatbot Error', {
            'query': user_query,
            'error': str(e)
        })
        return JsonResponse({'response': error_message})
    
