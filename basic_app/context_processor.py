def user_name(request):
    try:
        user_name = request.user
    except:
        user_name = ""
    return {'user_name':user_name}

def email(request):
    try:
        email = request.user.email
    except:
        email = ""

    return {'email':email}