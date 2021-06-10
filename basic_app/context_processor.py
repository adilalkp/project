def user_name(request):
    if request.user:
        user_name = request.user
    else:
        user_name =  ""
    return {'user_name':user_name}

def email(request):
    if request.user:
        email = request.user.email
    else:
        email = ""

    return {'email':email}