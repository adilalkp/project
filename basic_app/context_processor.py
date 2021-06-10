def user_name(request):
    if request.user != 'AnonymousUser':
        user_name = request.user
    else:
        user_name =  ""
    return {'user_name':user_name}

def email(request):
    if request.user != 'AnonymousUser':
        email = request.user.email
    else:
        email = ""

    return {'email':email}