def user_name(request):
    user_name = request.user
    return {'user_name':user_name}

def email(request):
    email = request.user.email
    return {'email':email}