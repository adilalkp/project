def user_name(request):
    username = request.user
    return user_name

def email(request):
    email = request.user.email
    return email