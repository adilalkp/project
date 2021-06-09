def user_name(request):
    username = request.user
    return username

def email(request):
    email = request.user.email
    return email