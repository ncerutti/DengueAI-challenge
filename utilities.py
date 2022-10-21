def get_expname_options(options):
    expname = 'test'
    return expname

def get_expname_datetime():
    # datetime object containing current date and time
    from datetime import datetime
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%Y%m%d-%H%M%S")
    print("date and time =", dt_string)
    return dt_string