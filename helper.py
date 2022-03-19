def string_to_num(val):
    if( type(val) == str ):
        return float(val.replace(",","."))
    return val