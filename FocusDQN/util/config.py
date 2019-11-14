r'''
    Configuration file parse utility.
'''

import configparser


def is_int(val_str):
    start_digit = 0
    if val_str[0] == '-':
        start_digit = 1
    flag = True
    for i in range(start_digit, len(val_str)):
        if str(val_str[i]) < '0' or str(val_str[i]) > '9':
            flag = False
            break
    return flag


def is_float(val_str):
    flag = False
    if '.' in val_str and not val_str.startswith('./') and not val_str.startswith('../'):
        if len(val_str.split('.')) == 2:
            if is_int(val_str.split('.')[0]) and is_int(val_str.split('.')[1]):
                flag = True
            else:
                flag = False
        elif len(val_str.split('.')) == 1:
            if is_int(val_str.split('.')[0]):
                flag = True
            else:
                flag = False
        else:
            flag = False
    elif 'e' in val_str and len(val_str.split('e')) == 2:
        if is_int(val_str.split('e')[0]) and is_int(val_str.split('e')[1]):
            flag = True
        else:
            flag = False
    else:
        flag = False
    return flag


def is_bool(val_str):
    if val_str == 'True' or val_str == 'true' or val_str == 'False' or val_str == 'false':
        return True
    else:
        return False


def parse_bool(val_str):
    if val_str == 'True' or val_str == 'true':
        return True
    else:
        return False


def is_list(val_str):
    if val_str[0] == '[' and val_str[-1] == ']':
        return True
    else:
        return False


def parse_list(val_str):
    sub_str = val_str[1:-1]
    splits = sub_str.split(',')
    output = []
    for item in splits:
        item = item.strip()
        if is_int(item):
            output.append(int(item))
        elif is_float(item):
            output.append(float(item))
        elif is_bool(item):
            output.append(parse_bool(item))
        elif is_none(item):
            output.append(None)
        else:
            output.append(item)
    return output


def is_none(val_str):
    if val_str == 'None' or val_str == 'none':
        return True
    else:
        return False


def is_tuple(val_str):
    if val_str[0] == '(' and val_str[-1] == ')':
        return True
    else:
        return False


def parse_tuple(val_str):
    return tuple(parse_list(val_str))


def is_dict(val_str):
    if val_str[0] == '{' and val_str[-1] == '}':
        return True
    else:
        return False


def parse_dict(val_str):
    sub_str = val_str[1:-1]
    splits = sub_str.split(',')
    output = {}
    for item in splits:
        item = item.strip()
        it_splits = item.split(':')
        if len(it_splits) != 2:
            raise Exception('Invalid dict string !!!')
        k = it_splits[0]
        v = it_splits[1]
        v = v.strip()
        if is_int(v):
            v = int(v)
        elif is_float(v):
            v = float(v)
        elif is_bool(v):
            v = parse_bool(v)
        elif is_none(v):
            v = None
        elif is_list(v):
            v = parse_list(v)
        elif is_tuple(v):
            v = parse_tuple(v)
        else:
            pass
        output[k] = v
    return output


def parse_value_from_string(val_str):
    if is_int(val_str):
        val = int(val_str)
    elif is_float(val_str):
        val = float(val_str)
    elif is_list(val_str):
        val = parse_list(val_str)
    elif is_bool(val_str):
        val = parse_bool(val_str)
    elif is_none(val_str):
        val = None
    else:
        val = val_str
    return val


def parse_config(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    output = {}
    for section in config.sections():
        output[section] = {}
        for key in config[section]:
            val_str = str(config[section][key])
            if len(val_str) > 0:
                val = parse_value_from_string(val_str)
            else:
                val = None
            print(section, key, val_str, val)
            output[section][key] = val
    return output
