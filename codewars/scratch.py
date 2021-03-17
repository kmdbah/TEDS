def expanded_form(num):
    num_str = str(num)
    num_str = num_str[::-1]
    num_len = len(num_str)
    places = []
    zeroes = ''
    for i in range(0,num_len):
        if num_str[i] != '0':
            places.append(num_str[i]+zeroes)
        zeroes = zeroes + '0'
    places = places[::-1]
    output = ' + '.join(places)
    return output

print(expanded_form(70304))
