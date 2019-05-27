def inprocess(strings):
    dicts = {'<':'>', '>':'<', '[':']', ']':'[' }

    flag = []
    i = 0
    res = []
    start = 0
    # split inputs to many strings
    while i <len(strings):
        s = strings[i]
        if s=='<' or s=='[':
            flag.append(s)
            if strings[start]=='@':
                res.append(strings[start:i])
                start = i

        elif flag and s==dicts[flag[-1]]:
            tmp = flag.pop()
            if flag:
                continue
            else:
                res.append(strings[start+1:i])
                start = i+1
            del tmp

        i += 1

    # process the res
    collect = []
    for id, lists in enumerate(res):
        if '|' not in lists:
            collect.append(lists)
            continue

        lists = lists.split('|')
        for s in lists:
            i = 0
            # while i < len(s):
            if s[i]=='[':
                collect.append(s[i+1:s.index(']')])
                i = s.index(']')+1
                if i<len(s):
                    collect.append(collect[-1]+s[i:])

            elif '[' not in s:
                collect.append(s)
    for c in res:
        print(c)
    print(collect)
    return collect

def testing(qurry, collect):
    i = 0
    while i<len(qurry):
        c = qurry[i]
        if c in collect:
            collect = collect[collect.index(c)+1:]
        elif c=='@':
            if qurry[i:qurry.index('}')+2] in collect:
                i = qurry.index('}') + 1
            else:
                return 0
        else:
            return 0
        i += 1
    return 1


if __name__ == '__main__':
    inputs = input()
    qurry = input()
    res = inprocess(inputs)
    print(testing(qurry,res))