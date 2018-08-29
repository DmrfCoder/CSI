import pandas as pd


def write(data, path):
    f = pd.HDFStore(path, 'a')
    l = len(data)
    x = []
    y = []
    for i in range(l):
        x.append(data[i].x)
        y.append(data[i].y)

    xd = pd.DataFrame(x)
    yd = pd.DataFrame(y)
    f.append('x', xd)
    f.append('y', yd)

    f.close()
    print('success')
