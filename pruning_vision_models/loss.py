if __name__ == '__main__':
    # Using readlines()
    file = open('E:/Projects/Pruning/ScratchPrunedLayers/ResNet56PrunedLayers.out', 'r')
    lines = file.readlines()
    losses = []
    for line in lines:
        line = line.strip()
        line = line.replace(' ', '')
        if line.__contains__('loss:'):
            line = line.split('-')[2]
            line = line.replace('loss:', '')
            losses.append(float(line.rstrip()))
            print(line.strip())

    print(losses)