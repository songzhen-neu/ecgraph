import context.context as context

def assignParam():
    serverNum = context.glContext.config['server_num']
    for j in range(serverNum):
        context.glContext.weightForServer[j] = {}
        for i in range(context.glContext.config['layerNum']):
            context.glContext.weightForServer[j][i] = []

    for i in range(context.glContext.config['layerNum']):
        weight = context.glContext.weights[i].detach().numpy()
        # weight=weight.tolist()
        context.glContext.bias[i] = context.glContext.bias[i].detach().numpy().tolist()
        rowDim = len(weight)
        colDim = len(weight[0])
        for j in range(serverNum):
            if j == (serverNum - 1):
                context.glContext.weightForServer[j][i] = weight[int(rowDim / serverNum) * (serverNum - 1):, :].tolist()
            else:
                context.glContext.weightForServer[j][i] = weight[
                                                          int(rowDim / serverNum) * j:int(rowDim / serverNum) * (j + 1),:].tolist()
    print('assign parameter end!')