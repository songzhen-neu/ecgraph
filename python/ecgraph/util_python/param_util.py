from ecgraph.context import context as context


def assignParam():
    if context.glContext.config['id'] == 0:
        serverNum = context.glContext.config['server_num']
        parameters= context.glContext.parameters
        parametersForServer= context.glContext.parametersForServer

        for i in range(serverNum):
            parametersForServer[str(i)]={}
        for key in parameters.keys():
            num_p=int(len(parameters[key])/serverNum)
            for i in range(serverNum):
                if i!=(serverNum-1):
                    parametersForServer[str(i)][key]=parameters[key][i*num_p:(i+1)*num_p]
                else:
                    parametersForServer[str(i)][key]=parameters[key][i*num_p:]

        for i in range(context.glContext.config['server_num']):
            context.glContext.dgnnServerRouter[i].initParameter(
                context.glContext.config['worker_num'],
                context.glContext.config['server_num'],
                context.glContext.config['feature_dim'],
                context.glContext.config['hidden'],
                context.glContext.config['class_num'],
                context.glContext.config['id'],
                context.glContext.parametersForServer[str(i)]
            )
        print('assign parameter end!')