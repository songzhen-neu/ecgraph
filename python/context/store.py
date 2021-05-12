import context.context as context

global changeRate
global embs
global isTrain
global threadCountList
global threadCountList_backProp
global commTime
global compTime

def init():
    for i in range(context.glContext.config['layerNum']):
        threadCountList.append(0)
        threadCountList_backProp.append(0)

threadCountList_backProp=[]
threadCountList=[]
changeRate={}
embs={}
compTime=0
commTime=0
