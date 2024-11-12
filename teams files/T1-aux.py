def getActionSequence(node):

    actions = []

    while node['PATH-COST'] > 0:

        actions.insert(0,node['ACTION'])

        node = node['PARENT']

    return actions

  

def getStartNode(problem):

    node = {'STATE':problem.getStartState(),'PATH-COST':0}

    return node



def getChildNode(sucessor,parent_node):

    child_node = {'STATE': sucessor[0],

                  'ACTION': sucessor[1],

                  'PARENT': parent_node,

                  'PATH-COST': parent_node['PATH-COST'] + sucessor[2]}

    return child_node

