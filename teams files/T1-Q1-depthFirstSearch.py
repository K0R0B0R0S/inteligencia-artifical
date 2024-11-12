def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.expand(problem.getStartState()))
    """

    node = getStartNode(problem)
    frontier = util.Stack()
    frontier.push(node)

    closed = set()

    while not frontier.isEmpty():
        node = frontier.pop()

        if node['STATE'] in closed:
            continue

        closed.add(node['STATE'])

        if problem.isGoalState(node['STATE']):
            return getActionSequence(node)

        for sucessor in problem.expand(node['STATE']):
            child_node = getChildNode(sucessor, node)
            frontier.push(child_node)

    return []