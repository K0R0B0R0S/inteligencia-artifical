def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    node = getStartNode(problem)
    fn_total_cost_for_node = lambda a_node: a_node['PATH-COST'] + heuristic(a_node['STATE'], problem=problem)

    frontier = util.PriorityQueueWithFunction(fn_total_cost_for_node)
    frontier.push(node)

    explored = set()

    while not frontier.isEmpty():
        node = frontier.pop()        

        if node['STATE'] in explored:
            continue        

        explored.add(node['STATE'])

        if problem.isGoalState(node['STATE']): 
          return getActionSequence(node)

        successors = problem.expand(node['STATE'])

        for sucessor in successors:
            child_node = getChildNode(sucessor, node)
            frontier.push(child_node)

    return []