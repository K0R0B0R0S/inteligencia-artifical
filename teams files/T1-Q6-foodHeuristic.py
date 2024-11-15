def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse than uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    "*** YOUR CODE HERE ***"
    position, foodGrid = state

    def getMazeDistance(start, end):
        """
        Returns the maze distance between any two points, using the search functions
        you have already built.
        """
        try:
            return problem.heuristicInfo[(start, end)]
        except:
            dist = mazeDistance(start, end, problem.startingGameState)
            problem.heuristicInfo[(start, end)] = dist
            return dist

    distances = []
    distances_food = [0]
    for food in foodGrid.asList():
        distances.append(getMazeDistance(position, food))
        for tofood in foodGrid.asList():
            distances_food.append(getMazeDistance(food, tofood))

    return min(distances)+max(distances_food) if len(distances) else max(distances_food)
