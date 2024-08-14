# gilman_classic
Team building algorithm for the Gilman Classic charity vb tournament

I've put in both simulated annealing and also inductive logic programming. going to use the ILP results because it's guaranteed to be more optimal when presented with constraints. 

unfortunately, the algorithm will take forever to find a truly optimal solution, but you can set a block on how long to run in `prob.solve(pulp.PULP_CBC_CMD(timeLimit=300))`

i won't be releasing the player ratings so don't even try me on that
