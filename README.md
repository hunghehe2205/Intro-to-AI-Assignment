# Pacman Search Algorithms Project

## Project Origin
This project is an assignment from an **Introduction to Artificial Intelligence** course, originally developed at UC Berkeley. The project focuses on implementing fundamental search algorithms and applying them to the classic Pacman game environment.

## Project Overview
This project implements various search algorithms and applies them to solve different pathfinding problems in a Pacman game environment. The implementation demonstrates core AI search concepts including uninformed search, informed search, and heuristic design.

## What This Project Does

### Core Search Algorithms (`search.py`)
The project implements four fundamental search algorithms:

1. **Depth-First Search (DFS)** - Explores the deepest nodes first using a stack-based approach
2. **Breadth-First Search (BFS)** - Explores nodes level by level using a queue-based approach
3. **Uniform Cost Search (UCS)** - Finds the lowest-cost path using a priority queue ordered by path cost
4. **A* Search** - Combines path cost with heuristic estimates for optimal and efficient pathfinding

Each algorithm follows the same general structure:
- Maintains a frontier of unexplored states
- Tracks explored states to avoid cycles
- Returns the sequence of actions to reach the goal

### Search Problems and Agents (`searchAgents.py`)
The project includes several search problems and specialized agents:

#### Search Problems Implemented:
1. **PositionSearchProblem** - Basic pathfinding to a specific position
2. **CornersProblem** - Visit all four corners of the maze efficiently
3. **FoodSearchProblem** - Collect all food pellets in the maze
4. **AnyFoodSearchProblem** - Find a path to the nearest food pellet

#### Key Features:
- **Heuristic Functions**: Manhattan distance and Euclidean distance heuristics for A* search
- **Advanced Heuristics**: Custom heuristics for corners and food problems using minimum spanning tree (MST) calculations
- **Specialized Agents**:
  - `StayEastSearchAgent` and `StayWestSearchAgent` with position-based cost functions
  - `AStarCornersAgent` for efficient corner visiting
  - `AStarFoodSearchAgent` for optimal food collection
  - `ClosestDotSearchAgent` for greedy food collection

## Implementation Highlights

### Search Algorithms (`search.py`)
- **Graph Search Implementation**: All algorithms use proper cycle detection with explored sets
- **Consistent Interface**: All search functions return a list of actions to reach the goal
- **Optimal Solutions**: UCS and A* guarantee optimal solutions when appropriate heuristics are used

### Advanced Problem Solving (`searchAgents.py`)
- **State Space Design**: Clever state representations (e.g., position + visited corners tuple)
- **Admissible Heuristics**: MST-based heuristics that never overestimate the true cost
- **Efficient Caching**: Reuse of computed distances to improve performance
- **Real-world Applications**: Demonstrates how AI search applies to practical pathfinding problems

The project showcases the progression from basic uninformed search to sophisticated informed search with custom heuristics, providing a comprehensive foundation in AI search techniques.

## How to Run and Test the Code

### Prerequisites
- Python 3.x
- All project files should be in the same directory

### Basic Usage
Run the Pacman game with different search algorithms using the following command structure:
```bash
python pacman.py -p SearchAgent -a fn=<search_function>
```

### Testing Different Search Algorithms

#### 1. Depth-First Search (DFS)
```bash
python pacman.py -p SearchAgent -a fn=dfs
python pacman.py -l tinyMaze -p SearchAgent -a fn=dfs
python pacman.py -l mediumMaze -p SearchAgent -a fn=dfs
```

#### 2. Breadth-First Search (BFS)
```bash
python pacman.py -p SearchAgent -a fn=bfs
python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
python pacman.py -l bigMaze -p SearchAgent -a fn=bfs -z .5
```

#### 3. Uniform Cost Search (UCS)
```bash
python pacman.py -p SearchAgent -a fn=ucs
python pacman.py -l mediumMaze -p SearchAgent -a fn=ucs
python pacman.py -l mediumDottedMaze -p StayEastSearchAgent
python pacman.py -l mediumScaryMaze -p StayWestSearchAgent
```

#### 4. A* Search
```bash
python pacman.py -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
```

### Testing Advanced Problems

#### Corners Problem
```bash
python pacman.py -l tinyCorners -p SearchAgent -a fn=bfs,prob=CornersProblem
python pacman.py -l mediumCorners -p SearchAgent -a fn=bfs,prob=CornersProblem
python pacman.py -l mediumCorners -p AStarCornersAgent -z 0.5
```

#### Food Search Problem
```bash
python pacman.py -l testSearch -p AStarFoodSearchAgent
python pacman.py -l trickySearch -p AStarFoodSearchAgent
```

#### Closest Dot Search
```bash
python pacman.py -l bigSearch -p ClosestDotSearchAgent -z .5
```

### Command Line Options
- `-l <layout>`: Specify the maze layout (tinyMaze, mediumMaze, bigMaze, etc.)
- `-p <agent>`: Choose the agent type (SearchAgent, AStarCornersAgent, etc.)
- `-a <args>`: Pass arguments to the agent (fn=dfs, prob=CornersProblem, etc.)
- `-z <zoom>`: Set the display zoom level (useful for large mazes)
- `-q`: Run in quiet mode (no graphics)

### Example Test Sequence
To test all implementations systematically:
```bash
# Test basic search algorithms
python pacman.py -l tinyMaze -p SearchAgent -a fn=dfs
python pacman.py -l tinyMaze -p SearchAgent -a fn=bfs
python pacman.py -l tinyMaze -p SearchAgent -a fn=ucs
python pacman.py -l tinyMaze -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic

# Test corners problem
python pacman.py -l tinyCorners -p AStarCornersAgent

# Test food search
python pacman.py -l testSearch -p AStarFoodSearchAgent
```

### Performance Testing
For performance comparison, run the same maze with different algorithms:
```bash
python pacman.py -l mediumMaze -p SearchAgent -a fn=dfs -q
python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs -q
python pacman.py -l mediumMaze -p SearchAgent -a fn=ucs -q
python pacman.py -l mediumMaze -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic -q
```

The `-q` flag runs without graphics for faster execution and cleaner output showing search statistics.