#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <ctime>
#include <algorithm>
#include <omp.h>

typedef struct Node
{
	int id;
	double x_coordinate;
	double y_coordinate;

	Node::Node(int i, double x, double y)
		: id(i), x_coordinate(x),y_coordinate(y){}
}Node;

typedef struct Member
{
	std::vector<int> path; // Array of the path taken by this member
	double fitness; // Contains the total distance of the path

	Member::Member()
		: fitness(DBL_MAX){}

	bool operator < (const Member& m) const
	{
		return (fitness < m.fitness);
	}
}Member;

/* Global Variables */
std::vector<Node*> g_data;
std::vector<Member*> g_population;
Member* g_currentBest;
Member* g_currentSecondBest;
const int g_populationSize = 100;
const int g_evaluationLimit = 1e6;
unsigned int g_accEvaluations;
const int g_nParents = 2;
const double g_mutationProbability = 0.75;
const int g_mutationLimit = 1;
const double g_crossoverProbability = 1.0;

void getDataFromFile()
{
	std::string id, x, y;
	std::ifstream file("coordinates.txt");

	if (!file.is_open())
	{
		std::cout << "Could not open file.\n";
		exit(1);
	}

	while (file >> id >> x >> y)
	{
		g_data.push_back(new Node(std::stoi(id), std::stod(x), std::stod(y)));
	}
}

void updateFitness(Member *m)
{
	double acc = 0.0;
	int i;

	for (i = 0; i < m->path.size() - 2; ++i)
	{
		acc += sqrt((g_data[m->path[i + 1] - 1]->x_coordinate - g_data[m->path[i] - 1]->x_coordinate) * 
			(g_data[m->path[i + 1] - 1]->x_coordinate - g_data[m->path[i] - 1]->x_coordinate) + 
			(g_data[m->path[i + 1] - 1]->y_coordinate - g_data[m->path[i] - 1]->y_coordinate) * 
			(g_data[m->path[i + 1] - 1]->y_coordinate - g_data[m->path[i] - 1]->y_coordinate));
	}

	acc += sqrt((g_data[m->path[0] - 1]->x_coordinate - g_data[m->path[i + 1] - 1]->x_coordinate) *
		(g_data[m->path[0] - 1]->x_coordinate - g_data[m->path[i + 1] - 1]->x_coordinate) +
		(g_data[m->path[0] - 1]->y_coordinate - g_data[m->path[i + 1] - 1]->y_coordinate) *
		(g_data[m->path[0] - 1]->y_coordinate - g_data[m->path[i + 1] - 1]->y_coordinate));

	m->fitness = acc;

	#pragma omp atomic
	g_accEvaluations++;

	if (g_accEvaluations % 50000 == 0)
		std::cout << "# Evaluations: " << g_accEvaluations << " Fitness: " << g_population[0]->fitness << " Current Best: " << g_currentBest->fitness << " Current Second Best: " << g_currentSecondBest->fitness << std::endl;
}

bool myfunction(Member *i, Member *j) { return (i->fitness<j->fitness); } // Used for sorting

void initializePopulation()
{

	Member *tempMember;
	//#pragma omp parallel for private(tempMember)
	for (int i = 0; i < g_populationSize; i++)
	{
		tempMember = new Member();
		for (int j = 0; j < g_data.size(); ++j) // Initialize each member
		{
			tempMember->path.push_back(g_data[j]->id);
		}

		std::random_shuffle(tempMember->path.begin(), tempMember->path.end()); // Randomize initial path

		updateFitness(tempMember);

		//#pragma omp atomic
		g_population.push_back(tempMember);
	}

	std::sort(g_population.begin(), g_population.end(), myfunction); // Sort ascending on fitness, using < overload
	g_currentBest = g_population[0];
	g_currentSecondBest = g_population[1];
}

void selectParents(int *fatherIndex, int *motherIndex)
{
	double totalFitness = 0.0, r, sum = 0.0, scale = 0.0;
	std::vector<double> probability;
	bool alreadyPicked = false, keepChoosing = true;
	int nParentsChosen = 0;

	for (int i = 0; i < g_populationSize; ++i)
	{
		totalFitness += 1 / g_population[i]->fitness; // Calculate the sum of the fitness. Inverse to make the best fitness the largest.
	}

	//scale = 1 / totalFitness;

	for (int i = 0; i < g_populationSize; ++i)
	{
		probability.push_back(((1 / g_population[i]->fitness) / totalFitness) /** scale*/); // Calculate the probability to select each member as parent for Crossover
		//sum += probability[i];
	}

	while(keepChoosing) // Choose n parents
	{
		r = ((double)rand() / (RAND_MAX)); // Generate a random number for selection

		for (int j = 0; j < g_populationSize; ++j) // Roulette selection
		{
			sum += probability[j]; // Accumulate probability
			if(r < sum)
			{ 
				for (int k = 0; k < g_nParents; ++k) // Avoid duplicate parents
				{
					if (*fatherIndex == j)
					{
						alreadyPicked = true;
						break;
					}
				}
				if (!alreadyPicked)
				{
					if (nParentsChosen == 0)
						*fatherIndex = j;
					else
						*motherIndex = j;

					nParentsChosen++;
				}

				alreadyPicked = false;
				break;
			}
		}
		sum = 0.0;

		if (nParentsChosen >= g_nParents)
		{
			keepChoosing = false;
		}
	}
}

std::vector<Member*> performCrossover(const int &motherIndex, const int &fatherIndex)
{
	int startIndex, endIndex, maxIndex = g_data.size() - 1, size = maxIndex + 1, n;
	Member *child1, *child2;
	std::vector<Member*> children;
	std::vector<int> tempVector, taken;
	bool alreadyTaken = false;

	for (int i = 0; i < g_nParents / 2; ++i) // Generate a number of children equal to the number of parents
	{
		startIndex = rand() % (maxIndex + 1);
		endIndex = rand() % (maxIndex + 1);

		if (endIndex > startIndex)
			n = endIndex - startIndex;
		else
			n = g_data.size() - startIndex + (endIndex + 1);

		child1 = new Member;
		child2 = new Member;

		for (int j = 0; j < size; ++j)
		{
			tempVector.push_back(-1);
		}

		for (int j = 0; j < n; ++j) // Assign genes to the first child, from the father
		{
			if (startIndex + j > maxIndex)
			{
				tempVector[startIndex + j - size] = g_population[fatherIndex]->path[startIndex + j - size];
				taken.push_back(tempVector[startIndex + j - size]);
			}
			else
			{
				tempVector[startIndex + j] = g_population[fatherIndex]->path[startIndex + j];
				taken.push_back(tempVector[startIndex + j]);
			}
		}
		for (int j = 0; j < size; ++j) // Assign genes to the first child, from the mother
		{
			for (int k = 0; k < taken.size(); ++k) // Avoid duplicates
			{
				if (taken[k] == g_population[motherIndex]->path[j])
				{
					alreadyTaken = true;
				}
			}
			if (!alreadyTaken)
			{
				for (int l = 0; l < size; ++l) // Assign to unassigned indeces
				{
					if (tempVector[l] == -1)
					{
						tempVector[l] = g_population[motherIndex]->path[j];
						break;
					}
				}
			}
			alreadyTaken = false;
		}

		child1->path = tempVector;
		tempVector.empty();
		taken.clear();

		for (int j = 0; j < size; ++j)
		{
			tempVector[j] = -1;
		}

		for (int j = 0; j < n; ++j) // Assign genes to the second child, from the father
		{
			if (endIndex + j > maxIndex)
			{
				tempVector[endIndex + j - size] = g_population[fatherIndex]->path[endIndex + j - size];
				taken.push_back(tempVector[endIndex + j - size]);
			}
			else
			{
				tempVector[endIndex + j] = g_population[fatherIndex]->path[endIndex + j];
				taken.push_back(tempVector[endIndex + j]);
			}
		}
		for (int j = 0; j < size; ++j) // Assign genes to the second child, from the mother
		{
			for (int k = 0; k < taken.size(); ++k) // Avoid duplicates
			{
				if (taken[k] == g_population[motherIndex]->path[j])
				{
					alreadyTaken = true;
				}
			}
			if (!alreadyTaken)
			{
				for (int l = 0; l < size; ++l) // Assign to unassigned indeces
				{
					if (tempVector[l] == -1)
					{
						tempVector[l] = g_population[motherIndex]->path[j];
						break;
					}
				}
			}
			alreadyTaken = false;
		}
		child2->path = tempVector;

		/*if (child1->path[51] == -1 || child2->path[51] == -1)
			std::cout << "Stuff" << std::endl;*/
	}

	children.push_back(child1);
	children.push_back(child2);

	return children;
}

void mutate(const std::vector<Member*> &children)
{
	double r;
	int a, b, size = g_data.size(), temp; 
	for (int i = 0; i < children.size(); ++i)
	{
		r = ((double)rand() / (RAND_MAX));

		if (r < g_mutationProbability)
		{
			a = rand() % size;
			do
			{
				b = rand() % size;
			} while (b == a);

			temp = children[i]->path[a];
			children[i]->path[a] = children[i]->path[b];
			children[i]->path[b] = temp;
		}

		updateFitness(children[i]);
	}
}

void tspGA()
{
	g_accEvaluations = 0;
	initializePopulation();
	std::vector<Member*> newPopulation;
	std::vector<Member*> spawn;
	int fatherIndex = -1, motherIndex = -1;

	for (int i = 0; i < g_populationSize; i++)
	{
		newPopulation.push_back(nullptr);
	}

#pragma omp parallel private(spawn, fatherIndex, motherIndex)
	{
		while (g_accEvaluations < g_evaluationLimit)
		{
			#pragma omp for
			for (int i = 1; i < g_populationSize / 2; i++)
			{
				selectParents(&fatherIndex, &motherIndex);
				spawn = performCrossover(fatherIndex, motherIndex);
				mutate(spawn);

				newPopulation[i * 2] = spawn[0];
				newPopulation[i * 2 + 1] = spawn[1];

				spawn.empty();
			}

			#pragma omp single
			{
				for (int i = 2; i < g_population.size(); ++i)
				{
					delete g_population[i];
				}
				g_currentBest = g_population[0];
				g_currentSecondBest = g_population[1];
				newPopulation[0] = g_currentBest;
				newPopulation[1] = g_currentSecondBest;

				g_population = newPopulation;

				std::sort(g_population.begin(), g_population.end(), myfunction); // Sort ascending on fitness, using < overload;
			}
		}
	}
}

int main()
{
	getDataFromFile();
	srand(time(NULL));

	tspGA();

	std::cout << std::endl << "The shortest path found is " << g_population[0]->fitness << " units long." << std::endl;


	system("pause");
	for (int i = 0; i < g_data.size(); ++i)
	{
		delete g_data[i];
	}
	for (int i = 0; i < g_population.size(); ++i)
	{
		delete g_population[i];
	}
	return EXIT_SUCCESS;
}