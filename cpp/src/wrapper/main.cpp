#include <unordered_set>
#include <string>
#include <cfloat>

#include <gegelati.h>

#include "include/wrapper/GymWrapper.h"

int renderBest();

int main() {

    if(true) {
        renderBest();
        exit(0);
    }

	// Create the instruction set for programs
	Instructions::Set set;
	auto minus = [](float a, float b) -> float { return (float)a - (float)b; };
	auto add = [](float a, float b) -> float { return a + b; };
	auto max = [](float a, float b) -> float { return std::max(a, b); };
	auto modulo = [](float a, float b) -> float { return b != 0.0 ? fmod(a, b) : DBL_MIN; };
	auto cond = [](float a, float b) -> float { return a < b ? -a : a; };
	auto positive = [](float a, float b) -> float { return b < 0 ? -a : a; };


	set.add(*(new Instructions::LambdaInstruction<float, float>(minus)));
	set.add(*(new Instructions::LambdaInstruction<float, float>(add)));
	set.add(*(new Instructions::LambdaInstruction<float, float>(max)));
	set.add(*(new Instructions::LambdaInstruction<float, float>(modulo)));
	set.add(*(new Instructions::LambdaInstruction<float, float>(cond)));
	set.add(*(new Instructions::LambdaInstruction<float, float>(positive)));


	// Set the parameters for the learning process.
	// (Controls mutations probability, program lengths, and graph size
	// among other things)
	// Loads them from "params.json" file
	Learn::LearningParameters params;
	File::ParametersParser::loadParametersFromJson(ROOT_DIR "/params.json", params);
#ifdef NB_GENERATIONS
	params.nbGenerations = NB_GENERATIONS;
#endif // !NB_GENERATIONS


	// Instantiate the LearningEnvironment
	// add a "true" in the constructor args to swap to non-adversarial
	GymWrapper le("MountainCar-v0",3,2);

	// Instantiate and init the learning agent
	Learn::ParallelLearningAgent la(le, set, params);
	la.init();

	// Adds a logger to the LA (to get statistics on learning) on std::cout
	auto logCout = *new Log::LABasicLogger(la);

	// Adds another logger that will log in a file
	std::ofstream o("log");
	auto logFile = *new Log::LABasicLogger(la, o);

	// Create an exporter for all graphs
	File::TPGGraphDotExporter dotExporter("out_000.dot", la.getTPGGraph());




	// Train for NB_GENERATIONS generations
	for (int i = 0; i < params.nbGenerations; i++) {
		char buff[12];
		sprintf(buff, "out_%03d.dot", i);
		dotExporter.setNewFilePath(buff);
		dotExporter.print();
		/*std::multimap<std::shared_ptr<Learn::EvaluationResult>, const TPG::TPGVertex*> result;
		result = la.evaluateAllRoots(i, Learn::LearningMode::VALIDATION);*/

		la.trainOneGeneration(i);

        dotExporter.setNewFilePath("/home/asimonu/Bureau/Gegelati/gym-wrapper/out_best.dot");
        dotExporter.print();
	}

	// Keep best policy
	la.keepBestPolicy();
	dotExporter.setNewFilePath("out_best.dot");
	dotExporter.print();



	// cleanup
	for (unsigned int i = 0; i < set.getNbInstructions(); i++) {
		delete (&set.getInstruction(i));
	}

	// if we want to test the best agent

	return 0;
}

int renderBest() {
    // Create the instruction set for programs
    Instructions::Set set;
    auto minus = [](double a, double b)->double {return (double)a - (double)b; };
    auto add = [](double a, double b)->double {return a + b; };
    auto max = [](double a, double b)->double {return std::max(a, b); };
    auto modulo = [](double a, double b)->double {return b != 0.0 ? fmod(a,b):DBL_MIN;};
    auto nulltest = [](double a)->double {return (a == -1.0) ? 10.0 : 0.0; };
    auto circletest = [](double a)->double {return (a == 0.0) ? 10.0 : 0.0; };
    auto crosstest = [](double a)->double {return (a == 1.0) ? 10.0 : 0.0; };
    auto test15 = [](double a)->double {return (a >= 15.0) ? 10.0 : 0.0; };
    auto cond = [](double a, double b) -> double { return a < b ? -a : a; };


    set.add(*(new Instructions::LambdaInstruction<double, double>(minus)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(add)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(max)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(modulo)));
    set.add(*(new Instructions::LambdaInstruction<double>(nulltest)));
    set.add(*(new Instructions::LambdaInstruction<double>(circletest)));
    set.add(*(new Instructions::LambdaInstruction<double>(crosstest)));
    set.add(*(new Instructions::LambdaInstruction<double>(test15)));
    set.add(*(new Instructions::LambdaInstruction<double,double>(cond)));


    // Instantiate the LearningEnvironment
    GymWrapper le("MountainCar-v0",3,2, true);

    // Instantiate the environment that will embed the LearningEnvironment
    Environment env(set, le.getDataSources(), 8);

    // Instantiate the TPGGraph that we will loead
    auto tpg = TPG::TPGGraph(env);

    // Instantiate the tee that will handle the decisions taken by the TPG
    TPG::TPGExecutionEngine tee(env);

    // Create an importer for the best graph and imports it
    File::TPGGraphDotImporter dotImporter("/home/asimonu/Bureau/Gegelati/gym-wrapper/out_best.dot", env, tpg);
    dotImporter.importGraph();

    // takes the first root of the graph, anyway out_best has only 1 root (the best)
    auto root = tpg.getRootVertices().front();

    bool play = true;
    // let's play, the only way to leave this loop is to enter -1
    while(play){
        uint64_t action=((const TPG::TPGAction *) tee.executeFromRoot(* root).back())->getActionID();
        //std::cout<<"Action "<<std::to_string(action)<<std::endl;
        le.doAction(action);
        usleep(20000); // waiting x microseconds

        if(le.isTerminal()){
            std::cout<<"End reached. Closing..."<<std::endl;
            play = false;
        }
    }


    // cleanup
    for (unsigned int i = 0; i < set.getNbInstructions(); i++) {
        delete (&set.getInstruction(i));
    }

    return 0;
}
