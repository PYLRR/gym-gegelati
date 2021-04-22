#include "include/wrapper/GymWrapper.h"

void GymWrapper::initialize(std::string chosenEnv){
    this->envName = chosenEnv;
    this->client = Gym::client_create("127.0.0.1", 5000);
    this->env = client->make(this->envName);
}

void GymWrapper::doAction(uint64_t actionID) {
    Gym::State s;

    std::vector<float> action({(float)actionID});

    env->step(action, render, &s);

    for (int i=0; i<this->parametersToObserve; i++) {
        this->state.setDataAt(typeid(float), i, s.observation[i]);
    }

    this->reward+=s.observation[0]; // add current position
    if(s.observation[0]>maxDistance){
        maxDistance=s.observation[0];
    }
    this->turn++;
}

void GymWrapper::reset(size_t seed, Learn::LearningMode mode) {
    // Create seed from seed and mode
    size_t hash_seed = Data::Hash<size_t>()(seed) ^Data::Hash<Learn::LearningMode>()(mode);
    this->rng.setSeed(hash_seed);

    // sets 0 for each input data
    for (int i = 0; i < state.getLargestAddressSpace(); i++) {
        this->state.setDataAt(typeid(float), i, 0.0);
    }
    this->turn=0;
    this->reward=0;
    this->maxDistance=-std::numeric_limits<float>::max();
    this->client.reset();


    Gym::State s;
    this->env->reset(&s);
    for (int i=0; i<this->parametersToObserve; i++) {
        this->state.setDataAt(typeid(float), i, s.observation[i]);
    }
}

std::vector<std::reference_wrapper<const Data::DataHandler>> GymWrapper::getDataSources() {
    auto result = std::vector<std::reference_wrapper<const Data::DataHandler>>({this->state});
    return result;
}

bool GymWrapper::isTerminal() const {
    return false;
}

bool GymWrapper::isCopyable() const {
    return true;
}

Learn::LearningEnvironment *GymWrapper::clone() const {
    return new GymWrapper(*this);
}

double GymWrapper::getScore() const {
    return this->maxDistance;
}

