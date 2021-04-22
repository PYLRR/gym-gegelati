#ifndef TIC_TAC_TOE_WITH_OPPONENT_H
#define TIC_TAC_TOE_WITH_OPPONENT_H

#include <random>

#include <gegelati.h>

#include "include/gym/gym.h"
#include "GymWrapper.h"

/**
 * LearningEnvironment to play the tic tac toe game against a random player.
 * The principle of the tic tac toe is as follows. we have a 3x3 board.
 * Alternatively, each player plays a turn, and has to put a circle (for a
 * player) or a cross (for the other) in a cell. The player succeeding to align
 * 3 of his symbols in a row, column or in diagonal wins
 *
 * In this LearningEnvironment, the trained agent plays against a random algo
 */
class GymWrapper : public Learn::LearningEnvironment {
protected:

  /// turn currently played, used to end the simulation (e.g. we run 100 steps)
  int turn;

  /// reward of the simulation
  float reward;

  /// name of the environment of sim (e.g. "MountainCar-v0")
  std::string envName;

  /// Contains the environment of the simulation
  boost::shared_ptr<Gym::Environment> env;

  /// Contains the current state of the game
  Data::PrimitiveTypeArray<float> state;

  /// Handles connection with Gym server
  boost::shared_ptr<Gym::Client> client;

  /// Randomness control
  Mutator::RNG rng;

  /// maximum reached distance for mountain car
  float maxDistance;

  /// Initializes the env of simulation and returns size of action space
  virtual void initialize(std::string chosenEnv);

  /// nb of parameters from the gym output vector that we want to look at
  int parametersToObserve;

  /// save of action Space, used to make a clone of the LE
  int actionSpaceSize;

  /// save of the environment name, used to make a clone of the LE
  std::string chosenEnv;

public:
  /**
   * Constructor.
   *
   * @param chosenEnv name of the environment of sim (e.g. "MountainCar-v0")
   */
  GymWrapper(std::string chosenEnv, int actionSpaceSize, int nbInputParameters):
  LearningEnvironment(actionSpaceSize), parametersToObserve(nbInputParameters),
  actionSpaceSize(actionSpaceSize), chosenEnv(chosenEnv)
  {
    initialize(chosenEnv);
    this->reset(0);
  };

  /**
   * Copy Constructor
   *
   * Copies the fields of the other GymWrapper, but creates a new gym
   * environment, so that it generates a different session id to communicate
   * with the server.
   *
   * @param other The GymWrapper we want to copy
   */
  GymWrapper(const GymWrapper &other):
  LearningEnvironment(other.actionSpaceSize),
  chosenEnv(other.chosenEnv), actionSpaceSize(other.actionSpaceSize),
  parametersToObserve(other.parametersToObserve), state(other.state)
  {
      initialize(chosenEnv);
      this->reset(0);
  }

  /// Destructor
  ~GymWrapper(){};

  /// Inherited via LearningEnvironment
  virtual void doAction(uint64_t actionID) override;

  /// Inherited via LearningEnvironment
  virtual void
  reset(size_t seed = 0,
        Learn::LearningMode mode = Learn::LearningMode::TRAINING) override;

  /// Inherited via LearningEnvironment
  virtual std::vector<std::reference_wrapper<const Data::DataHandler>>
  getDataSources() override;

  /**
   * Inherited from LearningEnvironment.
   *
   * The score will be directly given by the environment
   */
  double getScore() const override;

  /// Inherited via LearningEnvironment
  virtual bool isTerminal() const override;

  /// Inherited via LearningEnvironment
  virtual bool isCopyable() const override;

  /// Inherited via LearningEnvironment
  virtual LearningEnvironment *clone() const override;


};

#endif