#include "Planet_class.h"

using namespace std;
using namespace arma;

Planet::Planet()
{
  pos = zeros(3);
  vel = zeros(3);
  M = 0;
}

Planet::Planet(vec Position, vec Velocity, double Mass)
{
  pos = Position;
  vel = Velocity;
  M = Mass;

}
