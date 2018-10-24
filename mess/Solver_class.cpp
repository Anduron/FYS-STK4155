#include "Solver_class.h"

using namespace std;
using namespace arma;

Verlet::Verlet()
{
  pos = zeros(3);
  vel = zeros(3);
  M = 0;
}

Verlet::Verlet(Position, Velocity, Acceleration, Mass, steps, planets)
{
  pos = Position;
  vel = Velocity;
  acc = Acceleration
  M = Mass;
  n = steps
  N = planets

}

Verlet::MethodV(mat pos, mat vel, vec mass, )
{






    }
  }

  for (k = 0; k < n; k++){
    prevacc = acc(k)
    curracc = acc(k+1)

    pos(k+1) = pos(k) + dt*vel(k) + dt*(dt/2)*prevacc;
    vel(k+1) = vel(k) + (dt/2)*(curracc+prevacc);
  }

}
