#ifndef RCCAR_H
#define RCCAR_H
#include "DCmotor.h"

class RCCar {
  DCmotor &F_L, &F_R, &B_L, &B_R;
public:
  RCCar(DCmotor& FL, DCmotor& FR, DCmotor& BL, DCmotor& BR);
  void forward();
  void backward();
  void stop();
  void left();
  void right();
};

#endif // RCCAR_H
