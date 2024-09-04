#ifndef DCMOTOR_H
#define DCMOTOR_H

#include <Arduino.h>

class DCmotor {
  int pinL, pinR;
public:
  DCmotor(int L, int R);
  void Forward();
  void Backward();
  void Stop();
  void RotateLeft();
  void RotateRight();
};

#endif // DCMOTOR_H
