#include "rccar.h"

RCCar::RCCar(DCmotor& FL, DCmotor& FR, DCmotor& BL, DCmotor& BR)
: F_L(FL), F_R(FR), B_L(BL), B_R(BR) {}

void RCCar::forward() {
  F_L.Forward();
  F_R.Forward();
  B_L.Forward();
  B_R.Forward();
}

void RCCar::backward() {
  F_L.Backward();
  F_R.Backward();
  B_L.Backward();
  B_R.Backward();
}

void RCCar::stop() {
  F_L.Stop();
  F_R.Stop();
  B_L.Stop();
  B_R.Stop();
}

void RCCar::left() {
  F_R.Forward();
  B_R.Forward();
}

void RCCar::right() {
  F_L.Forward();
  B_L.Forward();
}