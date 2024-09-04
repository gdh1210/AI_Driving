#include "DCmotor.h"

DCmotor::DCmotor(int L, int R) : pinL(L), pinR(R) {
  pinMode(pinL, OUTPUT);
  pinMode(pinR, OUTPUT);
}

void DCmotor::RotateLeft() {
  digitalWrite(pinL, LOW);
  digitalWrite(pinR, HIGH);
}

void DCmotor::RotateRight() {
  digitalWrite(pinL, HIGH);
  digitalWrite(pinR, LOW);
}

void DCmotor::Stop() {
  digitalWrite(pinL, LOW);
  digitalWrite(pinR, LOW);
}

void DCmotor::Forward() {
  RotateLeft();
}

void DCmotor::Backward() {
  RotateRight();
}
