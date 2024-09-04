#include "DCmotor.h"
#include "rccar.h"
//초음파 지정자
const int trigPin = 8;
const int echoPin = 13;
long duration;
int distance;
//모터 지정자
DCmotor dcF_L(2, 3);
DCmotor dcF_R(4, 5);
DCmotor dcB_L(7, 6);
DCmotor dcB_R(19, 10);
RCCar car(dcF_L, dcF_R, dcB_L, dcB_R);
void setup() {
  pinMode(trigPin, OUTPUT); 
  pinMode(echoPin, INPUT); 
  Serial.begin(9600);}
void loop() {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  duration = pulseIn(echoPin, HIGH);
  distance= duration*0.034/2;
  Serial.println(distance);
  if (distance<20) {
    car.backward();
    delay(2000);
    car.stop(); }
  if(Serial.available()>0) {
    int cmd=Serial.read();
    switch(cmd) {
      case 'w': car.forward(); break;
      case 's': car.backward(); break;
      case 'a': car.left(); break;
      case 'd': car.right(); break;
      case 'x': car.stop(); break;
      default: break; } } }