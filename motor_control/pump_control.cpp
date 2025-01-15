#include <AFMotor.h>

AF_DCMotor motor1(1); 
AF_DCMotor motor2(2); 
AF_DCMotor motor3(3); 
AF_DCMotor motor4(4); 

void setup() {
    Serial.begin(115200);
    Serial.println("Motors initialized. Ready for commands.");
    pinMode(LED_BUILTIN, OUTPUT);
}

// Function to control a motor
void controlMotor(AF_DCMotor& motor, int speed, int duration) {
    motor.setSpeed(speed);
    motor.run(FORWARD);
    delay(duration);
    motor.run(RELEASE);
}

void loop() {
    if (Serial.available() > 0) {
        int pumpId = Serial.parseInt(); // Reads multi-byte input
        switch (pumpId) {
            case 10:
                digitalWrite(LED_BUILTIN, HIGH);
                delay(1000);
                digitalWrite(LED_BUILTIN, LOW);
                delay(1000);
                controlMotor(motor1, 50, 3000);
                break;
            case 11:
                controlMotor(motor1, 75, 3000);
                break;
            case 12:
                controlMotor(motor1, 150, 3000);
                break;
            case 20:
                controlMotor(motor2, 50, 3000);
                break;
            case 21:
                controlMotor(motor2, 75, 3000);
                break;
            case 22:
                controlMotor(motor2, 200, 6000);
                break;
            case 2:
                motor2.setSpeed(80);
                motor2.run(FORWARD); 
                motor1.setSpeed(0);
                motor1.run(FORWARD); 
                motor3.setSpeed(0);
                motor3.run(FORWARD); 
                motor4.setSpeed(0);
                motor4.run(FORWARD); 
                break;
            case 3:
                motor3.setSpeed(80);
                motor3.run(FORWARD); 
                motor1.setSpeed(0);
                motor1.run(FORWARD); 
                motor2.setSpeed(0);
                motor2.run(FORWARD); 
                motor4.setSpeed(0);
                motor4.run(FORWARD); 
                break;
            case 4:
                motor4.setSpeed(80);
                motor4.run(FORWARD); 
                motor1.setSpeed(0);
                motor1.run(FORWARD); 
                motor2.setSpeed(0);
                motor2.run(FORWARD); 
                motor3.setSpeed(0);
                motor3.run(FORWARD); 
                break;
           
            default:
                Serial.println("Invalid command!");
                digitalWrite(LED_BUILTIN, HIGH);
                delay(500);
                digitalWrite(LED_BUILTIN, LOW);
        }
    } else {
        motor1.setSpeed(80);
        motor1.run(FORWARD);
        motor2.setSpeed(80);
        motor2.run(FORWARD);
        motor3.setSpeed(80);
        motor3.run(FORWARD);
        motor4.setSpeed(80);
        motor4.run(FORWARD);
    }
}


